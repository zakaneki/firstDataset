import os
import glob
import random
import shutil

def get_base_name(filename):
    """
    Extracts the base name from a filename, removing suffixes like
    '_augmented_X'.
    e.g., 'img_1_augmented_1.png' -> 'img_1'
    e.g., 'img_1.png' -> 'img_1'
    """
    filename_no_ext = os.path.splitext(filename)[0]
    if '_augmented_' in filename_no_ext:
        return filename_no_ext.split('_augmented_')[0]
    return filename_no_ext

def prepare_yolo_dataset(base_dir, validation_split=0.2):
    """
    Prepares the YOLO OBB dataset by splitting augmented images, copying labels,
    and writing correctly formatted train/val files.
    """
    augmented_images_dir = os.path.join(base_dir, 'augmented_images')
    generated_labels_dir = os.path.join(base_dir, 'generated_labels')
    yolo_data_dir = os.path.join(base_dir, 'yolo_data')

    # --- 1. Create YOLO directory structure ---
    dirs = {
        'train_images': os.path.join(yolo_data_dir, 'images', 'train'),
        'val_images': os.path.join(yolo_data_dir, 'images', 'val'),
        'train_labels': os.path.join(yolo_data_dir, 'labels', 'train'),
        'val_labels': os.path.join(yolo_data_dir, 'labels', 'val')
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
        for f in glob.glob(os.path.join(d, '*.*')):
            os.remove(f)

    # --- 2. Get all unique image identifiers ---
    all_image_files = glob.glob(os.path.join(augmented_images_dir, '*.png'))
    unique_base_names = sorted(list(set([
        get_base_name(os.path.basename(f)) for f in all_image_files
    ])))

    # --- 3. Split unique base names into training and validation sets ---
    random.shuffle(unique_base_names)
    num_val = int(len(unique_base_names) * validation_split)
    val_base_names = set(unique_base_names[:num_val])
    train_base_names = set(unique_base_names[num_val:])

    print(f"Total unique base images: {len(unique_base_names)}")
    print(f"Training set size: {len(train_base_names)}")
    print(f"Validation set size: {len(val_base_names)}")

    # --- 4. Process and copy files ---
    train_image_paths = []
    val_image_paths = []

    for img_path in all_image_files:
        img_filename = os.path.basename(img_path)
        base_name = get_base_name(img_filename)
        label_filename = f"{base_name}.txt"
        label_path = os.path.join(generated_labels_dir, label_filename)

        if not os.path.exists(label_path):
            print(f"Warning: Label not found for {img_filename}, skipping.")
            continue

        if base_name in val_base_names:
            target_img_dir = dirs['val_images']
            target_lbl_dir = dirs['val_labels']
            val_image_paths.append(os.path.join('images', 'val', img_filename))
        else:
            target_img_dir = dirs['train_images']
            target_lbl_dir = dirs['train_labels']
            train_image_paths.append(os.path.join('images', 'train', img_filename))
        
        shutil.copy(img_path, os.path.join(target_img_dir, img_filename))
        shutil.copy(label_path, os.path.join(target_lbl_dir, img_filename.replace('.png', '.txt')))
        
    # --- 5. Create train.txt and val.txt with fully corrected paths ---
    def write_yolo_file(file_path, image_paths):
        # Format paths with forward slashes and './' prefix for YOLO compatibility
        formatted_paths = []
        for p in image_paths:
            # Replace backslashes with forward slashes
            path = p.replace('\\', '/')
            # Add './' prefix if it's not there
            if not path.startswith('./'):
                path = './' + path
            formatted_paths.append(path)

        with open(file_path, 'w') as f:
            f.write("\n".join(formatted_paths))
            
    write_yolo_file(os.path.join(yolo_data_dir, 'train.txt'), train_image_paths)
    write_yolo_file(os.path.join(yolo_data_dir, 'val.txt'), val_image_paths)
        
    print("\nDataset preparation complete.")
    print(f"Train/Val text files created at: {yolo_data_dir}")

if __name__ == '__main__':
    prepare_yolo_dataset('.')
