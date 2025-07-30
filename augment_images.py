import cv2
import albumentations as A
import os
import shutil

# Define your augmentation pipeline
transform = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
    A.GaussNoise(std_range=(0.1, 0.2), p=0.5),
    A.MotionBlur(blur_limit=7, p=0.5),
    A.RandomGamma(gamma_limit=(80, 120), p=0.5),
])

# Input and output directories
input_dir = 'generated_images/'
output_dir = 'augmented_images/'

# --- Delete previously augmented images ---
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)

# List all files in the input directory
image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Loop through each image and apply augmentations
num_augmentations = 3  # Number of augmentations per image

for image_file in image_files:
    image_path = os.path.join(input_dir, image_file)
    image = cv2.imread(image_path)
    file_root, file_ext = os.path.splitext(image_file)

    # --- Copy the original image with its original name ---
    shutil.copy(image_path, os.path.join(output_dir, image_file))

    for i in range(num_augmentations):
        augmented = transform(image=image)
        augmented_image = augmented['image']
        # Save augmented images with a suffix
        output_path = os.path.join(
            output_dir, f'{file_root}_augmented_{i+1}{file_ext}'
        )
        cv2.imwrite(output_path, augmented_image)

print(f'Successfully augmented {len(image_files)} images, {num_augmentations} times each, and saved them to {output_dir}')
