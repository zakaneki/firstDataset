import cv2
import numpy as np
import random
import os
import glob

def add_lines_avoiding_symbols(image, symbol_masks, num_lines=15, color=(0, 0, 0), thickness=1, max_attempts=100):
    """
    Draws random lines on an image, avoiding a list of masked areas for symbols
    and also avoiding intersecting with other newly drawn lines.
    """
    if not symbol_masks:
        # If there are no symbols, just draw lines anywhere
        combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    else:
        # Combine all individual symbol masks into a single mask
        combined_mask = np.zeros_like(symbol_masks[0])
        for mask in symbol_masks:
            combined_mask = cv2.bitwise_or(combined_mask, mask)

    # --- Create a dilation kernel ---
    # This determines the size of the "keep-out" buffer zone.
    # A 7x7 kernel creates a 3-pixel buffer, which is very safe.
    kernel = np.ones((7, 7), np.uint8)
    
    # Create the initial keep-out zone by dilating the symbols mask
    combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)

    h, w = image.shape[:2]
    lines_drawn = 0
    attempts = 0

    while lines_drawn < num_lines and attempts < max_attempts * num_lines:
        attempts += 1
        
        x1, y1 = random.randint(0, w - 1), random.randint(0, h - 1)
        x2, y2 = random.randint(0, w - 1), random.randint(0, h - 1)

        # Create a temporary mask for the line to check for overlap
        line_check_mask = np.zeros_like(combined_mask)
        cv2.line(line_check_mask, (x1, y1), (x2, y2), 255, thickness)

        # Create a buffer zone for the PROPOSED line
        buffered_line_mask = cv2.dilate(line_check_mask, kernel, iterations=1)

        # Check if the line intersects with any symbol
        if np.any(cv2.bitwise_and(combined_mask, buffered_line_mask)):
            continue  # Line intersects with a symbol, so we skip it and try again

        # If the line is clear, draw it on the image
        cv2.line(image, (x1, y1), (x2, y2), color, thickness)

        # Add the new line to the combined mask to avoid future intersections
        combined_mask = cv2.bitwise_or(combined_mask, buffered_line_mask)

        lines_drawn += 1

    return image, combined_mask

def add_random_text(image, existing_elements_mask, num_texts=5, max_attempts=50):
    """
    Adds random text to an image, avoiding existing elements (symbols and lines).
    """
    h, w = image.shape[:2]
    
    # --- Define text properties ---
    fonts = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_PLAIN, cv2.FONT_HERSHEY_DUPLEX]
    short_texts = ["T1", "CB-A", "SW-42", "FDR-1", "V2", "P-3"]
    long_texts = ["Substation", "Control", "Phase A", "Auxiliary", "Main Bus", "Feeder"]
    
    kernel = np.ones((7, 7), np.uint8)
    texts_drawn = 0
    attempts = 0
    
    while texts_drawn < num_texts and attempts < max_attempts * num_texts:
        attempts += 1

        # --- Choose random text properties ---
        font = random.choice(fonts)
        font_scale = random.uniform(0.6, 1.2)
        thickness = random.randint(1, 2)
        color = (random.randint(0, 50), random.randint(0, 50), random.randint(0, 50))
        
        # Randomly choose between short and long text
        text = random.choice(short_texts if random.random() < 0.5 else long_texts)
        
        # Get text size to create a bounding box
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        if w - text_w <= 0 or h - text_h <= 0:
            continue
        
        # Find a random origin for the text
        x = random.randint(0, w - text_w)
        y = random.randint(text_h, h - baseline) # Ensure text is fully visible

        # Create a mask for the text's bounding box
        text_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.rectangle(text_mask, (x, y - text_h), (x + text_w, y + baseline), 255, -1)
        
        # Dilate to create a buffer zone
        buffered_text_mask = cv2.dilate(text_mask, kernel, iterations=1)

        # Check for overlap with existing elements
        if np.any(cv2.bitwise_and(existing_elements_mask, buffered_text_mask)):
            continue

        # Draw the text and update the mask
        cv2.putText(image, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
        existing_elements_mask = cv2.bitwise_or(existing_elements_mask, buffered_text_mask)
        texts_drawn += 1

    return image

def extract_symbols(image_path, label_path):
    """
    Extracts symbols from an image based on YOLO-like segmentation labels,
    preserving the class index for each symbol.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return [], (0, 0)
    
    h, w = image.shape[:2]
    
    symbols_with_classes = []
    if not os.path.exists(label_path):
        print(f"Error: Label file not found at {label_path}")
        return [], (w, h)

    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 9:
            continue
        
        class_index = int(parts[0])
        
        # Parse polygon points
        poly_norm = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)
        poly = (poly_norm * np.array([w, h])).astype(np.int32)
        
        # Create a mask for the symbol
        rect = cv2.boundingRect(poly)
        x, y, rect_w, rect_h = rect
        
        # Adjust polygon coordinates to be relative to the bounding box crop
        poly_local = poly - np.array([x, y])

        # Crop the symbol and create a 4-channel version for transparency
        cropped_symbol_bgr = image[y:y+rect_h, x:x+rect_w]
        cropped_symbol_bgra = cv2.cvtColor(cropped_symbol_bgr, cv2.COLOR_BGR2BGRA)
        
        # Create a mask for the polygon area
        mask = np.zeros((rect_h, rect_w), dtype=np.uint8)
        cv2.fillPoly(mask, [poly_local], (255, 255, 255))
        
        # Apply the mask to the alpha channel
        cropped_symbol_bgra[:, :, 3] = mask
        
        symbols_with_classes.append((class_index, cropped_symbol_bgra))
        
    return symbols_with_classes, (w, h)

def check_overlap(new_rect, placed_rects):
    """
    Checks if a new rectangle overlaps with any of the placed rectangles.
    """
    x1, y1, w1, h1 = new_rect
    for (x2, y2, w2, h2) in placed_rects:
        if not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1):
            return True
    return False

def place_symbols_randomly(symbols_with_classes, canvas_size=(1024, 1024), max_attempts=100):
    """
    Places symbols randomly on a new canvas, avoiding overlaps,
    and generates corresponding YOLO OBB labels.
    """
    canvas = np.ones((canvas_size[1], canvas_size[0], 4), dtype=np.uint8) * 255
    placed_rects = []
    placed_masks = []
    labels = []
    
    canvas_w, canvas_h = canvas_size

    for class_index, symbol in symbols_with_classes:
        placed = False
        for _ in range(max_attempts):
            # Random scale
            scale = random.uniform(0.7, 1.3)
            h, w = symbol.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            
            if new_h == 0 or new_w == 0:
                continue

            resized_symbol = cv2.resize(symbol, (new_w, new_h))
            
            # Random rotation
            angle = random.uniform(0, 360)
            center = (new_w // 2, new_h // 2)
            rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # Calculate new bounding box after rotation
            cos = np.abs(rot_mat[0, 0])
            sin = np.abs(rot_mat[0, 1])
            out_w = int((new_h * sin) + (new_w * cos))
            out_h = int((new_h * cos) + (new_w * sin))
            
            # Adjust rotation matrix to account for new dimensions
            rot_mat[0, 2] += (out_w / 2) - center[0]
            rot_mat[1, 2] += (out_h / 2) - center[1]
            
            rotated_symbol = cv2.warpAffine(resized_symbol, rot_mat, (out_w, out_h))
            
            # Random position
            if canvas_size[0] - out_w <= 0 or canvas_size[1] - out_h <= 0:
                continue
            
            rand_x = random.randint(0, canvas_size[0] - out_w)
            rand_y = random.randint(0, canvas_size[1] - out_h)
            
            new_rect = (rand_x, rand_y, out_w, out_h)
            
            if not check_overlap(new_rect, placed_rects):
                # Separate alpha channel and color channels for pasting
                alpha = rotated_symbol[:, :, 3] / 255.0
                color = rotated_symbol[:, :, :3]
                
                # Blend the symbol onto the canvas
                for c in range(0, 3):
                    canvas[rand_y:rand_y+out_h, rand_x:rand_x+out_w, c] = \
                        alpha * color[:, :, c] + \
                        (1 - alpha) * canvas[rand_y:rand_y+out_h, rand_x:rand_x+out_w, c]
                
                # Create and store a full-size mask for the placed symbol
                symbol_alpha_mask = rotated_symbol[:, :, 3]
                full_size_symbol_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
                full_size_symbol_mask[rand_y:rand_y+out_h, rand_x:rand_x+out_w] = symbol_alpha_mask
                placed_masks.append(full_size_symbol_mask)

                placed_rects.append(new_rect)
                
                # --- Generate YOLO OBB Label ---
                # Find the contour of the placed symbol from its alpha mask
                contours, _ = cv2.findContours(rotated_symbol[:, :, 3], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    main_contour = max(contours, key=cv2.contourArea)
                    
                    # Offset the contour by its placement position on the canvas
                    main_contour[:, :, 0] += rand_x
                    main_contour[:, :, 1] += rand_y
                    
                    # Get the minimum area rotated rectangle
                    obb = cv2.minAreaRect(main_contour)
                    
                    # Get the 4 corner points of the bounding box
                    box = cv2.boxPoints(obb) # (4, 2) array of floats
                    
                    # Normalize the coordinates
                    box[:, 0] /= canvas_w
                    box[:, 1] /= canvas_h
                    
                    # Flatten the points and format the label string
                    points = box.flatten()
                    label_str = f"{class_index} {points[0]:.6f} {points[1]:.6f} {points[2]:.6f} {points[3]:.6f} {points[4]:.6f} {points[5]:.6f} {points[6]:.6f} {points[7]:.6f}"
                    labels.append(label_str)

                placed = True
                break
        
        if not placed:
            print("Could not place a symbol after max attempts. It might be too large or the canvas too full.")

    # --- Start with a mask of all placed symbols ---
    if placed_masks:
        combined_mask = np.zeros_like(placed_masks[0])
        for mask in placed_masks:
            combined_mask = cv2.bitwise_or(combined_mask, mask)
    else:
        combined_mask = np.zeros((canvas_size[1], canvas_size[0]), dtype=np.uint8)

    line_thickness = random.randint(1, 3)
    num_lines = random.randint(4, 20)
    # After placing all symbols, add circuit lines that avoid them
    canvas, combined_mask = add_lines_avoiding_symbols(canvas, [combined_mask], thickness=line_thickness, num_lines=num_lines)
    canvas = add_random_text(canvas, combined_mask, num_texts=random.randint(3, 9))
    return canvas, labels

def main():
    """
    Main function to run the symbol extraction and placement process.
    """
    image_path = 'images/substation-with-single-transformer.png'
    
    # Construct label path from image path
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    label_path = os.path.join('labels', 'train', f'{base_name}.txt')

    print("Extracting symbols...")
    all_symbols_with_classes, original_size = extract_symbols(image_path, label_path)

    if not all_symbols_with_classes:
        print("No symbols were extracted. Exiting.")
        return

    print(f"Extracted {len(all_symbols_with_classes)} base symbols.")
    
    num_images_to_generate = 150
    print(f"Generating {num_images_to_generate} new images with labels...")

    # Ensure the output directories exist and are clean
    output_img_dir = 'generated_images'
    output_lbl_dir = 'generated_labels'
    for d in [output_img_dir, output_lbl_dir]:
        os.makedirs(d, exist_ok=True)
        for f in glob.glob(os.path.join(d, '*.*')):
            os.remove(f)

    for i in range(num_images_to_generate):
        # Determine a random number of symbols to place in the new image
        num_symbols_to_place = random.randint(5, len(all_symbols_with_classes) * 2)
        
        # Randomly select symbols from the extracted list (with replacement)
        symbols_for_this_image = random.choices(all_symbols_with_classes, k=num_symbols_to_place)

        print(f"\n--- Creating image {i+1}/{num_images_to_generate} with {num_symbols_to_place} symbols ---")
        
        canvas_size = (original_size[0], original_size[1])
        result_image, labels = place_symbols_randomly(symbols_for_this_image, canvas_size=canvas_size)
        
        # Save the generated image
        output_img_filename = os.path.join(output_img_dir, f'randomly_placed_symbols_{i+1}.png')
        cv2.imwrite(output_img_filename, result_image)
        print(f"Successfully created image: {output_img_filename}")

        # Save the corresponding labels
        output_lbl_filename = os.path.join(output_lbl_dir, f'randomly_placed_symbols_{i+1}.txt')
        with open(output_lbl_filename, 'w') as f:
            f.write("\n".join(labels))
        print(f"Successfully created labels: {output_lbl_filename}")


if __name__ == '__main__':
    main() 