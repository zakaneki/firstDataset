import cv2
import numpy as np
import random
import os
import glob
import heapq

def create_grid(canvas_size, grid_size):
    """Create a grid for pathfinding."""
    h, w = canvas_size
    gh, gw = h // grid_size, w // grid_size
    return np.zeros((gh, gw), dtype=np.uint8)

def mark_rect_on_grid(grid, rect, grid_size, margin=0):
    """Mark a rectangle as blocked on the grid."""
    x, y, w, h = rect
    gx1, gy1 = max(0, (x - margin) // grid_size), max(0, (y - margin) // grid_size)
    gx2, gy2 = min(grid.shape[1]-1, (x + w + margin) // grid_size), min(grid.shape[0]-1, (y + h + margin) // grid_size)
    grid[gy1:gy2+1, gx1:gx2+1] = 1

def mark_path_on_grid(grid, path, junctions):
    """Mark bends and endpoint as junctions, mark path as shared (value 2)."""
    for idx, (x, y) in enumerate(path):
        # Always add the endpoint as a junction
        if idx == len(path) - 1:
            junctions.add((x, y))
            continue
        # Add bends as junctions
        if 0 < idx < len(path) - 1:
            prev = path[idx - 1]
            curr = path[idx]
            nxt = path[idx + 1]
            # If direction changes, it's a bend
            if (curr[0] - prev[0], curr[1] - prev[1]) != (nxt[0] - curr[0], nxt[1] - curr[1]):
                junctions.add((x, y))
                continue
        # Mark as shared path (value 2)
        grid[y, x] = 2

def point_to_grid(pt, grid_size):
    return (pt[0] // grid_size, pt[1] // grid_size)

def grid_to_point(gpt, grid_size):
    return (gpt[0] * grid_size + grid_size // 2, gpt[1] * grid_size + grid_size // 2)

def soften_l_bend_path(path, min_offset=3, max_offset=7):
    """Move bend points away from endpoints to the middle of the path."""
    if len(path) < 3:
        return path

    # Find the index where the direction changes (the bend)
    bend_idx = None
    for i in range(1, len(path)-1):
        dx1 = path[i][0] - path[i-1][0]
        dy1 = path[i][1] - path[i-1][1]
        dx2 = path[i+1][0] - path[i][0]
        dy2 = path[i+1][1] - path[i][1]
        if (dx1, dy1) != (dx2, dy2):
            bend_idx = i
            break
    if bend_idx is None:
        return path

    # Move the bend closer to the middle
    offset = random.randint(min_offset, max_offset)
    new_bend_idx = max(min(len(path) - 2, bend_idx + offset), 1)
    if path[bend_idx-1][0] == path[bend_idx][0]:  # vertical first
        new_bend = (path[bend_idx][0], path[new_bend_idx][1])
    else:  # horizontal first
        new_bend = (path[new_bend_idx][0], path[bend_idx][1])

    new_path = path[:new_bend_idx]
    new_path.append(new_bend)
    new_path.extend(path[new_bend_idx+1:])
    return new_path

def astar(grid, start, goal, junctions):
    """A* pathfinding with support for shared paths."""
    h, w = grid.shape
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    directions = [(-1,0), (1,0), (0,-1), (0,1)]

    def heuristic(a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]
        for dx, dy in directions:
            neighbor = (current[0]+dx, current[1]+dy)
            if 0 <= neighbor[0] < w and 0 <= neighbor[1] < h:
                cell_val = grid[neighbor[1], neighbor[0]]
                if cell_val == 1 and neighbor not in junctions and neighbor != goal:
                    continue
                # Lower cost for shared path
                move_cost = 1 if cell_val != 2 else 0.5
                tentative_g = g_score[current] + move_cost
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f, neighbor))
                    came_from[neighbor] = current
    return None

def place_symbols_with_pathfinding(symbols_with_classes, canvas_size=(1024, 1024), max_attempts=100):
    """Place symbols with pathfinding connections between them."""
    canvas = np.ones((canvas_size[1], canvas_size[0], 4), dtype=np.uint8) * 255
    placed_symbols = []
    placed_masks = []
    labels = []
    
    canvas_w, canvas_h = canvas_size
    grid_size = 4
    grid = create_grid(canvas_size, grid_size)
    junctions = set()
    # Create a separate line mask
    line_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    # Place symbols first
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
            
            # Random rotation (0, 90, 180, or 270 degrees for horizontal/vertical)
            angle = random.choice([0, 90, 180, 270])
            center = (new_w // 2, new_h // 2)
            rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # Calculate new bounding box after rotation
            cos = np.abs(rot_mat[0, 0])
            sin = np.abs(rot_mat[0, 1])
            out_w = int((new_h * sin) + (new_w * cos))
            out_h = int((new_h * cos) + (new_w * sin))
            
            # Adjust rotation matrix
            rot_mat[0, 2] += (out_w / 2) - center[0]
            rot_mat[1, 2] += (out_h / 2) - center[1]
            
            rotated_symbol = cv2.warpAffine(resized_symbol, rot_mat, (out_w, out_h))
            
            # Random position
            if canvas_size[0] - out_w <= 0 or canvas_size[1] - out_h <= 0:
                continue
            
            rand_x = random.randint(0, canvas_size[0] - out_w)
            rand_y = random.randint(0, canvas_size[1] - out_h)
            
            new_rect = (rand_x, rand_y, out_w, out_h)
            
            # Check overlap with existing symbols
            overlap = False
            for existing_rect, _, _, _, _, _, _ in placed_symbols:
                if not (rand_x + out_w < existing_rect[0] or existing_rect[0] + existing_rect[2] < rand_x or
                       rand_y + out_h < existing_rect[1] or existing_rect[1] + existing_rect[3] < rand_y):
                    overlap = True
                    break
            
            if not overlap:
                # Determine orientation based on rotation
                orientation = 'horizontal' if angle in [90, 270] else 'vertical'
                
                # Calculate anchors based on orientation
                anchors = calculate_anchors_for_symbol(rotated_symbol, new_rect, orientation)
                
                # Blend symbol onto canvas
                alpha = rotated_symbol[:, :, 3] / 255.0
                color = rotated_symbol[:, :, :3]
                
                for c in range(3):
                    canvas[rand_y:rand_y+out_h, rand_x:rand_x+out_w, c] = \
                        alpha * color[:, :, c] + \
                        (1 - alpha) * canvas[rand_y:rand_y+out_h, rand_x:rand_x+out_w, c]
                
                # Create mask
                symbol_alpha_mask = rotated_symbol[:, :, 3]
                full_size_symbol_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
                full_size_symbol_mask[rand_y:rand_y+out_h, rand_x:rand_x+out_w] = symbol_alpha_mask
                placed_masks.append(full_size_symbol_mask)

                # Store symbol info for pathfinding
                placed_symbols.append((new_rect, anchors, {'used_anchors': []}, class_index, rotated_symbol, rand_x, rand_y))
                placed = True
                break
        
        if not placed:
            print(f"Could not place symbol {class_index} after max attempts.")

    # Mark symbol areas as obstacles in the grid
    for rect, _, _, _, _, _, _ in placed_symbols:
        mark_rect_on_grid(grid, rect, grid_size, margin=grid_size)

    # Build connections between symbols
    num_symbols = len(placed_symbols)
    if num_symbols > 1:
        # Create a spanning tree to ensure connectivity
        nodes = list(range(num_symbols))
        random.shuffle(nodes)
        edges = []
        for i in range(1, num_symbols):
            j = random.randint(0, i - 1)
            edges.append((nodes[i], nodes[j]))
        
        # Track which symbols are connected
        connected_symbols = set()

        # Try to connect symbols with paths
        for i, j in edges:
            symbol1_rect, symbol1_anchors, symbol1_data, _, _, _, _ = placed_symbols[i]
            symbol2_rect, symbol2_anchors, symbol2_data, _, _, _, _ = placed_symbols[j]
            
            # Try different anchor combinations
            found_path = False
            for p1 in symbol1_anchors:
                for p2 in symbol2_anchors:
                    if p1 in symbol1_data['used_anchors'] or p2 in symbol2_data['used_anchors']:
                        continue
                    
                    g_start = point_to_grid(p1, grid_size)
                    g_goal = point_to_grid(p2, grid_size)
                    
                    # Check if grid coordinates are within bounds
                    h, w = grid.shape
                    if (g_start[0] < 0 or g_start[0] >= w or g_start[1] < 0 or g_start[1] >= h or
                        g_goal[0] < 0 or g_goal[0] >= w or g_goal[1] < 0 or g_goal[1] >= h):
                        print(f"Skipping connection between symbols {i} and {j} because grid coordinates are out of bounds")
                        continue

                    # Temporarily clear start and goal positions
                    original_start_val = grid[g_start[1], g_start[0]]
                    original_goal_val = grid[g_goal[1], g_goal[0]]
                    grid[g_start[1], g_start[0]] = 0
                    grid[g_goal[1], g_goal[0]] = 0
                    
                    path = astar(grid, g_start, g_goal, junctions)
                    
                    # Restore original values
                    grid[g_start[1], g_start[0]] = original_start_val
                    grid[g_goal[1], g_goal[0]] = original_goal_val
                    
                    if path is not None:
                        # Soften the bend
                        path = soften_l_bend_path(path, min_offset=3, max_offset=7)
                        
                        # Mark anchors as used
                        symbol1_data['used_anchors'].append(p1)
                        symbol2_data['used_anchors'].append(p2)
                        
                        # Draw the path
                        pts = [grid_to_point(gpt, grid_size) for gpt in path]

                        # Create a mask for the lines being drawn
                        temp_line_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
                        for k in range(len(pts) - 1):
                            cv2.line(canvas, pts[k], pts[k+1], (0, 0, 255), 2)
                            # Also draw on the mask with a thicker line to create buffer
                            cv2.line(temp_line_mask, pts[k], pts[k+1], 255, 4)  # Thicker for buffer
                        
                        # Dilate the line mask to create a buffer zone
                        kernel = np.ones((7, 7), np.uint8)
                        temp_line_mask = cv2.dilate(temp_line_mask, kernel, iterations=1)
                        
                        # Add to the main line mask
                        line_mask = cv2.bitwise_or(line_mask, temp_line_mask)
                        
                        # Mark path on grid
                        mark_path_on_grid(grid, path, junctions)
                        # Mark both symbols as connected
                        connected_symbols.add(i)
                        connected_symbols.add(j)
                        found_path = True
                        break
                
                if found_path:
                    break
            
            if not found_path:
                print(f"No path found between symbols {i} and {j}")

    # Remove unconnected symbols and regenerate canvas
    if connected_symbols:
        # White out unconnected symbols from the existing canvas
        for idx in range(len(placed_symbols)):
            if idx not in connected_symbols:
                rect, _, _, _, _, rand_x, rand_y = placed_symbols[idx]
                x, y, w, h = rect
                
                # White out the symbol area
                canvas[rand_y:rand_y+h, rand_x:rand_x+w] = [255, 255, 255, 255]
        
        # Generate labels only for connected symbols
        new_labels = []
        new_placed_masks = []  # Create new mask list with only connected symbols

        for idx in sorted(connected_symbols):
            rect, _, _, class_index, rotated_symbol, rand_x, rand_y = placed_symbols[idx]
            x, y, w, h = rect
            
            # Generate YOLO OBB label for connected symbol
            contours, _ = cv2.findContours(rotated_symbol[:, :, 3], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                main_contour = max(contours, key=cv2.contourArea)
                main_contour[:, :, 0] += rand_x
                main_contour[:, :, 1] += rand_y
                
                obb = cv2.minAreaRect(main_contour)
                box = cv2.boxPoints(obb)
                box[:, 0] /= canvas_w
                box[:, 1] /= canvas_h
                
                points = box.flatten()
                label_str = f"{class_index} {points[0]:.6f} {points[1]:.6f} {points[2]:.6f} {points[3]:.6f} {points[4]:.6f} {points[5]:.6f} {points[6]:.6f} {points[7]:.6f}"
                new_labels.append(label_str)
            
            # Add only connected symbol masks to the new list
            if idx < len(placed_masks):
                new_placed_masks.append(placed_masks[idx])

        # Create a mask of all connected symbols and paths for text placement
        if new_placed_masks:
            # Create a mask from connected symbols only
            connected_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
            for idx in sorted(connected_symbols):
                if idx < len(placed_masks):
                    connected_mask = cv2.bitwise_or(connected_mask, placed_masks[idx])
            
            # Dilate the mask to include path areas
            kernel = np.ones((15, 15), np.uint8)  # Larger kernel to include paths
            connected_mask = cv2.dilate(connected_mask, kernel, iterations=1)
            
            # Add the line mask
            connected_mask = cv2.bitwise_or(connected_mask, line_mask)
        else:
            connected_mask = line_mask.copy()
        
        # Add random text that avoids connected symbols and paths
        canvas = add_random_text(canvas, connected_mask, num_texts=random.randint(3, 9))
        
        return canvas, new_labels
    else:
        # No connections found, return empty canvas
        return canvas, []

def calculate_anchors_for_symbol(symbol_img, rect, orientation):
    """Calculate anchor points for a symbol based on its orientation."""
    x, y, w, h = rect
    cx, cy = x + w // 2, y + h // 2
    
    if orientation == 'horizontal':
        # For horizontal orientation, anchors are on left and right sides
        # Place anchors outside the symbol bounds for better pathfinding
        anchors = [(x - 10, cy), (x + w + 10, cy)]  # 10 pixels outside edges
    else:  # vertical
        # For vertical orientation, anchors are on top and bottom
        # Place anchors outside the symbol bounds for better pathfinding
        anchors = [(cx, y - 10), (cx, y + h + 10)]  # 10 pixels outside edges
    
    return anchors

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
    # --- Source directories for original images and labels ---
    source_image_dir = 'images'
    source_label_dir = os.path.join('labels', 'train')

    print("Searching for images to process...")

    # Find all supported images (png, jpg, jpeg) in the source directory
    source_image_paths = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        source_image_paths.extend(glob.glob(os.path.join(source_image_dir, ext)))
    
    if not source_image_paths:
        print(f"No source images found in '{source_image_dir}'. Exiting.")
        return

    print(f"Found {len(source_image_paths)} source images.")
    
    # --- Extract symbols from ALL source images ---
    all_symbols_with_classes = []
    original_size = (1024, 1024) # Default size, will be updated by the first image

    print("\nExtracting symbols from all source images...")
    for image_path in source_image_paths:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        label_path = os.path.join(source_label_dir, f'{base_name}.txt')

        if not os.path.exists(label_path):
            print(f"Warning: Label file not found for {image_path}. Skipping.")
            continue
            
        print(f" - Processing {os.path.basename(image_path)}")
        symbols, size = extract_symbols(image_path, label_path)
        if symbols:
            all_symbols_with_classes.extend(symbols)
            original_size = size # Use the size from the last processed image

    if not all_symbols_with_classes:
        print("No symbols were extracted. Exiting.")
        return

    print(f"Extracted {len(all_symbols_with_classes)} base symbols.")
    
    num_images_to_generate = 20
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
        num_symbols_to_place = random.randint(5, min(30, len(all_symbols_with_classes))) # Avoid placing too many
        
        # Randomly select symbols from the extracted list (with replacement)
        symbols_for_this_image = random.choices(all_symbols_with_classes, k=num_symbols_to_place)

        print(f"\n--- Creating image {i+1}/{num_images_to_generate} with {num_symbols_to_place} symbols ---")
        
        result_image, labels = place_symbols_with_pathfinding(symbols_for_this_image, canvas_size=original_size)
        
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