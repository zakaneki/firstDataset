import cv2
import numpy as np
import random
import heapq

def create_grid(canvas_size, grid_size):
    """Create a grid for pathfinding."""
    h, w = canvas_size
    gh, gw = h // grid_size, w // grid_size
    return np.zeros((gh, gw), dtype=np.uint8)

def mark_rect_on_grid(grid, rect, grid_size, margin = 0):
    """Mark a rectangle as blocked on the grid."""
    x, y, w, h = rect
    gx1, gy1 = max(0, (x - margin) // grid_size), max(0, (y - margin) // grid_size)
    gx2, gy2 = min(grid.shape[1]-1, (x + w + margin) // grid_size), min(grid.shape[0]-1, (y + h + margin) // grid_size)
    grid[gy1:gy2+1, gx1:gx2+1] = 1

def mark_path_on_grid(grid, path, junctions):
    """Mark a path as blocked on the grid."""
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

import random

def soften_l_bend_path(path, min_offset=2, max_offset=6):
    """
    Given a path (list of grid points), if it's an L-shape, move the bend point
    away from the endpoint by a random offset.
    """
    if len(path) < 3:
        return path  # Too short to soften

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
        return path  # No bend found

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
    h, w = grid.shape
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    directions = [(-1,0), (1,0), (0,-1), (0,1)]  # 4-way movement

    def heuristic(a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])  # Manhattan

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            # reconstruct path
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
                    continue  # Blocked
                # Lower cost for shared path
                move_cost = 1 if cell_val != 2 else 0.5
                tentative_g = g_score[current] + move_cost
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f, neighbor))
                    came_from[neighbor] = current
    return None  # No path found
    
def place_symbols(num_symbols, canvas_size, symbol_size):
    placed = []
    for _ in range(num_symbols):
        for _ in range(100): # Max attempts
            x = random.randint(0, canvas_size[0] - symbol_size[0])
            y = random.randint(0, canvas_size[1] - symbol_size[1])
            rect = (x, y, symbol_size[0], symbol_size[1])
            
            # Check for overlap
            is_overlapping = any(
                not (x + symbol_size[0] < s['rect'][0] or s['rect'][0] + s['rect'][2] < x or
                     y + symbol_size[1] < s['rect'][1] or s['rect'][1] + s['rect'][3] < y)
                for s in placed
            )
            
            if not is_overlapping:
                orientation = random.choice(['horizontal', 'vertical'])
                cx, cy = x + rect[2] // 2, y + rect[3] // 2
                if orientation == 'horizontal':
                    anchors = [(x, cy), (x + rect[2], cy)]
                else: # vertical
                    anchors = [(cx, y), (cx, y + rect[3])]
                
                placed.append({
                    'rect': rect,
                    'orientation': orientation,
                    'anchors': anchors,
                    'used_anchors': []
                })
                break
    return placed

def build_connections(num_symbols):
    # Build a random spanning tree to ensure connectivity
    nodes = list(range(num_symbols))
    random.shuffle(nodes)
    edges = []
    for i in range(1, num_symbols):
        j = random.randint(0, i - 1)
        edges.append((nodes[i], nodes[j]))
    # Optionally add extra random edges
    return edges

def rect_contains_segment(rect, p_start, p_end):
    # Check if a horizontal or vertical segment overlaps a rectangle
    x, y, w, h = rect
    x1, y1 = p_start
    x2, y2 = p_end
    if x1 == x2:  # vertical
        if x <= x1 <= x + w:
            y_min, y_max = sorted([y1, y2])
            return not (y_max < y or y_min > y + h)
    elif y1 == y2:  # horizontal
        if y <= y1 <= y + h:
            x_min, x_max = sorted([x1, x2])
            return not (x_max < x or x_min > x + w)
    return False

def draw_smart_l_shaped_line(img, p1, p2, rect1, rect2, color, thickness):
    # Two possible bends
    mid1 = (p2[0], p1[1])  # horizontal then vertical
    mid2 = (p1[0], p2[1])  # vertical then horizontal

    # Check for overlap with either symbol
    overlap1 = (
        rect_contains_segment(rect1, p1, mid1) or
        rect_contains_segment(rect2, mid1, p2)
    )
    overlap2 = (
        rect_contains_segment(rect1, p1, mid2) or
        rect_contains_segment(rect2, mid2, p2)
    )

    # Prefer the path with no overlap
    if not overlap1:
        cv2.line(img, p1, mid1, color, thickness)
        cv2.line(img, mid1, p2, color, thickness)
    elif not overlap2:
        cv2.line(img, p1, mid2, color, thickness)
        cv2.line(img, mid2, p2, color, thickness)
    else:
        # If both overlap, just pick one (or add more logic)
        cv2.line(img, p1, mid1, color, thickness)
        cv2.line(img, mid1, p2, color, thickness)

def select_anchors_hardcoded(symbol1, symbol2):
    """Select the best pair of available anchors between two symbols."""
    available_anchors1 = [a for a in symbol1['anchors'] if a not in symbol1['used_anchors']]
    if not available_anchors1: available_anchors1 = symbol1['anchors']

    available_anchors2 = [a for a in symbol2['anchors'] if a not in symbol2['used_anchors']]
    if not available_anchors2: available_anchors2 = symbol2['anchors']

    pairs = []
    for a1 in available_anchors1:
        for a2 in available_anchors2:
            dist = (a1[0] - a2[0])**2 + (a1[1] - a2[1])**2
            pairs.append(((a1, a2), dist))
    # Sort by distance
    pairs.sort(key=lambda x: x[1])
    return [pair for pair, _ in pairs]

def get_junction_for_endpoint(endpoint, endpoint_junctions):
    # If this endpoint already has a path, return the last bend cell
    if endpoint in endpoint_junctions and endpoint_junctions[endpoint]:
        # Get the last path added
        _, existing_path = endpoint_junctions[endpoint][-1]
        # Find the last bend before the endpoint
        for i in range(len(existing_path)-2, 0, -1):
            if (existing_path[i][0] != existing_path[i+1][0] and
                existing_path[i][1] != existing_path[i+1][1]):
                return existing_path[i+1]
        # If no bend, just return the cell before the endpoint
        return existing_path[-2]
    else:
        return endpoint  # No previous connection, use anchor

def main():
    canvas_size = (800, 800)
    symbol_size = (60, 60)
    num_symbols = 8
    # Map: endpoint grid cell -> list of (element index, path)
    junctions = set()
    img = np.ones((canvas_size[1], canvas_size[0], 3), dtype=np.uint8) * 255

    # 1. Place symbols
    symbols = place_symbols(num_symbols, canvas_size, symbol_size)

    # 2. Draw symbols (as rectangles for now)
    for idx, symbol in enumerate(symbols):
        x, y, w, h = symbol['rect']
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 2)
        center = (x + w // 2, y + h // 2)
        cv2.putText(img, str(idx), (center[0]-10, center[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    

    grid_size = 4
    grid = create_grid(canvas_size, grid_size)

    # Mark all symbol bounding boxes as obstacles
    for symbol in symbols:
        mark_rect_on_grid(grid, symbol['rect'], grid_size)

    # 3. Build connections
    edges = build_connections(num_symbols)
    all_connections = [(f"e{i}-e{j}", i, j) for i,j in edges] 
    # For each connection:
    for conn_type, C1_idx, C2_idx in all_connections:
        symbol1 = symbols[C1_idx]
        symbol2 = symbols[C2_idx]
        anchor_pairs = select_anchors_hardcoded(symbols[C1_idx], symbols[C2_idx])
        found_path = False
        for p1, p2 in anchor_pairs:
            g_start = point_to_grid(p1, grid_size)
            g_goal = point_to_grid(p2, grid_size)
            original_start_val = grid[g_start[1], g_start[0]]
            original_goal_val = grid[g_goal[1], g_goal[0]]
            grid[g_start[1], g_start[0]] = 0
            grid[g_goal[1], g_goal[0]] = 0

            path = astar(grid, g_start, g_goal, junctions)

            grid[g_start[1], g_start[0]] = original_start_val
            grid[g_goal[1], g_goal[0]] = original_goal_val

            if path is not None:
                path = soften_l_bend_path(path, min_offset=3, max_offset=7)
                # Mark anchors as used
                if p1 not in symbol1['used_anchors']:
                    symbol1['used_anchors'].append(p1)
                if p2 not in symbol2['used_anchors']:
                    symbol2['used_anchors'].append(p2)
                # Draw and mark path as before
                pts = [grid_to_point(gpt, grid_size) for gpt in path]
                for k in range(len(pts) - 1):
                    cv2.line(img, pts[k], pts[k+1], (0, 0, 255), 2)
                mark_path_on_grid(grid, path, junctions)
                found_path = True
                break  # Stop after first successful path

        if not found_path:
            print(f"No path found for {conn_type}")

    cv2.imwrite("random_circuit.png", img)

if __name__ == "__main__":
    main()