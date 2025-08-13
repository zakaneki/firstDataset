import random
import cv2
import numpy as np
import os
import easyocr
from doctr.models import ocr_predictor
from doctr.io import DocumentFile
import tempfile
from ultralytics import YOLO
from elements import DetectedElement, Transformer, CircuitBreaker, Switch, MVLine
import math

class SymbolPredictor:
    def __init__(self, model_path="best.pt"):
        self.model_path = model_path
        self.model = None
        self.ocr_reader = None
        self.doctr_predictor = None
        self.load_model()
        self.load_ocr()

    def load_model(self):
        if os.path.exists(self.model_path):
            self.model = YOLO(self.model_path)
        else:
            raise FileNotFoundError(f"Model file {self.model_path} not found!")

    def load_ocr(self):
        self.ocr_reader = easyocr.Reader(['en'])
        self.doctr_predictor = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)

    def predict(self, image, conf=0.5):
        results = self.model.predict(image, conf=conf)
        return results[0]

    def detect_text_easyocr(self, image, points, radius, is_vertical, class_name):
        if self.ocr_reader is None:
                return None
        try:
            # Calculate bounding box of the element
            x_coords = points[:, 0]
            y_coords = points[:, 1]
            min_x, max_x = int(np.min(x_coords)), int(np.max(x_coords))
            min_y, max_y = int(np.min(y_coords)), int(np.max(y_coords))
            
            # Calculate element dimensions
            width = max_x - min_x
            height = max_y - min_y
                        
            # Use GUI parameter for search radius
            search_radius = int(max(width, height) * radius)
            
            # Define search regions based on orientation
            if is_vertical:
                # For vertical elements, search horizontally on both sides
                search_regions = [
                    # Left side
                    (max(0, min_x - search_radius), min_y - 10,
                    min_x + width // 2, max_y + 10  ),
                    # Right side
                    (min_x + width // 2, min_y - 10,
                    min(image.shape[1], max_x + search_radius), max_y + 10)
                ]
            else:
                # For horizontal elements, search vertically above and below
                search_regions = [
                    # Above
                    (min_x - 10, max(0, min_y - search_radius),
                    max_x + 10, min_y + height // 2),
                    # Below
                    (min_x - 10, min_y + height // 2,
                    max_x + 10, min(image.shape[0], max_y + search_radius))
                ]
            
            detected_texts = []
            print(f"Searching for text near element '{class_name}'")
            for region_idx, region in enumerate(search_regions):
                x1, y1, x2, y2 = region
                
                # Ensure all coordinates are integers
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Extract region from image
                region_img = image[y1:y2, x1:x2]
                
                if region_img.size == 0:
                    continue
                
                # Perform OCR on the region
                results = self.ocr_reader.readtext(region_img, batch_size=4)
                # results = pytesseract.image_to_data(region_img, output_type=pytesseract.Output.DICT)

                relevant_keywords = ['transformer', 'breaker', 'switch', 'line', 'bus', 'load', 'gen']
                for (bbox, text, confidence) in results:
                    # Filter text based on confidence and relevance
                    if confidence > 0.1:  # Adjust confidence threshold as needed
                    # Check if text is relevant to electrical symbols
                        text_lower = text.lower()
                        
                        
                        # Check if text contains relevant keywords or is short (likely a label)
                        is_relevant = any(keyword in text_lower for keyword in relevant_keywords) or len(text.strip()) <= 10
                        
                        if is_relevant:
                            # Convert bbox coordinates back to original image coordinates
                            orig_bbox = [
                                [x1 + int(bbox[0][0]), y1 + int(bbox[0][1])],
                                [x1 + int(bbox[1][0]), y1 + int(bbox[1][1])],
                                [x1 + int(bbox[2][0]), y1 + int(bbox[2][1])],
                                [x1 + int(bbox[3][0]), y1 + int(bbox[3][1])]
                            ]

                            detected_texts.append({
                                'text': text.strip(),
                                'confidence': confidence,
                                'bbox': orig_bbox,
                                'region': 'left' if region == search_regions[0] else 'right' if len(search_regions) == 2 else 'above' if region == search_regions[0] else 'below'
                            })
                        # Return the most relevant text based on proximity and confidence
            if detected_texts:
                # Calculate the center of the element's bounding box
                element_center_x = (min_x + max_x) / 2
                element_center_y = (min_y + max_y) / 2

                best_text = None
                best_score = -1

                for text_info in detected_texts:
                    # Calculate the center of the text's bounding box
                    text_bbox = np.array(text_info['bbox'])
                    text_center_x = np.mean(text_bbox[:, 0])
                    text_center_y = np.mean(text_bbox[:, 1])
                    
                    # Calculate Euclidean distance
                    distance = np.sqrt((element_center_x - text_center_x)**2 + (element_center_y - text_center_y)**2)
                    # Calculate a score that balances confidence and distance
                    # We want high confidence and low distance.
                    score = text_info['confidence'] / (1 + distance) # Add 1 to avoid division by zero
                    
                    if score > best_score:
                        best_score = score
                        best_text = text_info
                print(f"Best detected text: '{best_text['text']}' with confidence {best_text['confidence']:.2f} in region {best_text['region']}")
                return best_text
            
            return None
            
        except Exception as e:
            print(f"Error in text detection: {str(e)}")
            return None

    def detect_text_doctr(self, image, points, radius, is_vertical, class_name):
        if self.doctr_predictor is None:
                return None
        try:
            # Calculate bounding box of the element
            x_coords = points[:, 0]
            y_coords = points[:, 1]
            min_x, max_x = int(np.min(x_coords)), int(np.max(x_coords))
            min_y, max_y = int(np.min(y_coords)), int(np.max(y_coords))
            width = max_x - min_x
            height = max_y - min_y
            search_radius = int(max(width, height) * radius)
            if is_vertical:
                search_regions = [
                    (max(0, min_x - search_radius), min_y - 5, min_x + width // 2, max_y + 5),
                    (min_x + width // 2, min_y - 5, min(image.shape[1], max_x + search_radius), max_y + 5)
                ]
            else:
                search_regions = [
                    (min_x - 5, max(0, min_y - search_radius), max_x + 5, min_y + height // 2),
                    (min_x - 5, min_y + height // 2, max_x + 5, min(image.shape[0], max_y + search_radius))
                ]
            detected_texts = []
            for region_idx, region in enumerate(search_regions):
                x1, y1, x2, y2 = [int(v) for v in region]
                region_img = image[y1:y2, x1:x2]
                if region_img.size == 0:
                    continue
                # Ensure region is 3-channel RGB
                if region_img.ndim == 2:
                    region_img = cv2.cvtColor(region_img, cv2.COLOR_GRAY2RGB)
                elif region_img.shape[2] == 4:
                    region_img = cv2.cvtColor(region_img, cv2.COLOR_RGBA2RGB)


                # Write ROI to a temp file and feed path to doctr
                tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                tmp_path = tmp.name
                tmp.close()
                try:
                    cv2.imwrite(tmp_path, cv2.cvtColor(region_img, cv2.COLOR_RGB2BGR))
                    doc = DocumentFile.from_images([tmp_path])
                    result = self.doctr_predictor(doc)
                finally:
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass

                print(f"Searching for text near element '{class_name}'")
                # Parse doctr result
                for page in result.pages:
                    page_h, page_w = getattr(page, "dimensions", (region_img.shape[0], region_img.shape[1]))
                    for block in page.blocks:
                        for line in block.lines:
                            for word in line.words:
                                text = getattr(word, "value", "")
                                confidence = float(getattr(word, "confidence", 1.0))
                                if confidence > 0.1 and text:
                                    # Recover bbox from geometry if available (relative coords)
                                    bbox = None
                                    geom = getattr(word, "geometry", None)
                                    if geom is not None:
                                        g = np.array(geom, dtype=float)
                                        # Accept quadrilateral (4x2) or two-corner (2x2)
                                        if g.ndim == 2 and g.shape[1] == 2:
                                            if g.shape[0] == 4:
                                                pts = g.copy()
                                            elif g.shape[0] == 2:  # tl, br -> expand to rectangle
                                                tl, br = g[0], g[1]
                                                pts = np.array([tl, [br[0], tl[1]], br, [tl[0], br[1]]], dtype=float)
                                            else:
                                                pts = None
                                            if pts is not None:
                                                pts[:, 0] = pts[:, 0] * page_w + x1
                                                pts[:, 1] = pts[:, 1] * page_h + y1
                                                bbox = pts.astype(int).tolist()

                                    text_lower = text.lower()
                                    relevant_keywords = ['transformer', 'breaker', 'switch', 'line', 'bus', 'load', 'gen']
                                    is_relevant = any(keyword in text_lower for keyword in relevant_keywords) or len(text.strip()) <= 10
                                    if is_relevant:
                                        detected_texts.append({
                                            'text': text.strip(),
                                            'confidence': confidence,
                                            'bbox': None,  # doctr does not provide bbox in this API
                                            'region': 'left' if region == search_regions[0] else 'right' if len(search_regions) == 2 else 'above' if region == search_regions[0] else 'below'
                                        })
            if detected_texts:
                element_center = np.array([(min_x + max_x) / 2.0, (min_y + max_y) / 2.0])
                def score(item):
                    if item['bbox']:
                        bb = np.array(item['bbox'])
                        center = np.array([bb[:, 0].mean(), bb[:, 1].mean()])
                        dist = np.linalg.norm(center - element_center)
                        return item['confidence'] / (1.0 + dist)
                    return item['confidence']
                best_text = max(detected_texts, key=score)
                print(f"Doctr best detected text: '{best_text['text']}' conf={best_text['confidence']:.2f} region={best_text['region']}")
                return best_text
            return None
        except Exception as e:
            print(f"Error in doctr text detection: {str(e)}")
            return None

    def detect_element_orientation_from_obb(self, points):
        """Detect element orientation from YOLO OBB coordinates"""
        try:
            # Get the 4 corner points
            corners = points.reshape(-1, 2)
            
            # Calculate all 4 sides of the bounding box
            sides = []
            for i in range(4):
                # Calculate distance between consecutive corners (with wrap-around)
                side = np.linalg.norm(corners[(i+1) % 4] - corners[i])
                sides.append(side)
                        
            # Find the longest and shortest sides
            longest_side = max(sides)
            shortest_side = min(sides)
                        
            # Determine orientation based on aspect ratio
            aspect_ratio = longest_side / shortest_side if shortest_side > 0 else 1
            
            # If aspect ratio is significant (>1.5), use it to determine orientation
            if aspect_ratio > 1.5:
                # For rectangular elements, the longest side indicates the orientation.
                # Find the vector of the longest side and determine if it's more vertical or horizontal.
                longest_side_index = np.argmax(sides)
                p1 = corners[longest_side_index]
                p2 = corners[(longest_side_index + 1) % 4]
                
                # Calculate the vector of the longest side
                dx = abs(p2[0] - p1[0])
                dy = abs(p2[1] - p1[1])
                
                # If the change in y is greater than the change in x, it's vertical
                is_vertical = dy > dx
            else:
                # For more square elements, use the overall bounding box
                min_x, max_x = np.min(corners[:, 0]), np.max(corners[:, 0])
                min_y, max_y = np.min(corners[:, 1]), np.max(corners[:, 1])
                width = max_x - min_x
                height = max_y - min_y
                is_vertical = height > width            
            return is_vertical
            
        except Exception as e:
            print(f"Error in OBB orientation detection: {str(e)}")
            # Fallback to simple aspect ratio
            min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
            min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])
            width = max_x - min_x
            height = max_y - min_y
            is_vertical = height > width
            print(f"OBB fallback: width={width}, height={height}, is_vertical={is_vertical}")
            return is_vertical
    
    def parse_detections(self, result, class_names):
        detections = []
        boxes = result.obb
        if boxes is not None:
            for box in boxes:
                # Get OBB coordinates (8 values: x1,y1,x2,y2,x3,y3,x4,y4)
                xyxyxyxy = box.xyxyxyxy.cpu().numpy()
                points = xyxyxyxy.reshape(-1, 2).astype(np.int32)

                # Get class and confidence
                cls = int(box.cls.cpu().numpy().item())
                conf = float(box.conf.cpu().numpy().item())

                # Get class name
                class_name = class_names.get(cls, f"Class {cls}")

                # Choose class type
                if class_name == "Transformer":
                    element = Transformer(cls, class_name, points, conf)
                elif class_name == "Circuit Breaker":
                    element = CircuitBreaker(cls, class_name, points, conf)
                elif class_name == "Switch":
                    element = Switch(cls, class_name, points, conf)
                elif class_name == "MV Line":
                    element = MVLine(cls, class_name, points, conf)
                else:
                    element = DetectedElement(cls, class_name, points, conf)
                detections.append(element)
        return detections
    
    def extract_wires(self, image):
        """Extract wire polylines from the image using edge + Hough segments, then merge into polylines."""
        img = image.copy()
        if img.ndim == 3 and img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img

        bw = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 10
        )

        min_len = int(0.01833 * max(img.shape[:2]) + 3)

        segs = None

        # Prefer LSD
        lsd = None
        try:
            lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_ADV, 1)
        except Exception:
            lsd = None

        if lsd is not None:
            lines, _width, _prec, _nfa = lsd.detect(bw)
            # img_vis = bw.copy()

            # if lines is not None:
            #     for l in lines:
            #         x1, y1, x2, y2 = map(int, l[0])
            #         color = tuple(random.randint(0, 255) for _ in range(3))
            #         cv2.line(img_vis, (x1, y1), (x2, y2), color, 5)
            #     cv2.imwrite("lsd_raw_lines.png", img_vis)
            if lines is not None:
                # lines: Nx1x4 -> (x1,y1,x2,y2)
                segs_lsd = []
                for l in lines:
                    x1, y1, x2, y2 = map(float, l[0])
                    if math.hypot(x2 - x1, y2 - y1) >= min_len:
                        segs_lsd.append((int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))))
                if segs_lsd:
                    segs = segs_lsd

        # Fallback to Canny + Hough if LSD unavailable or empty
        if segs is None:
            thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_,
                                        cv2.THRESH_BINARY_INV, 15, 10)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)
            edges = cv2.Canny(thr, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30,
                                    minLineLength=min_len, maxLineGap=10)
            if lines is None:
                return []
            segs = [tuple(map(int, l[0])) for l in lines]

        # Merge segments (graph-based, supports branching)
        segs = self.merge_segments_iterative(segs, int(0.00667 * max(img.shape[:2]) + 8.33))
        polylines = [np.array([[x1, y1], [x2, y2]], dtype=np.int32) for (x1, y1, x2, y2) in segs]
        return polylines
    
    def merge_segments_iterative(self, segs, dist_thresh=35, angle_thresh_deg=5):
        """
        Merge collinear/parallel and overlapping/adjacent segments (horizontal/vertical only).
        Handles subset/overlap cases robustly.
        """
        def seg_angle(s):
            x1, y1, x2, y2 = s
            return math.degrees(math.atan2(y2 - y1, x2 - x1))

        def is_horiz(s):
            a = seg_angle(s)
            return abs(a) < angle_thresh_deg or abs(a - 180) < angle_thresh_deg or abs(a + 180) < angle_thresh_deg

        def is_vert(s):
            a = seg_angle(s)
            return abs(abs(a) - 90) < angle_thresh_deg

        def merge_axis_aligned(s1, s2):
            # Assumes both are horizontal or both are vertical
            if is_horiz(s1):
                y = int(round((s1[1] + s1[3] + s2[1] + s2[3]) / 4))
                xs = [s1[0], s1[2], s2[0], s2[2]]
                return (min(xs), y, max(xs), y)
            else:
                x = int(round((s1[0] + s1[2] + s2[0] + s2[2]) / 4))
                ys = [s1[1], s1[3], s2[1], s2[3]]
                return (x, min(ys), x, max(ys))

        def overlap_1d(a1, a2, b1, b2, tol):
            # Returns True if [a1,a2] and [b1,b2] overlap or touch within tol
            a1, a2 = sorted([a1, a2])
            b1, b2 = sorted([b1, b2])
            return not (a2 < b1 - tol or b2 < a1 - tol)

        segs = list(segs)
        merged = True
        while merged:
            merged = False
            used = [False] * len(segs)
            new_segs = []
            i = 0
            while i < len(segs):
                if used[i]:
                    i += 1
                    continue
                s1 = segs[i]
                found = False
                for j in range(i + 1, len(segs)):
                    if used[j]:
                        continue
                    s2 = segs[j]
                    # Both horizontal
                    if is_horiz(s1) and is_horiz(s2):
                        # y must be close
                        y1 = (s1[1] + s1[3]) / 2
                        y2 = (s2[1] + s2[3]) / 2
                        if abs(y1 - y2) < dist_thresh:
                            # x projections must overlap/touch
                            if overlap_1d(s1[0], s1[2], s2[0], s2[2], dist_thresh):
                                merged_seg = merge_axis_aligned(s1, s2)
                                used[i] = used[j] = True
                                new_segs.append(merged_seg)
                                merged = True
                                found = True
                                break
                    # Both vertical
                    elif is_vert(s1) and is_vert(s2):
                        x1 = (s1[0] + s1[2]) / 2
                        x2 = (s2[0] + s2[2]) / 2
                        if abs(x1 - x2) < dist_thresh:
                            if overlap_1d(s1[1], s1[3], s2[1], s2[3], dist_thresh):
                                merged_seg = merge_axis_aligned(s1, s2)
                                used[i] = used[j] = True
                                new_segs.append(merged_seg)
                                merged = True
                                found = True
                                break
                if not found:
                    new_segs.append(s1)
                    used[i] = True
                i += 1
            segs = new_segs
        return segs

    def find_intersections(self, segs, max_dim, thresh=20):
        """
        Find all intersection points between segments.
        If an intersection is near an existing one (within thresh), reuse that point.
        Also returns: dict {intersection_idx: [segment_idx, ...]}
        Returns: (list of (x, y) intersection points, mapping dict)
        """
        def seg_intersect(a1, a2, b1, b2, eps = 10):
            # Returns intersection point if segments (a1,a2) and (b1,b2) cross, else None
            x1, y1 = a1
            x2, y2 = a2
            x3, y3 = b1
            x4, y4 = b2
            denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
            if abs(denom) < 1e-8:
                return None  # Parallel or coincident
            px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4)) / denom
            py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4)) / denom

            # Check if intersection is within both segments
            def on_seg(xa, ya, xb, yb, xp, yp):
                return min(xa, xb)-eps <= xp <= max(xa, xb)+eps and min(ya, yb)-eps<= yp <= max(ya, yb)+eps
            
            if on_seg(x1, y1, x2, y2, px, py) and on_seg(x3, y3, x4, y4, px, py):
                return (int(round(px)), int(round(py)))
            return None

        intersections = []
        mapping = {}
        for i, s1 in enumerate(segs):
            a1 = (s1[0], s1[1])
            a2 = (s1[2], s1[3])
            for j in range(i+1, len(segs)):
                s2 = segs[j]
                b1 = (s2[0], s2[1])
                b2 = (s2[2], s2[3])
                pt = seg_intersect(a1, a2, b1, b2, int(0.005 * max_dim + 5))
                if pt is not None:
                    # Check if close to existing intersection
                    found = False
                    idx = None
                    for k, existing in enumerate(intersections):
                        if np.hypot(existing[0]-pt[0], existing[1]-pt[1]) < thresh:
                            # Use existing
                            pt = existing
                            found = True
                            idx = k
                            break
                    if not found:
                        intersections.append(pt)
                        idx = len(intersections) - 1
                    # Add both segments to the mapping for this intersection
                    mapping.setdefault(idx, set()).update([i, j])
        # Convert sets to lists for consistency
        mapping = {k: list(v) for k, v in mapping.items()}
        self.intersections = intersections
        return intersections, mapping
    
    @staticmethod
    def build_intersection_graph(intersections, mapping, segs):
        """
        Returns: adjacency dict {intersection_idx: set(neighbor_intersection_idx)}
        and segment_to_intersections {segment_idx: [int_idx1, int_idx2]}
        """
        adj = {i: set() for i in range(len(intersections))}
        segment_to_intersections = {}
        for int_idx, seg_idxs in mapping.items():
            for seg_idx in seg_idxs:
                # For each segment, find all intersections it touches
                if seg_idx not in segment_to_intersections:
                    segment_to_intersections[seg_idx] = []
                segment_to_intersections[seg_idx].append(int_idx)
        # Now, for each segment that touches two intersections, connect those intersections
        for seg_idx, int_idxs in segment_to_intersections.items():
            if len(int_idxs) < 2:
                continue
            # Sort intersection points along the segment
            x1, y1, x2, y2 = segs[seg_idx]
            p1 = np.array([x1, y1])
            p2 = np.array([x2, y2])
            seg_vec = p2 - p1
            seg_len = np.linalg.norm(seg_vec)
            if seg_len < 1e-6:
                continue
            # Project intersection points onto the segment
            def proj_param(pt):
                pt = np.array(pt)
                return np.dot(pt - p1, seg_vec) / (seg_len ** 2)
            int_idxs_sorted = sorted(int_idxs, key=lambda idx: proj_param(intersections[idx]))
            # Connect consecutive intersections
            for a, b in zip(int_idxs_sorted, int_idxs_sorted[1:]):
                adj[a].add(b)
                adj[b].add(a)
        return adj, segment_to_intersections
    
    @staticmethod
    def map_elements_to_intersections(elements, segs, segment_to_intersections, thresh=50):
        """
        Returns: dict {element_idx: [intersection_idx, ...]}
        For each element, finds all segments that touch it, then all intersections on those segments.
        """
        mapping = {i: set() for i in range(len(elements))}
        for el_idx, el in enumerate(elements):
            poly = np.array(el.bbox, dtype=np.int32).reshape(-1, 2)
            for seg_idx, (x1, y1, x2, y2) in enumerate(segs):
                # Check if segment touches element (using endpoints or segment-edge distance)
                p1, p2 = (x1, y1), (x2, y2)
                # Use cv2.pointPolygonTest for endpoints
                d1 = cv2.pointPolygonTest(poly.reshape(-1, 1, 2), p1, measureDist=True)
                d2 = cv2.pointPolygonTest(poly.reshape(-1, 1, 2), p2, measureDist=True)
                if (d1 is not None and d1 >= -thresh) or (d2 is not None and d2 >= -thresh):
                    # Add all intersections on this segment
                    for int_idx in segment_to_intersections.get(seg_idx, []):
                        mapping[el_idx].add(int_idx)
                else:
                    # Also check if segment crosses the polygon (element)
                    for i in range(len(poly)):
                        q1 = tuple(poly[i])
                        q2 = tuple(poly[(i+1) % len(poly)])
                        if segments_intersect(p1, p2, q1, q2):
                            for int_idx in segment_to_intersections.get(seg_idx, []):
                                mapping[el_idx].add(int_idx)
                            break
        # Convert sets to lists
        mapping = {k: list(v) for k, v in mapping.items()}
        return mapping

    @staticmethod
    def find_element_connections(adj, element_to_ints):
        """
        Returns: list of (element_a_idx, element_b_idx) pairs that are connected via the wire graph.
        """
        # For each element, BFS from its intersections to find other elements
        connections = set()
        for a_idx, ints_a in element_to_ints.items():
            visited = set()
            queue = list(ints_a)
            while queue:
                cur = queue.pop(0)
                visited.add(cur)
                # Check if any other element touches this intersection
                for b_idx, ints_b in element_to_ints.items():
                    if b_idx != a_idx and any(i == cur for i in ints_b):
                        # Found a connection
                        connections.add(tuple(sorted((a_idx, b_idx))))
                # Traverse to neighbors
                for nb in adj[cur]:
                    if nb not in visited:
                        queue.append(nb)
        return connections
    
    @staticmethod
    def find_direct_element_connections(elements, segs, thresh=50):
        """
        Returns: set of (element_a_idx, element_b_idx) pairs directly connected by a segment.
        """
        direct_pairs = set()
        polys = [np.array(el.bbox, dtype=np.int32).reshape(-1, 2) for el in elements]
        for seg_idx, (x1, y1, x2, y2) in enumerate(segs):
            p1, p2 = (x1, y1), (x2, y2)
            touched = []
            for e_idx, poly in enumerate(polys):
                # Use cv2.pointPolygonTest for endpoints
                d1 = cv2.pointPolygonTest(poly.reshape(-1, 1, 2), p1, measureDist=True)
                d2 = cv2.pointPolygonTest(poly.reshape(-1, 1, 2), p2, measureDist=True)
                if (d1 is not None and d1 >= -thresh) or (d2 is not None and d2 >= -thresh):
                    touched.append(e_idx)
                else:
                    # Also check if segment crosses the polygon (element)
                    for i in range(len(poly)):
                        q1 = tuple(poly[i])
                        q2 = tuple(poly[(i+1) % len(poly)])
                        if segments_intersect(p1, p2, q1, q2):
                            touched.append(e_idx)
                            break
            # If two different elements are touched by this segment, connect them
            if len(touched) >= 2:
                for i in range(len(touched)):
                    for j in range(i+1, len(touched)):
                        a, b = touched[i], touched[j]
                        if a != b:
                            direct_pairs.add(tuple(sorted((a, b))))
        return direct_pairs

    def compute_connectivity_via_graph(self, elements, polylines, max_dim, int_thresh=10, touch_thresh=50):
        """
        Convenience wrapper that uses the intersection graph pipeline and returns edges like compute_connectivity.
        """
        # Convert polylines to simple segments (two endpoints)
        segs = []
        for poly in polylines:
            p = np.asarray(poly).reshape(-1, 2)
            if len(p) >= 2:
                x1, y1 = map(int, p[0])
                x2, y2 = map(int, p[-1])
                segs.append((x1, y1, x2, y2))

        intersections, intersection_to_segments = self.find_intersections(segs, max_dim, thresh=int_thresh)
        adj, seg_to_ints = self.build_intersection_graph(intersections, intersection_to_segments, segs)
        el_to_ints = self.map_elements_to_intersections(elements, segs, seg_to_ints, thresh=touch_thresh)
        pairs_graph = self.find_element_connections(adj, el_to_ints)
        pairs_direct = self.find_direct_element_connections(elements, segs, thresh=touch_thresh)
        all_pairs = pairs_graph | pairs_direct
        return [{'a': a, 'b': b, 'polyline_idx': None} for a, b in all_pairs]

def segments_intersect(p1, p2, q1, q2):
    """Returns True if segments (p1,p2) and (q1,q2) intersect."""
    def ccw(a, b, c):
        return (c[1]-a[1]) * (b[0]-a[0]) > (b[1]-a[1]) * (c[0]-a[0])
    return (ccw(p1, q1, q2) != ccw(p2, q1, q2)) and (ccw(p1, p2, q1) != ccw(p1, p2, q2))