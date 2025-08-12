import cv2
import numpy as np
import os
import easyocr
from doctr.models import ocr_predictor
from doctr.io import DocumentFile
import tempfile
from ultralytics import YOLO
from elements import DetectedElement, Transformer, CircuitBreaker, Switch, MVLine

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
                    (max(0, min_x - search_radius), min_y - 5,
                    min_x + width // 2, max_y + 5),
                    # Right side
                    (min_x + width // 2, min_y - 5,
                    min(image.shape[1], max_x + search_radius), max_y + 5)
                ]
            else:
                # For horizontal elements, search vertically above and below
                search_regions = [
                    # Above
                    (min_x - 5, max(0, min_y - search_radius),
                    max_x + 5, min_y + height // 2),
                    # Below
                    (min_x - 5, min_y + height // 2,
                    max_x + 5, min(image.shape[0], max_y + search_radius))
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