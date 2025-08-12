import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
from ultralytics import YOLO
import threading
from predictor import SymbolPredictor

class YOLOGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Electrical Symbol Detector")
        self.root.geometry("1200x800")
        self.predictor = SymbolPredictor()
          
        # Initialize OCR reader var
        self.ocr_engine_var = tk.StringVar(value="easyocr")
        
        # Variables
        self.original_image = None
        self.result_image = None
        self.image_path = None
        
        # Class names
        self.class_names = {
            0: "Transformer",
            1: "Circuit Breaker", 
            2: "Switch",
            3: "MV Line"
        }
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Electrical Symbol Detection with YOLO", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Left panel - Controls
        left_panel = ttk.Frame(main_frame)
        left_panel.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Upload button
        self.upload_btn = ttk.Button(left_panel, text="Upload Image", 
                                    command=self.upload_image, width=20)
        self.upload_btn.grid(row=0, column=0, pady=(0, 10))
        
        # Detect button
        self.detect_btn = ttk.Button(left_panel, text="Detect Symbols", 
                                    command=self.detect_symbols, width=20, state="disabled")
        self.detect_btn.grid(row=1, column=0, pady=(0, 10))
        
        # Clear button
        self.clear_btn = ttk.Button(left_panel, text="Clear Results", 
                                   command=self.clear_results, width=20)
        self.clear_btn.grid(row=2, column=0, pady=(0, 20))
        
        # Confidence threshold
        ttk.Label(left_panel, text="Confidence Threshold:").grid(row=3, column=0, pady=(0, 5))
        self.conf_var = tk.DoubleVar(value=0.5)
        self.conf_scale = ttk.Scale(left_panel, from_=0.1, to=1.0, 
                                   variable=self.conf_var, orient="horizontal")
        self.conf_scale.grid(row=4, column=0, pady=(0, 5))
        self.conf_label = ttk.Label(left_panel, text="0.5")
        self.conf_label.grid(row=5, column=0, pady=(0, 20))
        self.conf_scale.configure(command=self.update_conf_label)
        
        # Search radius parameter
        ttk.Label(left_panel, text="Text Search Radius (multiplier):").grid(row=6, column=0, pady=(0, 5))
        self.radius_var = tk.DoubleVar(value=2.0)
        self.radius_scale = ttk.Scale(left_panel, from_=0.5, to=5.0, 
                                     variable=self.radius_var, orient="horizontal")
        self.radius_scale.grid(row=7, column=0, pady=(0, 5))
        self.radius_label = ttk.Label(left_panel, text="2.0")
        self.radius_label.grid(row=8, column=0, pady=(0, 20))
        self.radius_scale.configure(command=self.update_radius_label)
        
        # OCR engine selection
        ttk.Label(left_panel, text="OCR Engine:").grid(row=13, column=0, pady=(0, 5))
        ocr_combo = ttk.Combobox(left_panel, textvariable=self.ocr_engine_var, values=["easyocr", "doctr"], state="readonly", width=20)
        ocr_combo.grid(row=14, column=0, pady=(0, 10))

        # Bounding box toggle
        self.show_bbox_var = tk.BooleanVar(value=True)
        self.bbox_toggle = ttk.Checkbutton(left_panel, text="Show Bounding Boxes", 
                                          variable=self.show_bbox_var, 
                                          command=self.toggle_bounding_boxes)
        self.bbox_toggle.grid(row=9, column=0, pady=(0, 20))
        
        # Legend frame
        legend_frame = ttk.LabelFrame(left_panel, text="Legend of Detected Symbols", padding="5")
        legend_frame.grid(row=10, column=0, pady=(0, 10), sticky=(tk.W, tk.E))
        
        # Legend canvas
        self.legend_canvas = tk.Canvas(legend_frame, width=200, height=150, bg="white", relief="sunken", bd=1)
        self.legend_canvas.grid(row=0, column=0, pady=(5, 0))
        
        # Results info
        ttk.Label(left_panel, text="Detection Results:", font=("Arial", 12, "bold")).grid(row=11, column=0, pady=(0, 10))
        
        # Results text
        self.results_text = tk.Text(left_panel, width=25, height=15, wrap=tk.WORD)
        self.results_text.grid(row=12, column=0, pady=(0, 10))
        
        # Scrollbar for results
        results_scrollbar = ttk.Scrollbar(left_panel, orient="vertical", command=self.results_text.yview)
        results_scrollbar.grid(row=12, column=1, sticky=(tk.N, tk.S))
        self.results_text.configure(yscrollcommand=results_scrollbar.set)
        
        # Right panel - Image display
        right_panel = ttk.Frame(main_frame)
        right_panel.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        right_panel.columnconfigure(0, weight=1)
        right_panel.rowconfigure(0, weight=1)
        
        # Image canvas
        self.canvas = tk.Canvas(right_panel, bg="white", relief="sunken", bd=2)
        self.canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Bind resize event to canvas
        self.canvas.bind('<Configure>', self._on_canvas_resize)
        
        # Scrollbars for canvas
        h_scrollbar = ttk.Scrollbar(right_panel, orient="horizontal", command=self.canvas.xview)
        h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        v_scrollbar = ttk.Scrollbar(right_panel, orient="vertical", command=self.canvas.yview)
        v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.canvas.configure(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief="sunken")
        status_bar.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
    def update_conf_label(self, value):
        """Update confidence label when scale changes"""
        self.conf_label.config(text=f"{float(value):.2f}")
    
    def update_radius_label(self, value):
        """Update radius label when scale changes"""
        self.radius_label.config(text=f"{float(value):.1f}")
    
    def upload_image(self):
        """Upload and display an image"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                self.image_path = file_path
                self.original_image = cv2.imread(file_path)
                if self.original_image is None:
                    raise ValueError("Could not read image")
                
                # Convert BGR to RGB
                self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
                
                # Display image
                self.display_image(self.original_image)
                
                # Enable detect button
                self.detect_btn.config(state="normal")
                
                # Update status
                self.status_var.set(f"Image loaded: {os.path.basename(file_path)}")
                
                # Clear previous results
                self.results_text.delete(1.0, tk.END)
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def display_image(self, image):
        """Display image on canvas"""
        # Get canvas size
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            # Canvas not yet sized, use default
            canvas_width, canvas_height = 800, 600
        
        # Resize image to fit canvas while maintaining aspect ratio
        h, w = image.shape[:2]
        scale = min(canvas_width / w, canvas_height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize image
        resized_image = cv2.resize(image, (new_w, new_h))
        
        # Convert to PIL Image
        pil_image = Image.fromarray(resized_image)
        self.photo = ImageTk.PhotoImage(pil_image)
        
        # Clear canvas
        self.canvas.delete("all")
        
        # Calculate center position
        center_x = (canvas_width - new_w) // 2
        center_y = (canvas_height - new_h) // 2
        
        # Display image centered
        self.canvas.create_image(center_x, center_y, anchor="nw", image=self.photo)
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def detect_symbols(self):
        """Run YOLO detection on the uploaded image"""     
        if self.original_image is None:
            messagebox.showerror("Error", "Please upload an image first!")
            return
        
        # Disable buttons during detection
        self.detect_btn.config(state="disabled")
        self.upload_btn.config(state="disabled")
        self.status_var.set("Running detection...")
        
        # Run detection in a separate thread to avoid blocking GUI
        thread = threading.Thread(target=self._run_detection)
        thread.daemon = True
        thread.start()
    
    def _run_detection(self):
        """Run detection in background thread"""
        try:
            # Use SymbolPredictor for detection
            result = self.predictor.predict(self.original_image, conf=self.conf_var.get())
            # Update GUI in main thread
            self.root.after(0, lambda: self._process_results(result))
        except Exception as e:
            self.root.after(0, lambda: self._handle_detection_error(str(e)))
    
    def _process_results(self, result):
        """Process and display detection results"""
        try:
            # Create a copy of original image for drawing
            result_image = self.original_image.copy()
            
            # # Get detections
            # boxes = result.obb
            # detections = []
            
            # if boxes is not None:
            #     # First pass: collect all detections for orientation analysis
            #     all_detections = []
            #     for box in boxes:
            #         xyxyxyxy = box.xyxyxyxy.cpu().numpy()
            #         points = xyxyxyxy.reshape(-1, 2).astype(np.int32)
            #         all_detections.append(points)
                
            #     # Second pass: process each detection with text recognition (without drawing)
            #     for box in boxes:
            #         # Get OBB coordinates (8 values: x1,y1,x2,y2,x3,y3,x4,y4)
            #         xyxyxyxy = box.xyxyxyxy.cpu().numpy()
                    
            #         # Get class and confidence - extract single elements
            #         cls = int(box.cls.cpu().numpy().item())
            #         conf = float(box.conf.cpu().numpy().item())
                    
            #         # Get class name
            #         class_name = self.class_names.get(cls, f"Class {cls}")
                    
            #         # Convert to points for drawing
            #         points = xyxyxyxy.reshape(-1, 2).astype(np.int32)
                    
            #         # Detect text near the element
            #         detected_text = self._detect_text_near_element(result_image, points, class_name)
                    
            #         # Add to detections list
            #         detections.append({
            #             'class': class_name,
            #             'confidence': conf,
            #             'bbox': points.tolist(),  # Store all 4 corner points
            #             'detected_text': detected_text,
            #             'color': self._get_class_color(cls)
            #         })
            
            # Use SymbolPredictor to parse detections into DetectedElement objects
            elements = self.predictor.parse_detections(result, self.class_names)
            detections = []

            for element in elements:
                # Convert bbox to list for compatibility
                points = np.array(element.bbox, dtype=np.int32)
                # Detect text near the element
                detected_text = self._detect_text_near_element(result_image, points, element.class_name)
                # Build detection dict for legacy code
                detections.append({
                    'class': element.class_name,
                    'confidence': element.confidence,
                    'bbox': points.tolist(),
                    'detected_text': detected_text,
                    'color': element.color if element.color is not None else self._get_class_color(element.class_id)
                })

            # Store detections for toggle functionality
            self.current_detections = detections
            
            # Update legend in UI component
            self._update_legend(detections)
            
            # Draw image with or without bounding boxes
            self._redraw_image_with_boxes()
            
            # Update results text
            self._update_results_text(detections)
            
            # Re-enable buttons
            self.detect_btn.config(state="normal")
            self.upload_btn.config(state="normal")
            self.status_var.set(f"Detection complete: {len(detections)} symbols found")
            
        except Exception as e:
            self._handle_detection_error(str(e))
    
    def _update_legend(self, detections):
        """Update the legend canvas with detected classes"""
        # Clear the legend canvas
        self.legend_canvas.delete("all")
        
        if not detections:
            self.legend_canvas.create_text(100, 75, text="No symbols detected", 
                                         fill="gray", font=("Arial", 10))
            return
        
        # Get unique classes from detections
        detected_classes = list(set([det['class'] for det in detections]))
        
        # Legend parameters
        box_size = 15
        text_offset = 25
        line_height = 25
        start_y = 20
        
        # Draw each class with its color
        for i, class_name in enumerate(detected_classes):
            # Get class ID for color
            class_id = None
            for cls_id, name in self.class_names.items():
                if name == class_name:
                    class_id = cls_id
                    break
            
            if class_id is not None:
                color = self._get_class_color(class_id)
                # Convert BGR to RGB for tkinter (swap B and R)
                rgb_color = f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}'
                
                # Draw color box
                box_x = 10
                box_y = start_y + i * line_height
                self.legend_canvas.create_rectangle(box_x, box_y, 
                                                  box_x + box_size, box_y + box_size, 
                                                  fill=rgb_color, outline="black")
                
                # Draw class name
                text_x = box_x + text_offset
                text_y = box_y + box_size // 2
                self.legend_canvas.create_text(text_x, text_y, text=class_name, 
                                             anchor="w", font=("Arial", 10))
    
    def _get_class_color(self, class_id):
        """Get color for class ID"""
        colors = [
            (255, 0, 0),    # Red for Transformer
            (0, 255, 0),    # Green for Circuit Breaker
            (0, 0, 255),    # Blue for Switch
            (255, 0, 255)   # Magenta for MV Line
        ]
        return colors[class_id % len(colors)]
    
    def _update_results_text(self, detections):
        """Update the results text area"""
        self.results_text.delete(1.0, tk.END)
        
        if not detections:
            self.results_text.insert(tk.END, "No symbols detected.\n")
            return
        
        
        self.results_text.insert(tk.END, f"Detected {len(detections)} symbols:\n\n")
        
        for i, det in enumerate(detections, 1):
            self.results_text.insert(tk.END, 
                f"{i}. {det['class']}\n"
                f"   Confidence: {det['confidence']:.3f}\n"
                f"   BBox: {det['bbox']}\n")
            
            if det.get('detected_text'):
                text_info = det['detected_text']
                self.results_text.insert(tk.END, 
                    f"   Detected Text: '{text_info['text']}'\n"
                    f"   Text Confidence: {text_info['confidence']:.3f}\n"
                    f"   Text Position: {text_info['region']}\n")
            
            self.results_text.insert(tk.END, "\n")
    
    def _handle_detection_error(self, error_msg):
        """Handle detection errors"""
        messagebox.showerror("Detection Error", f"Failed to run detection: {error_msg}")
        self.detect_btn.config(state="normal")
        self.upload_btn.config(state="normal")
        self.status_var.set("Detection failed")
    
    def clear_results(self):
        """Clear results and display original image"""
        if self.original_image is not None:
            self.display_image(self.original_image)
            self.result_image = None
            self.current_detections = []
            self.results_text.delete(1.0, tk.END)
            self.legend_canvas.delete("all")
            self.legend_canvas.create_text(100, 75, text="No symbols detected", 
                                         fill="gray", font=("Arial", 10))
            self.status_var.set("Results cleared")

    def _on_canvas_resize(self, event):
        """Handle canvas resize event to re-center image"""
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            return # No image to re-center if canvas is too small
        
        # Get the current image dimensions
        if hasattr(self, 'photo'):
            current_image_width = self.photo.width()
            current_image_height = self.photo.height()
        else:
            current_image_width = 0
            current_image_height = 0
        
        # Calculate new center position
        new_center_x = (canvas_width - current_image_width) // 2
        new_center_y = (canvas_height - current_image_height) // 2
        
        # Move the image to the new center
        self.canvas.delete("all") # Clear existing image
        if hasattr(self, 'photo'):
            self.canvas.create_image(new_center_x, new_center_y, anchor="nw", image=self.photo)
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _detect_text_near_element(self, image, points, class_name):
        """Detect text near a detected element"""
        ocr_engine = self.ocr_engine_var.get()
        is_vertical = self.predictor.detect_element_orientation_from_obb(points)
        radius = self.radius_var.get()
        if ocr_engine == "easyocr":
            return self.predictor.detect_text_easyocr(image, points, radius, is_vertical, class_name)
        elif ocr_engine == "doctr":
            return self.predictor.detect_text_doctr(image, points, radius, is_vertical, class_name)
        else:
            return None

    def toggle_bounding_boxes(self):
        """Toggle bounding box visibility"""
        if hasattr(self, 'result_image') and self.result_image is not None:
            self._redraw_image_with_boxes()
        elif hasattr(self, 'original_image') and self.original_image is not None:
            self.display_image(self.original_image)
    
    def _redraw_image_with_boxes(self):
        """Redraw the image with or without bounding boxes based on toggle state"""
        if not hasattr(self, 'current_detections') or not self.current_detections:
            return
        
        # Start with original image
        result_image = self.original_image.copy()

         # --- Calculate scaling factor based on image size ---
        img_h, img_w = result_image.shape[:2]
        scale = max(img_w, img_h) / 800  # 800 is a reference size, adjust as needed
        thickness = max(2, int(2 * scale))  # Minimum thickness of 2
        circle_radius = max(12, int(15 * scale))  # Minimum radius of 12
        
        # Always draw index numbers first (regardless of toggle state)
        for i, det in enumerate(self.current_detections):
            points = np.array(det['bbox'], dtype=np.int32)
            
            # Draw index number (always visible)
            index_text = str(i + 1)  # 1-based indexing
            font_scale = 0.6 * scale
            font_thickness = max(1, int(3 * scale))
            index_size = cv2.getTextSize(index_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
            
            # Position index in top-left corner of the bounding box
            leftmost_x = np.min(points[:, 0])
            topmost_y = np.min(points[:, 1])
            
            # Default position: top-left of the bounding box
            index_x = leftmost_x - 5
            index_y = topmost_y - 5
            
            # If the top-left position is outside the image frame, move to top-right
            if index_x - circle_radius < 0 or index_y - circle_radius < 0:
                rightmost_x = np.max(points[:, 0])
                index_x = rightmost_x + 5
                # Check if top-right is also out of bounds, if so, adjust
                if index_x + circle_radius > result_image.shape[1]:
                    index_x = result_image.shape[1] - circle_radius - 1
                if index_y - circle_radius < 0:
                    index_y = circle_radius + 1
            
            cv2.circle(result_image, (index_x, index_y), circle_radius, (0, 0, 0), -1)
            cv2.circle(result_image, (index_x, index_y), circle_radius, (255, 255, 255), thickness)
            
            # Draw index number
            cv2.putText(result_image, index_text,
                      (index_x - index_size[0] // 2, index_y + index_size[1] // 2),
                      cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
        
        # Draw bounding boxes and labels only if toggle is on
        if self.show_bbox_var.get():
            for i, det in enumerate(self.current_detections):
                points = np.array(det['bbox'], dtype=np.int32)
                color = det['color']
                conf = det['confidence']
                
                # Draw oriented bounding box
                cv2.polylines(result_image, [points], True, color, thickness)
                
                # Calculate position for confidence label
                center_x = int(np.mean(points[:, 0]))
                center_y = int(np.mean(points[:, 1]))
                
                # Find the leftmost and rightmost points of the bounding box
                leftmost_x = np.min(points[:, 0])
                rightmost_x = np.max(points[:, 0])
                
                # Draw confidence label
                conf_text = f"{conf:.2f}"
                conf_font_scale = 0.6 * scale
                conf_font_thickness = max(1, int(2 * scale))
                conf_size = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, conf_font_scale, conf_font_thickness)[0]
                
                # Check if text fits on the right side
                image_width = result_image.shape[1]
                right_space = image_width - rightmost_x - 10
                left_space = leftmost_x - 10
                
                if right_space >= conf_size[0] + 10:  # Enough space on right
                    # Position label to the right of the detection
                    conf_x = rightmost_x + 10
                    conf_y = center_y
                else:
                    # Position label to the left of the detection
                    conf_x = leftmost_x - conf_size[0] - 10
                    conf_y = center_y
                
                # Background rectangle for confidence
                conf_bg_x1 = conf_x - 5
                conf_bg_y1 = conf_y - conf_size[1] - 5
                conf_bg_x2 = conf_x + conf_size[0] + 5
                conf_bg_y2 = conf_y + 5
                
                cv2.rectangle(result_image, (conf_bg_x1, conf_bg_y1), (conf_bg_x2, conf_bg_y2), color, -1)
                cv2.rectangle(result_image, (conf_bg_x1, conf_bg_y1), (conf_bg_x2, conf_bg_y2), (255, 255, 255), thickness)
                cv2.putText(result_image, conf_text, (conf_x, conf_y), 
                          cv2.FONT_HERSHEY_SIMPLEX, conf_font_scale, (255, 255, 255), conf_font_thickness)

        # Display the image
        self.result_image = result_image
        self.display_image(result_image)

def main():
    root = tk.Tk()
    app = YOLOGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 