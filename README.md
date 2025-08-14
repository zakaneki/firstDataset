# Electrical Symbol Detection with YOLO OBB

This project detects electrical symbols in circuit diagrams and extracts connectivity (wires, junctions, element connections). It includes a simple GUI for running detection, visualizing results, and analyzing the connectivity graph.


## Project Overview

- **Model**: YOLO v11 OBB (Oriented Bounding Box)
- **Classes**: 4 electrical symbols
- **Input**: Circuit diagram images (1024x1024)
- **Output**: Oriented bounding boxes with class predictions

## 📁 Project Structure
```
circuit-vision/
├── augmented_images/         # Augmented images
├── images/                   # Source images
├── generated_images/         # Generated training images
├── generated_labels/         # Generated training labels
│   ├── frame_0_delay-0.01s.jpg
│   ├── single-line-diagram-substation-two-parallel-transformers.png
│   └── substation-with-single-transformer.png
├── labels/                   # Source labels (YOLO OBB format)
│   └── train/
├── augment_images.py         # Image augmentation script
├── best.pt                   # Trained model weights
├── data.yaml                 # Dataset configuration
├── elements.py               # Defines DetectedElement and symbol classes (Transformer, CircuitBreaker, etc.)
├── predictor.py              # SymbolPredictor: runs YOLO OBB detection, OCR, wire extraction, and connectivity analysis
├── prepare_yolo_data.py      # Prepares and formats data for YOLO OBB training
├── process_image.py          # Generates synthetic circuit images and YOLO OBB labels
├── yolo_gui.py               # Tkinter GUI for detection, visualization, and connectivity graph analysis
├── README.md
```

##  Quick Start

### 1. Generate Training Data

```bash
# Generate 250 synthetic images with electrical symbols
python process_image.py
```

This creates:
- `generated_images/` - 250 synthetic circuit diagrams
- `generated_labels/` - Corresponding YOLO OBB labels

### 2. Organize Dataset

```bash
# Create proper YOLO directory structure
mkdir -p images/train images/val labels/train labels/val

# Split generated data 80/20 (train/val)
# Copy 200 images to train, 50 to val
# Copy corresponding labels
```

### 3. Train the Model

```bash
# Train YOLO OBB model
yolo obb train model=yolo11m-obb.pt data=data.yaml epochs=100 imgsz=1024 batch=8 project=runs/train name=electrical_obb_v1
```

### 4. Predict on New Images

```bash
# Predict on a single image
yolo obb predict model=runs/train/electrical_obb_v1/weights/best.pt source=test_image.png

# Predict on a directory
yolo obb predict model=runs/train/electrical_obb_v1/weights/best.pt source=test_images/

# Predict with confidence threshold
yolo obb predict model=runs/train/electrical_obb_v1/weights/best.pt source=test_image.png conf=0.5
```

### 5. Launch the GUI for Detection & Analysis

```bash
python yolo_gui.py
```

- Use the GUI to upload images, run detection, and visualize results.
- Switch to the "Graph Analysis" tab to view and analyze the connectivity graph of detected elements and wires.


### 6. Experiment with OCR and Connectivity

- Try both EasyOCR and Doctr for text extraction (configurable in the GUI).
- Adjust detection and OCR parameters for best results on your data.

##  Training Details

### Model Configuration
- **Model**: YOLO v11 Medium OBB (`yolo11m-obb.pt`)
- **Input Size**: 1024x1024 pixels
- **Batch Size**: 8 (adjust based on GPU memory)
- **Epochs**: 100

### Dataset Information
- **Training Images**: 200 synthetic circuit diagrams
- **Validation Images**: 50 synthetic circuit diagrams
- **Classes**: 4 electrical symbols
- **Label Format**: YOLO OBB (8 coordinates per detection)

### Class Mapping
```
0: Transformer
1: Circuit Breaker  
2: Switch
3: MV Line
```

## 🔧 Data Generation Process

### 1. Symbol Extraction
The `process_image.py` script:
- Extracts electrical symbols from source images
- Organizes symbols by class type
- Ensures even distribution across classes

### 2. Synthetic Image Generation
For each generated image:
- Randomly selects symbols from each class
- Places symbols with pathfinding connections
- Adds circuit lines between symbols
- Generates YOLO OBB labels

### 3. Data Augmentation
The `augment_images.py` script:
- Applies random transformations
- Creates additional training samples
- Maintains label consistency

##  Training Metrics

The model is evaluated using:
- **mAP50-95**: Primary metric for model selection
- **mAP50**: Mean Average Precision at IoU=0.50
- **Precision**: Detection precision
- **Recall**: Detection recall

##  Prediction Results

After training, the model can:
- Detect electrical symbols in circuit diagrams
- Predict oriented bounding boxes
- Provide confidence scores
- Handle rotated and scaled symbols

## 📁 Output Files

### Training Outputs
- `runs/train/electrical_obb_v1/weights/best.pt` - Best model weights
- `runs/train/electrical_obb_v1/weights/last.pt` - Last epoch weights
- `runs/train/electrical_obb_v1/results.png` - Training curves
- `runs/train/electrical_obb_v1/confusion_matrix.png` - Confusion matrix

## 🛠️ Customization

### Adjust Training Parameters
```bash
# Different model sizes
yolo train model=yolo11n-obb.pt data=data.yaml  # Nano (fastest)
yolo train model=yolo11s-obb.pt data=data.yaml  # Small
yolo train model=yolo11m-obb.pt data=data.yaml  # Medium
yolo train model=yolo11l-obb.pt data=data.yaml  # Large
yolo train model=yolo11x-obb.pt data=data.yaml  # Extra Large

# Different training parameters
yolo train model=yolo11m-obb.pt data=data.yaml epochs=100 imgsz=640 batch=16 patience=50
```

### Modify Data Generation
Edit `process_image.py`:
- Change `num_images_to_generate` for different dataset sizes
- Adjust symbol placement parameters
- Modify line drawing settings

### Customize the GUI (`yolo_gui.py`)
- Change default OCR engine or add new OCR options.
- Adjust GUI layout, add new tabs, or modify result visualization.
- Update connectivity graph analysis or export features.
- Integrate additional post-processing or reporting tools.

## 🔍 Troubleshooting

### Common Issues

1. **Out of Memory**
   ```bash
   # Reduce batch size
   yolo train model=yolo11m-obb.pt data=data.yaml batch=4
   ```

2. **Slow Training**
   ```bash
   # Use smaller model
   yolo train model=yolo11n-obb.pt data=data.yaml
   ```

3. **Poor Performance**
   - Increase dataset size
   - Add more data augmentation
   - Train for more epochs

4. **Cache Issues**
   ```bash
   # Clear validation cache
   rm runs/val/*/val.cache
   ```

5. **GUI Issues**
   - If the GUI does not launch, ensure all dependencies (Tkinter, PIL, matplotlib, networkx) are installed.
   - For OCR errors, check that EasyOCR and Doctr are installed and properly configured.
   - If detection is slow in the GUI, try reducing image size.


## 📝 Requirements

- Python 3.8+
- PyTorch
- Ultralytics YOLO
- OpenCV
- NumPy
- Tkinter
- Pillow (PIL)
- matplotlib
- networkx
- EasyOCR
- Doctr

##  Contributing

Feel free to submit issues and enhancement requests!