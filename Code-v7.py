
# Import image dataset file bu using the google drive id
!gdown --id 1i8D3bvLy8_vIbbn3gQXLbYYzNi3Kw7ED

# Extract zip file
!unzip '/content/archive.zip'

# Install all the requires libraries
!pip install torch torchvision ultralytics opencv-python matplotlib numpy tqdm

# streamlit_app.py
import streamlit as st
import pandas as pd

st.title("My App")

# Import Required Libraries
import torch
import torchvision
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import time
import random
from tqdm import tqdm
from ultralytics import YOLO
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.ops import box_iou
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from torchvision import transforms

# -------------------------------
# Model Initialization
# -------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load YOLO model (Using YOLOv3 pretrained weights)
yolo_model = YOLO("yolov3.pt")  # Ensure you have downloaded yolov3.pt beforehand

# Load Faster R-CNN for Object Detection
faster_rcnn_model = fasterrcnn_resnet50_fpn(pretrained=True)
faster_rcnn_model.eval()

# Load DeepLabV3+ for Road Segmentation
deeplab_model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
deeplab_model.eval()

# Load SegFormer (Transformer-Based Model for future use/development)
feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
segformer_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

# -------------------------------
# Helper Functions
# -------------------------------

def process_image(img_path):
    """Run object detection and segmentation on a single image and return results along with processing time."""
    image = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    start_time = time.time()
    # YOLO Object Detection
    conf, iou_thresh, max_det = 0.5, 0.5, 50
    yolo_results = yolo_model(img_path, conf=conf, iou=iou_thresh, max_det=max_det)
    yolo_img = yolo_results[0].plot()  # Annotated image
    yolo_img_rgb = cv2.cvtColor(yolo_img, cv2.COLOR_BGR2RGB)

    # Faster R-CNN Object Detection for evaluation (using tensor version of image)
    image_tensor = torchvision.transforms.ToTensor()(image).unsqueeze(0)
    with torch.no_grad():
        frcnn_results = faster_rcnn_model(image_tensor)

    # DeepLabV3+ Segmentation
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(image_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        output = deeplab_model(input_tensor)["out"][0]
    segmentation = output.argmax(0).byte().cpu().numpy()

    # Create segmentation colormap (example: red for a specific class, green for another)
    colormap = np.zeros((segmentation.shape[0], segmentation.shape[1], 3), dtype=np.uint8)
    colormap[segmentation == 15] = [255, 0, 0]  # Example: Person in red (if class 15)
    colormap[segmentation == 7] = [0, 255, 0]   # Example: Car in green (if class 7)

    process_time = time.time() - start_time

    # Compute performance metrics (IoU and placeholder for mAP)
    mean_iou = None
    mAP = None
    if yolo_results and len(yolo_results[0].boxes) > 0:
        pred_boxes = torch.tensor([box.xyxy[0].tolist() for box in yolo_results[0].boxes])
        gt_boxes = frcnn_results[0]['boxes'] if len(frcnn_results[0]['boxes']) > 0 else torch.empty(0)
        if pred_boxes.numel() > 0 and gt_boxes.numel() > 0:
            mean_iou = box_iou(pred_boxes, gt_boxes).mean().item()
        else:
            mean_iou = 0.0
        # Note: mAP calculation would require a full evaluation pipeline comparing with ground truth labels.
        mAP = 0.0  # Placeholder value; implement mAP calculation as needed.

    results = {
        "original": image_rgb,
        "yolo_detection": yolo_img_rgb,
        "segmentation": colormap,
        "process_time": process_time,
        "mean_iou": mean_iou,
        "mAP": mAP,
    }
    return results

def evaluate_model(dataset_folder, num_images=10):
    """Evaluate the model on a new dataset folder, compute FPS, mean IoU, and report results."""
    image_paths = glob.glob(os.path.join(dataset_folder, "*.jpg"))
    selected_images = random.sample(image_paths, min(num_images, len(image_paths)))

    total_time = 0.0
    iou_list = []
    map_list = []  # Currently placeholder
    for img_path in tqdm(selected_images, desc="Evaluating images"):
        result = process_image(img_path)
        total_time += result["process_time"]
        if result["mean_iou"] is not None:
            iou_list.append(result["mean_iou"])
        if result["mAP"] is not None:
            map_list.append(result["mAP"])

    avg_time = total_time / len(selected_images)
    fps = 1 / avg_time if avg_time > 0 else 0
    avg_iou = sum(iou_list) / len(iou_list) if iou_list else 0.0
    avg_map = sum(map_list) / len(map_list) if map_list else 0.0

    metrics = {
        "Average Processing Time (s)": avg_time,
        "FPS": fps,
        "Mean IoU": avg_iou,
        "mAP (placeholder)": avg_map
    }

    print("Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.2f}")
    return metrics

# Extract zip file
!unzip '/content/road_dataset.zip'

# -------------------------------
# Testing on New Dataset
# -------------------------------

# Example: Set path to new dataset folder
new_dataset_folder = "/content/trafic_data/train/images"  # Update with the new dataset path
if os.path.exists(new_dataset_folder):
    metrics = evaluate_model(new_dataset_folder, num_images=10)
else:
    print("New dataset folder not found. Please update the path.")

# -------------------------------
# Deployment: Simple Flask Web App
# -------------------------------

# For deployment, we set up a Flask app to serve the model.
from flask import Flask, request, render_template, redirect, url_for, send_from_directory

app = Flask(__name__)
UPLOAD_FOLDER = "./uploads"
RESULT_FOLDER = "./results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Save uploaded file
        if "file" not in request.files:
            return "No file uploaded", 400
        file = request.files["file"]
        if file.filename == "":
            return "No file selected", 400
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Process the uploaded image
        result = process_image(filepath)
        # Save the YOLO detection output for display (or choose any result)
        result_filename = f"result_{file.filename}"
        result_path = os.path.join(RESULT_FOLDER, result_filename)
        cv2.imwrite(result_path, cv2.cvtColor(result["yolo_detection"], cv2.COLOR_RGB2BGR))

        return redirect(url_for("result", filename=result_filename))
    return render_template("index.html")

@app.route("/result/<filename>")
def result(filename):
    return render_template("result.html", result_image=url_for("uploaded_file", filename=filename))

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(RESULT_FOLDER, filename)

if __name__ == "__main__":
    # Running in debug mode for testing; for production consider using a WSGI server
    app.run(debug=True)

"""----------------------------------

-------------------------
"""

import time
import cv2
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO
from torchvision.ops import box_iou

# Example: YOLO model initialization
yolo_model = YOLO("yolov3.pt")  # or "yolov8n.pt" etc.

def detect_and_segment(img_path, faster_rcnn_model, deeplab_model, device):
    """Perform YOLO detection, Faster R-CNN detection (for IoU), and DeepLabV3+ segmentation."""
    # Read image
    image_bgr = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # -----------------------------
    # 1) YOLO Detection
    # -----------------------------
    # Run inference
    conf, iou_thresh, max_det = 0.5, 0.5, 50
    yolo_results = yolo_model(img_path, conf=conf, iou=iou_thresh, max_det=max_det)

    # Convert YOLO’s annotated result to RGB for plotting
    yolo_annot_bgr = yolo_results[0].plot()  # This is BGR
    yolo_annot_rgb = cv2.cvtColor(yolo_annot_bgr, cv2.COLOR_BGR2RGB)

    # -----------------------------
    # 2) Faster R-CNN (for IoU calc)
    # -----------------------------
    with torch.no_grad():
        # Convert image to tensor
        image_tensor = torch.from_numpy(image_bgr[..., ::-1].copy().transpose(2, 0, 1)).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0)  # shape: (1, C, H, W)
        frcnn_results = faster_rcnn_model(image_tensor)

    # -----------------------------
    # 3) DeepLabV3+ Segmentation
    # -----------------------------
    import torchvision.transforms as T
    preprocess = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(image_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        seg_output = deeplab_model(input_tensor)["out"][0]
    seg_mask = seg_output.argmax(0).byte().cpu().numpy()

    # Create simple colormap for a couple of classes
    # e.g., 15 = Person (red), 7 = Car (green)
    seg_colormap = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) * 0  # blank canvas
    seg_colormap[seg_mask == 15] = [255, 0, 0]  # red
    seg_colormap[seg_mask == 7]  = [0, 255, 0]  # green

    # -----------------------------
    # Compute IoU
    # -----------------------------
    mean_iou = 0.0
    if len(yolo_results[0].boxes) > 0:
        # YOLO predicted boxes
        pred_boxes = torch.tensor([box.xyxy[0].tolist() for box in yolo_results[0].boxes])
        # Faster R-CNN boxes
        gt_boxes = frcnn_results[0]['boxes'] if len(frcnn_results[0]['boxes']) > 0 else torch.empty(0)

        if pred_boxes.numel() > 0 and gt_boxes.numel() > 0:
            mean_iou = box_iou(pred_boxes, gt_boxes).mean().item()

    return image_rgb, yolo_annot_rgb, seg_colormap, yolo_results, mean_iou

def display_results(img_path, image_rgb, yolo_annot_rgb, seg_colormap, yolo_results, mean_iou):
    """Display side-by-side results and print textual metrics (speed, shape, etc.)."""
    # Print YOLO textual output
    # This prints bounding boxes, time, etc. for the first result in yolo_results
    print(yolo_results[0].summary())

    # Alternatively, we can manually print speed + shape
    speeds = yolo_results[0].speed  # dictionary: {'preprocess': x, 'inference': y, 'postprocess': z}
    pre_ms, inf_ms, post_ms = speeds['preprocess'], speeds['inference'], speeds['postprocess']
    orig_shape = yolo_results[0].orig_shape  # (H, W)

    print(
        f"image 1/1 {img_path}, shape: {orig_shape}, "
        f"Speed: {pre_ms:.1f}ms preprocess, {inf_ms:.1f}ms inference, {post_ms:.1f}ms postprocess "
        f"per image at shape (1, 3, {orig_shape[0]}, {orig_shape[1]})"
    )
    print(f"Mean IoU: {mean_iou:.2f}")

    # Show side-by-side figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image_rgb)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(yolo_annot_rgb)
    axes[1].set_title("YOLOv8 Detection")
    axes[1].axis('off')

    axes[2].imshow(seg_colormap)
    axes[2].set_title("DeepLabV3+ Segmentation")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()


# -----------------------------
# Usage Example
# -----------------------------
if __name__ == "__main__":
    import torchvision
    # Load Faster R-CNN
    faster_rcnn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    faster_rcnn_model.eval()

    # Load DeepLabV3+
    deeplab_model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
    deeplab_model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    deeplab_model.to(device)

    # Test on a single image from your new dataset
    test_image_path = '/content/trafic_data/train/images/02_jpg.rf.65a084066fc353cd023eb5c953f40efe.jpg'  # Change to an actual path
    image_rgb, yolo_annot_rgb, seg_colormap, yolo_results, mean_iou = detect_and_segment(
        test_image_path,
        faster_rcnn_model,
        deeplab_model,
        device
    )
    display_results(
        test_image_path,
        image_rgb,
        yolo_annot_rgb,
        seg_colormap,
        yolo_results,
        mean_iou
    )

# -------------------------------
# Deployment: Simple Flask Web App
# -------------------------------

# For deployment, we set up a Flask app to serve the model.
from flask import Flask, request, render_template, redirect, url_for, send_from_directory

app = Flask(__name__)
UPLOAD_FOLDER = "./uploads"
RESULT_FOLDER = "./results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Save uploaded file
        if "file" not in request.files:
            return "No file uploaded", 400
        file = request.files["file"]
        if file.filename == "":
            return "No file selected", 400
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Process the uploaded image
        result = process_image(filepath)
        # Save the YOLO detection output for display (or choose any result)
        result_filename = f"result_{file.filename}"
        result_path = os.path.join(RESULT_FOLDER, result_filename)
        cv2.imwrite(result_path, cv2.cvtColor(result["yolo_detection"], cv2.COLOR_RGB2BGR))

        return redirect(url_for("result", filename=result_filename))
    return render_template("index.html")

@app.route("/result/")
def result(filename):
    return render_template("result.html", result_image=url_for("uploaded_file", filename=filename))

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(RESULT_FOLDER, filename)

if __name__ == "__main__":
    # Running in debug mode for testing; for production consider using a WSGI server
    app.run(debug=True)

import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from ultralytics import YOLO
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
from torchvision.ops import box_iou
from torchvision import transforms
import time

# -------------------------------
# Model Initialization
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Use the improved YOLOv3u model as suggested
yolo_model = YOLO("yolov3u.pt")  # Ensure the 'yolov3u.pt' file is available

# Load Faster R-CNN with updated weights parameter
faster_rcnn_model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
faster_rcnn_model.eval()

# Load DeepLabV3+ with updated weights parameter
deeplab_model = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1)
deeplab_model.eval()
deeplab_model.to(device)

# -------------------------------
# Helper Functions
# -------------------------------
def detect_and_segment(img_path):
    """
    Run YOLO detection, Faster R-CNN (for IoU) and DeepLabV3+ segmentation on the image at img_path.
    Returns the original image, annotated detection image, segmentation colormap, the YOLO results, and mean IoU.
    """
    # Load image with OpenCV
    image_bgr = cv2.imread(img_path)
    if image_bgr is None:
        raise ValueError(f"Unable to read image from {img_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # -----------------------------
    # YOLO Detection
    # -----------------------------
    conf, iou_thresh, max_det = 0.5, 0.5, 50
    start_time = time.time()
    yolo_results = yolo_model(img_path, conf=conf, iou=iou_thresh, max_det=max_det)
    yolo_inference_time = time.time() - start_time

    # Get annotated image
    yolo_annot_bgr = yolo_results[0].plot()  # BGR image
    yolo_annot_rgb = cv2.cvtColor(yolo_annot_bgr, cv2.COLOR_BGR2RGB)

    # -----------------------------
    # Faster R-CNN for IoU Calculation
    # -----------------------------
    # Convert image to tensor (avoid negative strides by calling .copy())
    image_tensor = torch.from_numpy(image_bgr[..., ::-1].copy().transpose(2, 0, 1)).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0)
    with torch.no_grad():
        frcnn_results = faster_rcnn_model(image_tensor)

    # -----------------------------
    # DeepLabV3+ Segmentation
    # -----------------------------
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(image_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        seg_output = deeplab_model(input_tensor)["out"][0]
    seg_mask = seg_output.argmax(0).byte().cpu().numpy()

    # Create segmentation colormap (example: red for persons, green for cars)
    seg_colormap = np.zeros_like(image_rgb)
    seg_colormap[seg_mask == 15] = [255, 0, 0]  # Person (red)
    seg_colormap[seg_mask == 7]  = [0, 255, 0]   # Car (green)

    # -----------------------------
    # Compute Mean IoU
    # -----------------------------
    mean_iou = 0.0
    if yolo_results and yolo_results[0].boxes is not None and len(yolo_results[0].boxes) > 0:
        # Extract YOLO predicted boxes
        pred_boxes = torch.tensor([box.xyxy[0].tolist() for box in yolo_results[0].boxes])
        # Extract Faster R-CNN boxes
        gt_boxes = frcnn_results[0]['boxes'] if len(frcnn_results[0]['boxes']) > 0 else torch.empty(0)
        if pred_boxes.numel() > 0 and gt_boxes.numel() > 0:
            mean_iou = box_iou(pred_boxes, gt_boxes).mean().item()

    return image_rgb, yolo_annot_rgb, seg_colormap, yolo_results, mean_iou, yolo_inference_time

def save_side_by_side(original, detection, segmentation, save_path):
    """Save a side-by-side image comparing original, detection, and segmentation."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    axes[1].imshow(detection)
    axes[1].set_title("YOLO Detection")
    axes[1].axis('off')
    axes[2].imshow(segmentation)
    axes[2].set_title("DeepLabV3+ Segmentation")
    axes[2].axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Flask Web App Setup
app = Flask(__name__)
UPLOAD_FOLDER = "./uploads"
RESULT_FOLDER = "./results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")
        if not file:
            return "No file uploaded!", 400

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Process Image
        orig, det, seg, _, mean_iou, inf_time = detect_and_segment(filepath)
        result_filename = f"result_{file.filename}"
        result_path = os.path.join(RESULT_FOLDER, result_filename)
        save_side_by_side(orig, det, seg, result_path)

        return render_template_string(
            '''
            <h1>Object Detection Results</h1>
            <img src="{{ url_for('result_file', filename=result_filename) }}" width="600">
            <p>Inference Time: {{ inf_time }} ms</p>
            <p>Mean IoU: {{ mean_iou }}</p>
            <a href="/">Upload Another Image</a>
            ''',
            result_filename=result_filename, inf_time=round(inf_time*1000, 2), mean_iou=round(mean_iou, 2)
        )
    return render_template_string(
        '''
        <h1>Upload an Image for Detection</h1>
        <form method="post" enctype="multipart/form-data">
            <input type="file" name="file">
            <input type="submit" value="Upload">
        </form>
        ''')

@app.route("/results/<filename>")
def result_file(filename):
    return send_from_directory(RESULT_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)

import os
import glob
import random
import cv2
import torch
import torchvision.transforms as transforms
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from ultralytics import YOLO
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
yolo_model = YOLO("yolov8n.pt")
faster_rcnn_model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).to(device).eval()
deeplab_model = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT).to(device).eval()

# Define paths for original and new datasets
original_image_folder = "./bdd100k/images/10k/val/"
new_image_folder = "./new_test_dataset/"
output_folder = "./results/"
os.makedirs(output_folder, exist_ok=True)

def process_images(image_folder, dataset_name):
    image_paths = glob.glob(os.path.join(image_folder, "*.jpg"))
    random_images = random.sample(image_paths, min(10, len(image_paths)))
    dataset_output_folder = os.path.join(output_folder, dataset_name)
    os.makedirs(dataset_output_folder, exist_ok=True)

    for img_path in random_images:
        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # YOLO Object Detection
        yolo_results = yolo_model(img_path, conf=0.5, iou=0.5, max_det=50)
        yolo_img = yolo_results[0].plot()
        yolo_img_rgb = cv2.cvtColor(yolo_img, cv2.COLOR_BGR2RGB)

        # Faster R-CNN Object Detection
        image_tensor = torchvision.transforms.ToTensor()(image).unsqueeze(0).to(device)
        with torch.no_grad():
            frcnn_results = faster_rcnn_model(image_tensor)

        # DeepLabV3+ Segmentation
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        input_tensor = preprocess(image_rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            output = deeplab_model(input_tensor)["out"][0]
        segmentation = output.argmax(0).byte().cpu().numpy()

        # Save images
        output_path = os.path.join(dataset_output_folder, os.path.basename(img_path))
        cv2.imwrite(output_path, cv2.cvtColor(yolo_img, cv2.COLOR_RGB2BGR))

        print(f"Processed and saved: {output_path}")

# Process original and new datasets
process_images(original_image_folder, "original")
process_images(new_image_folder, "new_test")

# Flask App for Hosting
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/results/<filename>')
def results(filename):
    return send_from_directory(output_folder, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

"""--------------------------------------
--------------------------------------
--------------------------------------
"""

# Install dependencies
!pip install torch torchvision ultralytics opencv-python matplotlib numpy tqdm flask pyngrok

# Import libraries
import torch
import torchvision
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import time
import random
from tqdm import tqdm
from ultralytics import YOLO
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.ops import box_iou
from torchvision import transforms
from flask import Flask, request, render_template_string, send_from_directory
from pyngrok import ngrok
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------------
# Data Preparation
# ---------------------------

# Original Dataset (BDD100K)
!gdown --id 1i8D3bvLy8_vIbbn3gQXLbYYzNi3Kw7ED -O /content/archive.zip
!unzip -q '/content/archive.zip' -d '/content/bdd100k'

# New Test Dataset (Replace with your dataset ID)
!gdown --id 1460hpGSD2yVazBvdKvfsN41krYZEG3yT -O /content/new_test.zip
!unzip -q '/content/new_test.zip' -d '/content/new_test'

# ---------------------------
# Model Initialization
# ---------------------------

# Download YOLOv3 weights
!wget https://github.com/ultralytics/yolov3/releases/download/v9.6/yolov3.pt

# Initialize models
yolo_model = YOLO("yolov3.pt")
# Load Faster R-CNN for Object Detection
faster_rcnn_model = fasterrcnn_resnet50_fpn(pretrained=True)
faster_rcnn_model.eval()

# Load DeepLabV3+ for Road Segmentation
deeplab_model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
deeplab_model.eval()

# Load SegFormer (Transformer-Based Model for Future Work)
feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
segformer_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

def process_data(dataset_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    image_paths = glob.glob(os.path.join(dataset_path, "*.jpg"))
    sampled_images = random.sample(image_paths, min(10, len(image_paths)))

    for img_path in tqdm(sampled_images, desc="Processing Dataset"):
        # Load and preprocess image
        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # YOLO Object Detection
        conf, iou, max_det = 0.5, 0.5, 50
        yolo_results = yolo_model(img_path, conf=conf, iou=iou, max_det=max_det)
        yolo_img = yolo_results[0].plot()
        yolo_img_rgb = cv2.cvtColor(yolo_img, cv2.COLOR_BGR2RGB)  # Convert to RGB

        # Faster R-CNN Object Detection
        image_tensor = torchvision.transforms.ToTensor()(image).unsqueeze(0)
        with torch.no_grad():
            frcnn_results = faster_rcnn_model(image_tensor)

        # DeepLabV3+ Segmentation
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        input_tensor = preprocess(image_rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            output = deeplab_model(input_tensor)["out"][0]

        segmentation = output.argmax(0).byte().cpu().numpy()

        # Create segmentation colormap
        colormap = np.zeros((segmentation.shape[0], segmentation.shape[1], 3), dtype=np.uint8)
        colormap[segmentation == 15] = [255, 0, 0]  # Red for Person
        colormap[segmentation == 7] = [0, 255, 0]  # Green for Car

        # Compute IoU & mAP (if objects detected)
        if yolo_results and len(yolo_results[0].boxes) > 0:
            pred_boxes = torch.tensor([box.xyxy[0].tolist() for box in yolo_results[0].boxes])
            gt_boxes = frcnn_results[0]['boxes'] if len(frcnn_results[0]['boxes']) > 0 else torch.empty(0)

            if pred_boxes.numel() > 0 and gt_boxes.numel() > 0:
                iou = box_iou(pred_boxes, gt_boxes).mean().item()
                print(f"Mean IoU: {iou:.2f}")
            else:
                print("No valid ground truth boxes for IoU calculation.")

        # Save image
        output_path = os.path.join(output_folder, os.path.basename(img_path))
        cv2.imwrite(output_path, cv2.cvtColor(yolo_img, cv2.COLOR_RGB2BGR))

        # Display results side-by-side
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(image_rgb)
        axes[0].set_title("Original Image")
        axes[1].imshow(yolo_img_rgb)
        axes[1].set_title("YOLOv8 Detection")
        axes[2].imshow(colormap)
        axes[2].set_title("DeepLabV3+ Segmentation")
        plt.show()

# Process both datasets
print("Initial Training Model")
process_data("/content/bdd100k/bdd100k/images/10k/val/", "/content/original_results")
print("-----------------------------")

print("New Dataset Test Model")
process_data("/content/new_test/trafic_data/train/images", "/content/new_test_results")
print("---------------------------------")

# Make sure your models and device are already initialized
# For example:
# from ultralytics import YOLO
# from torchvision.models.detection import fasterrcnn_resnet50_fpn
# yolo_model = YOLO("yolov3.pt")
# faster_rcnn_model = fasterrcnn_resnet50_fpn(pretrained=True)
# faster_rcnn_model.eval()
# deeplab_model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
# deeplab_model.eval()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Flask Web App Setup
app = Flask(__name__)
UPLOAD_FOLDER = "./uploads"
RESULT_FOLDER = "./results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")
        if not file:
            return "No file uploaded!", 400

        # Save the uploaded file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # For this example, we consider the UPLOAD_FOLDER as the dataset.
        # When only one file is uploaded, process_data will process that image.
        process_data(UPLOAD_FOLDER, RESULT_FOLDER)

        # Assume the processed image is saved with the same filename in RESULT_FOLDER
        result_filename = file.filename

        return render_template_string(
            '''
            <h1>Object Detection Results</h1>
            <img src="{{ url_for('result_file', filename=result_filename) }}" width="600">
            <a href="/">Upload Another Image</a>
            ''',
            result_filename=result_filename
        )
    return render_template_string(
        '''
        <h1>Upload an Image for Detection</h1>
        <form method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept=".jpg,.jpeg,.png" required>
            <input type="submit" value="Upload">
        </form>
        '''
    )

@app.route("/results/<filename>")
def result_file(filename):
    return send_from_directory(RESULT_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)

# For deployment, we set up a Flask app to serve the model.
from flask import Flask, request, render_template, redirect, url_for, send_from_directory

app = Flask(__name__)
UPLOAD_FOLDER = "./uploads"
RESULT_FOLDER = "./results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Save uploaded file
        if "file" not in request.files:
            return "No file uploaded", 400
        file = request.files["file"]
        if file.filename == "":
            return "No file selected", 400
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Process the uploaded image
        result = process_image(filepath)
        # Save the YOLO detection output for display (or choose any result)
        result_filename = f"result_{file.filename}"
        result_path = os.path.join(RESULT_FOLDER, result_filename)
        cv2.imwrite(result_path, cv2.cvtColor(result["yolo_detection"], cv2.COLOR_RGB2BGR))

        return redirect('''
            <h1>Object Detection Results</h1>
            <img src="{{ url_for('result_file', filename=result_filename) }}" width="600">
            <a href="/">Upload Another Image</a>
            ''',
            result_filename=result_filename)
    return render_template('''
        <h1>Upload an Image for Detection</h1>
        <form method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept=".jpg,.jpeg,.png" required>
            <input type="submit" value="Upload">
        </form>
        ''')

@app.route("/result/")
def result(filename):
    return render_template("result.html", result_image=url_for("uploaded_file", filename=filename))

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(RESULT_FOLDER, filename)

if __name__ == "__main__":
    # Running in debug mode for testing; for production consider using a WSGI server
    app.run(debug=True)