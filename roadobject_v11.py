import os
import random
import time
import zipfile
import urllib.request
import gdown
import cv2
import numpy as np
import torch
from flask import Flask, request, render_template_string, send_from_directory, url_for
from pyngrok import ngrok, conf
import matplotlib.pyplot as plt
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision import transforms
from torchvision.ops import box_iou
from ultralytics import YOLO
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from tqdm import tqdm
from dotenv import load_dotenv

# ========== ENV & AUTH SETUP ==========
load_dotenv()
ngrok.set_auth_token(os.getenv("NGROK_AUTH_TOKEN"))
conf.get_default().region = "us"
conf.get_default().bind_tls = True

# ========== PATH SETUP ==========
UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# ========== DEVICE ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ========== DOWNLOAD FILES ==========
def download_dataset_files():
    os.makedirs("data", exist_ok=True)
    if not os.path.exists("data/archive.zip"):
        gdown.download(id="1i8D3bvLy8_vIbbn3gQXLbYYzNi3Kw7ED", output="data/archive.zip", quiet=False)
    if not os.path.exists("data/new_test.zip"):
        gdown.download(id="1460hpGSD2yVazBvdKvfsN41krYZEG3yT", output="data/new_test.zip", quiet=False)

    with zipfile.ZipFile("data/archive.zip", 'r') as zip_ref:
        zip_ref.extractall("data/bdd100k")
    with zipfile.ZipFile("data/new_test.zip", 'r') as zip_ref:
        zip_ref.extractall("data/new_test")

    if not os.path.exists("yolov3.pt"):
        urllib.request.urlretrieve("https://github.com/ultralytics/yolov3/releases/download/v9.6/yolov3.pt", "yolov3.pt")

download_dataset_files()

# ========== MODEL INITIALIZATION ==========
yolo_model = YOLO("yolov3.pt")
faster_rcnn_model = fasterrcnn_resnet50_fpn(pretrained=True).eval()
deeplab_model = torch.hub.load('pytorch/vision:v0.13.0', 'deeplabv3_resnet101', pretrained=True).eval()

feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
segformer_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

# ========== MAIN PROCESS FUNCTION ==========
def process_data(dataset_path, output_folder, is_flask=False):
    os.makedirs(output_folder, exist_ok=True)

    if os.path.isfile(dataset_path):
        processing_list = [dataset_path]
    else:
        image_paths = glob.glob(os.path.join(dataset_path, "*.jpg"))
        sampled_images = random.sample(image_paths, min(10, len(image_paths)))
        processing_list = sampled_images

    metrics_list = []
    iterator = tqdm(processing_list, desc="Processing") if not is_flask else processing_list

    for img_path in iterator:
        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # YOLO Detection
        yolo_start = time.time()
        yolo_results = yolo_model(img_path, conf=0.5, iou=0.5, max_det=50)
        yolo_time = (time.time() - yolo_start) * 1000

        yolo_img = yolo_results[0].plot()
        yolo_filename = os.path.basename(img_path)
        cv2.imwrite(os.path.join(output_folder, yolo_filename), cv2.cvtColor(yolo_img, cv2.COLOR_RGB2BGR))

        # Segmentation
        seg_start = time.time()
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        input_tensor = preprocess(image_rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            output = deeplab_model(input_tensor)["out"][0]
        segmentation = output.argmax(0).byte().cpu().numpy()
        seg_time = (time.time() - seg_start) * 1000

        colormap = np.zeros((segmentation.shape[0], segmentation.shape[1], 3), dtype=np.uint8)
        colormap[segmentation == 15] = [255, 0, 0]  # Red for person
        colormap[segmentation == 7] = [0, 255, 0]   # Green for car
        seg_filename = f"seg_{os.path.basename(img_path)}"
        cv2.imwrite(os.path.join(output_folder, seg_filename), cv2.cvtColor(colormap, cv2.COLOR_RGB2BGR))

        pred_boxes = torch.tensor([box.xyxy[0].tolist() for box in yolo_results[0].boxes]) if yolo_results[0].boxes else torch.empty(0)
        frcnn_tensor = transforms.ToTensor()(image).unsqueeze(0)
        with torch.no_grad():
            frcnn_results = faster_rcnn_model(frcnn_tensor)
        gt_boxes = frcnn_results[0]['boxes'] if len(frcnn_results[0]['boxes']) > 0 else torch.empty(0)
        iou = box_iou(pred_boxes, gt_boxes).mean().item() if pred_boxes.numel() > 0 and gt_boxes.numel() > 0 else 0.0

        metrics = {
            "yolo_time": yolo_time,
            "seg_time": seg_time,
            "iou": iou
        }
        metrics_list.append(metrics)

    return metrics_list

# ========== FLASK APP ==========
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if not file:
            return "No file uploaded!", 400

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        metrics_list = process_data(filepath, RESULT_FOLDER, is_flask=True)
        metrics = metrics_list[0]

        original_url = url_for('upload_file', filename=file.filename)
        yolo_url = url_for('result_file', filename=file.filename)
        seg_url = url_for('result_file', filename=f"seg_{file.filename}")

        return render_template_string('''
            <div style="display:flex;flex-direction:column;align-items:center; justify-content:center">
            <h1>Results</h1>
            <div style="display: flex; gap: 20px;">
                <div><img src="{{ original_url }}" width="300"><br>Original</div>
                <div><img src="{{ yolo_url }}" width="300"><br>YOLO Detection</div>
                <div><img src="{{ seg_url }}" width="300"><br>Segmentation</div>
            </div>
            <h3 style="margin-top:30px;">Metrics</h3>
            <div style="width:250px; text-align:left;margin-top:10px;">
            <p>YOLO Inference: {{ metrics.yolo_time | round(1) }} ms</p>
            <p>Segmentation Inference: {{ metrics.seg_time | round(1) }} ms</p>
            <p>Mean IoU: {{ metrics.iou | round(2) }}</p>
            </div>
            <a href="/">Upload Another</a>
            </div>
        ''', original_url=original_url, yolo_url=yolo_url, seg_url=seg_url, metrics=metrics)

    return render_template_string('''
        <div style="display:flex;flex-direction:column; justify-content:center; align-items:center">
        <h1 style="text-align:center">Upload Image</h1>
        <form method="post" enctype="multipart/form-data" style="text-align:center">
            <input type="file" name="file" accept=".jpg,.jpeg,.png" required>
            <input type="submit" value="Upload">
        </form>
        </div>
    ''')

@app.route("/results/<filename>")
def result_file(filename):
    return send_from_directory(RESULT_FOLDER, filename)

@app.route("/uploads/<filename>")
def upload_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == "__main__":
    try:
        tunnel = ngrok.connect(5000)
        print(f"Public URL: {tunnel.public_url}")
        app.run(host="0.0.0.0", port=5000)
    except Exception as e:
        print(f"Error starting app: {e}")
