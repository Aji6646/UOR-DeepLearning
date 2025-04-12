import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision import transforms
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation

# Title
st.title("Road Object Detection")

# Sidebar model selection
model_choice = st.sidebar.selectbox("Choose a model", ("YOLOv8", "Faster R-CNN", "DeepLabV3", "SegFormer"))

# Image uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    processed_image_cv = image_cv.copy()

    if model_choice == "YOLOv8":
        st.write("Running YOLOv8...")
        model = YOLO("yolov8n.pt")
        results = model(image_cv)[0]

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            cv2.rectangle(processed_image_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(processed_image_cv, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    elif model_choice == "Faster R-CNN":
        st.write("Running Faster R-CNN...")
        model = fasterrcnn_resnet50_fpn(pretrained=True)
        model.eval()
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        input_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            predictions = model(input_tensor)[0]

        for box, label, score in zip(predictions['boxes'], predictions['labels'], predictions['scores']):
            if score > 0.5:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(processed_image_cv, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(processed_image_cv, f"Label {label.item()} {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

    elif model_choice == "DeepLabV3":
        st.write("Running DeepLabV3...")
        model = deeplabv3_resnet50(pretrained=True).eval()
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)['out'][0]
        output_predictions = output.argmax(0).byte().cpu().numpy()
        mask = cv2.applyColorMap((output_predictions * 10).astype(np.uint8), cv2.COLORMAP_JET)
        processed_image_cv = cv2.addWeighted(image_cv, 0.6, mask, 0.4, 0)

    elif model_choice == "SegFormer":
        st.write("Running SegFormer...")
        feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

        inputs = feature_extractor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        segmentation = logits.argmax(dim=1)[0].cpu().numpy()
        mask = cv2.applyColorMap((segmentation * 10).astype(np.uint8), cv2.COLORMAP_TURBO)
        processed_image_cv = cv2.addWeighted(image_cv, 0.6, mask, 0.4, 0)

    # Convert final output to RGB
    output_image = cv2.cvtColor(processed_image_cv, cv2.COLOR_BGR2RGB)

    # Display original and output images side by side
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original Image", use_column_width=True)
    with col2:
        st.image(output_image, caption="Detection / Segmentation Result", use_column_width=True)

    # Add download button
    result_pil = Image.fromarray(output_image)
    st.download_button(
        label="Download Result Image",
        data=result_pil.tobytes(),
        file_name="result.png",
        mime="image/png"
    )
