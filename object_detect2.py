import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from collections import Counter
from pathlib import Path

# Set page configuration
st.set_page_config(layout="wide", page_title="Image Object Detection")
st.write("## Object Detection in Images")
st.write(":mag: Upload an image to detect objects. Detected objects will be highlighted with bounding boxes.")
st.sidebar.write("## Upload and Analyze :gear:")

# Set a flag to track if an image has been uploaded
image_uploaded = False

@st.cache(allow_output_mutation=True)
def load_classes():
    classes_file = str(Path(__file__).parent/'yolo/yolov3.txt')
    with open(classes_file, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    return classes

@st.cache(allow_output_mutation=True)
def load_model():
    config_path = str(Path(__file__).parent/'yolo/yolov3.cfg')
    weights_path = str(Path(__file__).parent/'yolo/yolov3.weights')
    net = cv2.dnn.readNet(weights_path, config_path)
    return net

def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = net.getUnconnectedOutLayers()
    if output_layers.ndim > 1:
        output_layers = output_layers.reshape(-1)
    return [layer_names[i - 1] for i in output_layers]

def process_and_display_image(upload, filename):
    image = Image.open(upload).convert('RGB')
    open_cv_image = np.array(image)[:, :, ::-1].copy()  # Convert RGB to BGR

    net = load_model()
    classes = load_classes()
    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
    
    blob = cv2.dnn.blobFromImage(open_cv_image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    Width = open_cv_image.shape[1]
    Height = open_cv_image.shape[0]

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    detected_objects = []
    for i in indices:
        i = i[0] if isinstance(i, np.ndarray) else i
        box = boxes[i]
        x, y, w, h = box
        detected_objects.append(classes[class_ids[i]])
        draw_prediction(open_cv_image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h), classes, COLORS)

    result_image = open_cv_image[:, :, ::-1]  # Convert BGR to RGB before displaying
    col1, col2 = st.columns(2)
    col1.write("Original Image :camera:")
    col1.image(image)

    col2.write("Detected Objects :mag:")
    col2.image(result_image)

    if detected_objects:
        object_count = Counter(detected_objects)
        detected_summary = ", ".join([f"{v} {k}(s)" for k, v in object_count.items()])
        st.write(f"### Detected {len(detected_objects)} object(s):")
        st.write(f"{detected_summary}")

    # Download Functionality
    result_bytes, result_name = convert_image(Image.fromarray(result_image), filename)
    st.sidebar.download_button("Download Result", result_bytes, result_name, "image/jpeg")

def convert_image(img, filename):
    buf = BytesIO()
    img.save(buf, format="JPEG")
    byte_im = buf.getvalue()
    return byte_im, f"{filename.split('.')[0]}_result.jpeg"

# Streamlit file uploader
my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if my_upload is not None:
    process_and_display_image(upload=my_upload, filename=my_upload.name)
    image_uploaded = True

if not image_uploaded:
    col1, col2 = st.columns(2)
    col1.write("Original Image :camera:")
    # Display a default image if no image is uploaded
    orig_img = Path(__file__).parent / 'image/people_img.jpg'
    with orig_img.open('rb') as f:
        col1.image(Image.open(f))

    col2.write("Detected Objects :mag:")
    # Display a processed image example if no image is uploaded
    pro_img = Path(__file__).parent / 'image/pro_img.jpg'
    with pro_img.open('rb') as f:
        col2.image(Image.open(f))
    
    st.write("### Detected 2 object(s):")
    st.write("2 person(s)")
