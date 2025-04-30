from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import base64
import numpy as np
import cv2

app = Flask(__name__)

# Allow all origins (for testing purposes, but you should configure this for security in production)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load YOLOv8 model (can use yolov8n.pt for speed or yolov8s.pt/yolov8m.pt for accuracy)
model = YOLO('yolov8n.pt')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/product_indent')
def product_indent():
    return render_template('product_indent.html')

@app.route('/detect', methods=['POST'])
def detect():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify([])

    # Decode the base64 image
    encoded_data = data['image'].split(',')[1]
    image_data = base64.b64decode(encoded_data)
    np_arr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Run YOLOv8 detection
    results = model(img)[0]

    # Format detection results
    detections = []
    for box in results.boxes:
        detections.append({
            "name": results.names[int(box.cls)],
            "confidence": float(box.conf),
            "xmin": int(box.xyxy[0][0]),
            "ymin": int(box.xyxy[0][1]),
            "xmax": int(box.xyxy[0][2]),
            "ymax": int(box.xyxy[0][3])
        })

    return jsonify(detections)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
