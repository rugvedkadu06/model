import cv2
import numpy as np
import os
import base64
from flask import Flask, request, jsonify
from ultralytics import YOLO
from pymongo import MongoClient
from bson.objectid import ObjectId # Import ObjectId for validation
from dotenv import load_dotenv
import cloudinary
import cloudinary.uploader
from datetime import datetime

# ============================
# STEP 1: INITIALIZE APP & SERVICES
# ============================
load_dotenv()  # Load variables from .env file

app = Flask(__name__)

# --- Load YOLOv8 Model ---
try:
    model = YOLO('best.pt')
    classes = model.names
    print("✅ YOLOv8 Model loaded successfully.")
except Exception as e:
    print(f"🛑 Error loading model: {e}")
    # Consider what to do if the model fails to load (e.g., exit or use a dummy model)
    exit()

# --- Connect to MongoDB ---
try:
    client = MongoClient(os.getenv('MONGO_URI'))
    db = client.test # Use or create a database named 'test'
    detections_collection = db.detections # Use or create a collection named 'detections'
    # Test connection
    client.server_info()
    print("✅ MongoDB connected successfully.")
except Exception as e:
    print(f"🛑 Error connecting to MongoDB: {e}")
    exit()

# --- Configure Cloudinary ---
try:
    cloudinary.config(
        cloud_name=os.getenv('CLOUDINARY_CLOUD_NAME'),
        api_key=os.getenv('CLOUDINARY_API_KEY'),
        api_secret=os.getenv('CLOUDINARY_API_SECRET')
    )
    print("✅ Cloudinary configured successfully.")
except Exception as e:
    print(f"🛑 Error configuring Cloudinary: {e}")
    exit()


# ============================
# STEP 2: DEFINE CORE LOGIC
# ============================
def assign_severity_priority(cls_name, box, img_w, img_h):
    """Assigns severity and priority based on detection class, size, and position."""
    x1, y1, x2, y2 = box
    area = max(0, (x2 - x1) * (y2 - y1))
    ratio = area / (img_w * img_h + 1e-9)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    # Normalize center distance to range [0, 1]
    center_dist = abs(cx - img_w/2)/img_w + abs(cy - img_h/2)/img_h

    if cls_name == "Garbage":
        if ratio > 0.10: return "High", "Urgent"
        elif ratio > 0.05: return "Medium", "Normal"
        else: return "Low", "Low"
    elif cls_name == "Potholes and RoadCracks":
        # Potholes close to the center are often more critical
        if ratio > 0.07 or center_dist < 0.2: return "High", "Urgent"
        elif ratio > 0.03: return "Medium", "Normal"
        else: return "Low", "Low"
    return "Unknown", "Low"


# ============================
# STEP 3: CREATE THE API ENDPOINT
# ============================
@app.route('/analyze', methods=['POST'])
def analyze_image():
    # --- 1. Get Issue ID from Form Data (NEW) ---
    issue_id = request.form.get('issue_id')
    if not issue_id:
        return jsonify({'error': 'Missing required form field: issue_id'}), 400
    
    # Optional: Basic validation to ensure it looks like a MongoDB ObjectId
    if not ObjectId.is_valid(issue_id):
        return jsonify({'error': 'Invalid issue_id format.'}), 400


    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    img_bytes = file.read()
    
    # Decode image for processing
    np_img = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({'error': 'Could not decode image'}), 400

    h, w = img.shape[:2]

    # --- Run Inference ---
    try:
        # Increased confidence threshold for production/real-world use
        results = model.predict(source=img, save=False, conf=0.45) 
    except Exception as e:
        return jsonify({'error': f'Error during YOLO inference: {e}'}), 500
        
    # --- Process and Annotate Image ---
    r = results[0]
    annotated_img = r.orig_img.copy()
    detections_data = []

    for b in r.boxes:
        xyxy = b.xyxy[0].cpu().numpy().tolist()
        cls_id = int(b.cls[0])
        cls_name = classes[cls_id]
        conf = float(b.conf[0])
        
        severity, priority = assign_severity_priority(cls_name, xyxy, w, h)
        x1, y1, x2, y2 = map(int, xyxy)
        
        color = (0, 0, 255) # Red for High
        if severity == "Medium": color = (0, 165, 255) # Orange
        elif severity == "Low": color = (0, 255, 0) # Green
        
        # Draw bounding box and label
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
        # Use smaller font scale for better visualization
        label = f"{cls_name} | {severity}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        # Adjust label position to be outside the box at the top
        cv2.rectangle(annotated_img, (x1, y1 - lh - 8), (x1 + lw, y1), color, -1)
        cv2.putText(annotated_img, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        detections_data.append({
            "class": cls_name, "confidence": conf,
            "bbox": [x1, y1, x2, y2], "severity": severity, "priority": priority
        })

    # --- Upload to Cloudinary & Save to MongoDB ---
    try:
        # Encode annotated image to bytes for upload
        _, img_encoded = cv2.imencode('.jpg', annotated_img)
        img_bytes_for_upload = img_encoded.tobytes()

        # Upload to Cloudinary
        upload_result = cloudinary.uploader.upload(
            img_bytes_for_upload,
            folder="janvaani_detections"
        )
        annotated_image_url = upload_result.get('secure_url')

        # Prepare document for MongoDB
        db_document = {
            'issueId': ObjectId(issue_id), # Store as MongoDB ObjectId
            'annotatedImageUrl': annotated_image_url,
            'detections': detections_data,
            'createdAt': datetime.utcnow() # Using standard datetime
        }

        # Insert into database
        # Optional: Check if a detection already exists for this issueId and update/replace it
        result = detections_collection.insert_one(db_document)
        
        # Prepare response for client
        db_document['_id'] = str(result.inserted_id)
        db_document['issueId'] = issue_id # Return original string for client clarity
        
        print(f"✅ Successfully processed image. Saved as document ID: {db_document['_id']} for Issue ID: {issue_id}")
        return jsonify(db_document), 201

    except Exception as e:
        print(f"🛑 Error during upload or database save: {e}")
        return jsonify({'error': 'Failed to save analysis results.'}), 500


if __name__ == '__main__':
    # --- FIX FOR DEPLOYMENT ---
    port = int(os.environ.get('PORT', 5000))
    # Note: Setting debug=True for local testing is fine, but ensure it's False in production
    app.run(host='0.0.0.0', port=port, debug=False)
