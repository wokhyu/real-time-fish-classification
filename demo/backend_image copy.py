import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import json
import base64

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn
import io

# ==================== CẤU HÌNH & TẢI MODEL ====================

# ĐƯỜNG DẪN MODEL CHO IMAGE & WEBCAM
MODEL_PATH = r"D:\best_new.pt"
CNN_MODEL_PATH = r"D:\model2.keras"
LABEL_PATH = r"D:\fish_classifier_labels.json"

# Cấu hình detection
DETECTION_CONF = 0.4
CNN_CONF_THRESHOLD = 0.5
WEBCAM_CONF_THRESHOLD = 0.5 
MIN_BOX_AREA = 400
IOU_THRESHOLD = 0.5  # Ngưỡng để loại bỏ bounding box trùng lặp

# -------------------- KHỞI TẠO APP --------------------
app = FastAPI(title="FishVision AI - Image & Webcam Backend")

origins = [
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "*"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- TẢI MODELS --------------------
print("🔄 Loading YOLOv8 model for image detection...")
try:
    yolo_model = YOLO(MODEL_PATH)
    print("✅ YOLOv8 model loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load YOLOv8 model: {e}")
    yolo_model = None

print("🔄 Loading CNN model for classification...")
try:
    cnn_model = load_model(CNN_MODEL_PATH, compile=False)
    print("✅ CNN model loaded successfully.")
except Exception as e:
    from tensorflow.keras import models, layers
    from tensorflow.keras.applications import MobileNetV2
    
    print("⚠️ Rebuilding CNN architecture...")
    base_model = MobileNetV2(input_shape=(224,224,3), include_top=False, weights='imagenet')
    base_model.trainable = False
    
    cnn_model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(463, activation='softmax')
    ])
    
    try:
        cnn_model.load_weights(CNN_MODEL_PATH)
        print("✅ Weights loaded into rebuilt CNN.")
    except:
        print("⚠️ CNN initialized with random weights.")

# Tải Class Labels
print("🔄 Loading class labels...")
try:
    with open(LABEL_PATH, 'r', encoding='utf-8') as f:
        class_indices = json.load(f)
    label_map = {int(k): v for k, v in class_indices.items()}
    print(f"✅ Loaded {len(label_map)} class labels.")
except Exception as e:
    label_map = {i: f"Species_{i}" for i in range(463)}
    print(f"⚠️ Using default labels. Error: {e}")

# ==================== HELPER FUNCTIONS ====================

def calculate_iou(box1, box2):
    """Tính IoU (Intersection over Union) giữa 2 bounding boxes."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Tính diện tích giao nhau
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    # Tính diện tích từng box
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # Tính union
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0


def non_max_suppression_custom(detections, iou_threshold=0.5):
    """
    Loại bỏ các detection trùng lặp dựa trên IoU.
    detections: list of dict với keys: bbox, confidence, species
    """
    if len(detections) == 0:
        return []
    
    # Sort by confidence (descending)
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    
    keep = []
    while len(detections) > 0:
        # Lấy detection có confidence cao nhất
        best = detections.pop(0)
        keep.append(best)
        
        # Loại bỏ các detection có IoU cao với best
        detections = [
            det for det in detections 
            if calculate_iou(best['bbox'], det['bbox']) < iou_threshold
        ]
    
    return keep


def generate_wikipedia_link(species_name: str) -> str:
    """Tạo link Wikipedia tiếng Anh dựa trên tên loài cá."""
    if species_name == "No Fish Detected" or species_name == "N/A" or species_name.startswith("Unknown"):
        return None
    wiki_title = species_name.replace(" ", "_")
    return f"https://en.wikipedia.org/wiki/{wiki_title}"


def draw_detection_on_frame(frame, species_name, confidence, bbox, fish_index=None):
    """Vẽ bounding box và thông tin lên frame."""
    x1, y1, x2, y2 = bbox
    
    # Màu box dựa trên confidence
    if confidence >= 0.7:
        color = (0, 255, 0)  # Xanh lá - High confidence
        thickness = 10
    elif confidence >= 0.5:
        color = (0, 165, 255)  # Cam - Medium confidence
        thickness = 7
    else:
        color = (0, 0, 255)  # Đỏ - Low confidence
        thickness = 5
    
    # Vẽ bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    
    # Chuẩn bị text
    fish_label = f"Fish #{fish_index}" if fish_index else "Fish"
    conf_text = f"{confidence*100:.1f}%"
    species_text = species_name if len(species_name) < 25 else species_name[:22] + "..."
    
    # Tính kích thước background cho text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2
    
    # ✅ FIX: Tính text_height TRƯỚC KHI sử dụng
    (text_width, text_height), baseline = cv2.getTextSize(
        species_text, font, font_scale, font_thickness
    )
    
    # Vẽ Fish # label (nếu có)
    y_offset = y1  # Vị trí bắt đầu từ trên xuống
    
    if fish_index:
        (label_width, label_height), _ = cv2.getTextSize(
            fish_label, font, font_scale, font_thickness
        )
        # Background cho Fish #
        cv2.rectangle(frame, 
                      (x1, y_offset - label_height - 10), 
                      (x1 + label_width + 10, y_offset), 
                      (255, 0, 0), -1)
        # Text Fish #
        cv2.putText(frame, fish_label, 
                    (x1 + 5, y_offset - 5), 
                    font, font_scale, (255, 255, 255), font_thickness)
        
        y_offset -= (label_height + 10)  # Di chuyển lên trên cho species name
    
    # Vẽ background cho species name
    cv2.rectangle(frame, 
                  (x1, y_offset - text_height - 10), 
                  (x1 + text_width + 10, y_offset), 
                  color, -1)
    
    # Vẽ text species name
    cv2.putText(frame, species_text, 
                (x1 + 5, y_offset - 5), 
                font, font_scale, (255, 255, 255), font_thickness)
    
    # Vẽ background cho confidence (dưới box)
    (conf_width, conf_height), _ = cv2.getTextSize(
        conf_text, font, font_scale, font_thickness
    )
    cv2.rectangle(frame, 
                  (x1, y2), 
                  (x1 + conf_width + 10, y2 + conf_height + 10), 
                  color, -1)
    
    # Vẽ confidence score
    cv2.putText(frame, conf_text, 
                (x1 + 5, y2 + conf_height + 5), 
                font, font_scale, (255, 255, 255), font_thickness)
    
    return frame


def process_multiple_fish_image(image_bytes: bytes, return_annotated=True):
    """
    Phát hiện và phân loại NHIỀU con cá trong 1 ảnh.
    Trả về danh sách tất cả các con cá được phát hiện.
    """
    if yolo_model is None or cnn_model is None:
        return {
            "total_fish": 0,
            "detections": [],
            "annotated_image": None,
            "message": "Models not loaded"
        }

    # Decode image từ bytes
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        raise ValueError("Could not decode image.")

    # YOLO detection
    results = yolo_model(frame, conf=DETECTION_CONF, verbose=False)
    
    all_detections = []
    annotated_frame = frame.copy()
    
    if results and results[0].boxes:
        boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()

        # Xử lý từng bounding box
        for box, yolo_conf in zip(boxes_xyxy, confs):
            x1, y1, x2, y2 = map(int, box)
            
            # Filter out boxes that are too small
            if (x2 - x1) * (y2 - y1) < MIN_BOX_AREA:
                continue

            # Crop fish region with padding
            crop_pad = 10
            y1p = max(0, y1 - crop_pad)
            y2p = min(frame.shape[0], y2 + crop_pad)
            x1p = max(0, x1 - crop_pad)
            x2p = min(frame.shape[1], x2 + crop_pad)
            crop_img = frame[y1p:y2p, x1p:x2p]

            if crop_img.size > 0:
                try:
                    # Preprocess for CNN
                    crop_img_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                    crop_img_rgb = cv2.resize(crop_img_rgb, (224, 224))
                    crop_img_rgb = preprocess_input(crop_img_rgb)
                    crop_img_rgb = np.expand_dims(crop_img_rgb, axis=0)

                    # CNN prediction
                    preds = cnn_model.predict(crop_img_rgb, verbose=0)
                    class_id_cnn = np.argmax(preds, axis=1)[0]
                    pred_confidence = float(preds[0][class_id_cnn])
                    species_name_cnn = label_map.get(class_id_cnn, f"Unknown_{class_id_cnn}")

                    # Only accept predictions above threshold
                    if pred_confidence > CNN_CONF_THRESHOLD:
                        detection = {
                            "species": species_name_cnn,
                            "confidence": pred_confidence,
                            "scientificName": species_name_cnn.replace(" ", "_"),
                            "wikiLink": generate_wikipedia_link(species_name_cnn),
                            "bbox": [x1, y1, x2, y2]
                        }
                        all_detections.append(detection)
                        
                except Exception as e:
                    print(f"⚠️ Error during CNN prediction: {e}")
                    continue

    # Apply NMS to remove overlapping detections
    all_detections = non_max_suppression_custom(all_detections, IOU_THRESHOLD)
    
    # Vẽ tất cả detections lên frame
    if return_annotated and len(all_detections) > 0:
        for idx, detection in enumerate(all_detections, 1):
            annotated_frame = draw_detection_on_frame(
                annotated_frame, 
                detection['species'], 
                detection['confidence'], 
                detection['bbox'],
                fish_index=idx
            )

    # Convert frame to base64
    annotated_image_base64 = None
    if return_annotated and len(all_detections) > 0:
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        annotated_image_base64 = f"data:image/jpeg;base64,{img_base64}"

    # Format response
    if len(all_detections) > 0:
        # Thêm description cho mỗi detection
        for idx, det in enumerate(all_detections, 1):
            det['fish_id'] = idx
            det['description'] = (
                f"Con cá #{idx}: {det['species']} "
                f"({det['confidence']*100:.1f}% confidence)"
            )
        
        return {
            "total_fish": len(all_detections),
            "detections": all_detections,
            "annotated_image": annotated_image_base64,
            "message": f"Đã phát hiện {len(all_detections)} con cá trong ảnh"
        }
    else:
        return {
            "total_fish": 0,
            "detections": [],
            "annotated_image": None,
            "message": f"Không tìm thấy cá hoặc độ tin cậy < {CNN_CONF_THRESHOLD*100:.0f}%"
        }


def process_webcam_frame(frame):
    """
    Xử lý frame từ webcam real-time với ngưỡng confidence cao hơn.
    CHỈ trả về con cá có confidence cao nhất.
    """
    if yolo_model is None or cnn_model is None:
        return frame, None

    # YOLO detection
    results = yolo_model(frame, conf=DETECTION_CONF, verbose=False)
    
    detection_info = None
    
    if results and results[0].boxes:
        boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()

        for box, yolo_conf in zip(boxes_xyxy, confs):
            x1, y1, x2, y2 = map(int, box)
            
            if (x2 - x1) * (y2 - y1) < MIN_BOX_AREA:
                continue

            crop_pad = 10
            y1p = max(0, y1 - crop_pad)
            y2p = min(frame.shape[0], y2 + crop_pad)
            x1p = max(0, x1 - crop_pad)
            x2p = min(frame.shape[1], x2 + crop_pad)
            crop_img = frame[y1p:y2p, x1p:x2p]

            if crop_img.size > 0:
                try:
                    crop_img_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                    crop_img_rgb = cv2.resize(crop_img_rgb, (224, 224))
                    crop_img_rgb = preprocess_input(crop_img_rgb)
                    crop_img_rgb = np.expand_dims(crop_img_rgb, axis=0)

                    preds = cnn_model.predict(crop_img_rgb, verbose=0)
                    class_id_cnn = np.argmax(preds, axis=1)[0]
                    pred_confidence = float(preds[0][class_id_cnn])
                    species_name_cnn = label_map.get(class_id_cnn, f"Unknown_{class_id_cnn}")

                    # Chỉ hiển thị khi confidence >= 0.6
                    if pred_confidence >= WEBCAM_CONF_THRESHOLD:
                        detection_info = {
                            "species": species_name_cnn,
                            "confidence": pred_confidence,
                            "bbox": [x1, y1, x2, y2]
                        }
                        
                        # Vẽ detection lên frame
                        frame = draw_detection_on_frame(
                            frame, 
                            species_name_cnn, 
                            pred_confidence, 
                            [x1, y1, x2, y2]
                        )
                        break
                        
                except Exception as e:
                    print(f"⚠️ Webcam processing error: {e}")
                    continue
    
    return frame, detection_info


# ==================== ENDPOINTS ====================

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "message": "FishVision AI - Multi-Fish Detection Backend",
        "status": "OK",
        "models": {
            "yolo": MODEL_PATH,
            "cnn": CNN_MODEL_PATH
        },
        "features": [
            "✅ Multi-fish detection in single image",
            "✅ Non-Maximum Suppression (IoU-based)",
            "✅ Real-time webcam processing"
        ],
        "endpoints": [
            "POST /analyze/image - Analyze image with multiple fish",
            "POST /webcam/stream - Real-time webcam processing"
        ]
    }


@app.post("/analyze/image")
async def analyze_image_endpoint(file: UploadFile = File(...)):
    """
    API phân tích ảnh - phát hiện NHIỀU con cá.
    Trả về danh sách tất cả các con cá + ảnh annotated.
    """
    if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only JPEG, PNG, or WEBP images are supported."
        )

    try:
        image_bytes = await file.read()
        result = process_multiple_fish_image(image_bytes, return_annotated=True)
        
        return result
        
    except ValueError as e:
        print(f"❌ ValueError: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal Server Error: {str(e)}"
        )


@app.post("/webcam/stream")
async def webcam_stream_endpoint(file: UploadFile = File(...)):
    """
    API xử lý frame từ webcam real-time.
    Chỉ trả về detection khi confidence >= 0.6
    """
    if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type."
        )

    try:
        image_bytes = await file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise ValueError("Could not decode frame.")

        # Xử lý frame
        processed_frame, detection_info = process_webcam_frame(frame)

        # Convert frame về JPEG
        _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return {
            "frame": f"data:image/jpeg;base64,{img_base64}",
            "detection": detection_info
        }
        
    except Exception as e:
        print(f"❌ Webcam processing error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error: {str(e)}"
        )


if __name__ == "__main__":
    print("=" * 60)
    print("🐟 FishVision AI - Multi-Fish Detection Backend")
    print("=" * 60)
    print(f"📍 Running on: http://0.0.0.0:8000")
    print(f"📊 YOLO Model: {MODEL_PATH}")
    print(f"🧠 CNN Model: {CNN_MODEL_PATH}")
    print(f"🏷️  Labels: {len(label_map)} species")
    print(f"🎯 Image Detection Threshold: {CNN_CONF_THRESHOLD}")
    print(f"🎯 Webcam Detection Threshold: {WEBCAM_CONF_THRESHOLD}")
    print(f"🔄 IoU Threshold (NMS): {IOU_THRESHOLD}")
    print("=" * 60)
    print("✅ Ready to detect MULTIPLE fish in images!")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)

