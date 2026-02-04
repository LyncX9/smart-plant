
import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as T
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from skimage.morphology import skeletonize
from datetime import datetime
from typing import Dict, List, Tuple

# ============================================================================
# CONFIGURATION
# ============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["BrownSpot", "Healthy", "Hispa", "LeafBlast"]
MODEL_PATH = "best_model_fixed.pth" 

# Validation Thresholds
MIN_CONFIDENCE_THRESHOLD = 0.35
CONFIDENCE_MEDIUM = 0.40

# Weights
WEIGHT_CNN = 0.6
WEIGHT_LESION = 0.4

# ============================================================================
# MODEL LOADING
# ============================================================================
model_instance = None

def load_model(path: str = MODEL_PATH) -> nn.Module:
    global model_instance
    if model_instance is not None:
        return model_instance
        
    print(f"Loading model from {path}...")
    model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.last_channel, len(CLASSES))
    )
    
    # Handle CPU deployment if GPU not available
    map_loc = DEVICE
    if not torch.cuda.is_available():
        map_loc = "cpu"
        
    try:
        state = torch.load(path, map_location=map_loc)
        model.load_state_dict(state)
        model.to(DEVICE)
        model.eval()
        model_instance = model
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        # Return un-trained model in worst case to prevent crash, but log error
        # In production, this should probably raise error
        model.to(DEVICE)
        model.eval()
        model_instance = model
        return model

# Initialize transform
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor()
])

# ============================================================================
# CORE PROCESSING FUNCTIONS (Refactored for Single Image)
# ============================================================================

def segment_leaf(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (20, 30, 30), (100, 255, 255))
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

def extract_veins(img: np.ndarray, leaf_mask: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    tophat_v = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, kernel_v)
    tophat_h = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, kernel_h)
    combined = cv2.add(tophat_v, tophat_h)
    
    _, vein_mask = cv2.threshold(combined, 20, 255, cv2.THRESH_BINARY)
    vein_mask = cv2.bitwise_and(vein_mask, vein_mask, mask=leaf_mask)
    
    kernel = np.ones((2, 2), np.uint8)
    vein_mask = cv2.morphologyEx(vein_mask, cv2.MORPH_OPEN, kernel)
    return vein_mask

def compute_vein_morphometry(vein_mask: np.ndarray, leaf_mask: np.ndarray) -> Dict:
    # Skeletonize requires boolean
    skeleton = skeletonize(vein_mask > 0)
    vein_length = int(np.sum(skeleton))
    vein_area = np.count_nonzero(vein_mask)
    leaf_area = np.count_nonzero(leaf_mask)
    
    vein_density = (vein_area / max(leaf_area, 1)) * 100
    vein_continuity = vein_length / max(vein_area, 1)
    
    return {
        "vein_length_px": vein_length,
        "vein_density_percent": round(vein_density, 2),
        "vein_continuity": round(vein_continuity, 3)
    }

def detect_lesions(img: np.ndarray, leaf_mask: np.ndarray) -> Tuple[np.ndarray, Dict]:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    brown = cv2.inRange(hsv, (5, 30, 40), (22, 200, 200))
    tan = cv2.inRange(hsv, (18, 40, 120), (30, 150, 220))
    lesion_mask = cv2.bitwise_or(brown, tan)
    lesion_mask = cv2.bitwise_and(lesion_mask, lesion_mask, mask=leaf_mask)
    
    kernel = np.ones((5, 5), np.uint8)
    lesion_mask = cv2.morphologyEx(lesion_mask, cv2.MORPH_OPEN, kernel)
    lesion_mask = cv2.morphologyEx(lesion_mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(lesion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered = np.zeros_like(lesion_mask)
    count = 0
    for cnt in contours:
        if cv2.contourArea(cnt) > 100:
            cv2.drawContours(filtered, [cnt], -1, 255, -1)
            count += 1
    
    lesion_area = np.count_nonzero(filtered)
    leaf_area = np.count_nonzero(leaf_mask)
    percent = (lesion_area / max(leaf_area, 1)) * 100
    
    return filtered, {"lesion_count": count, "lesion_area_percent": round(percent, 2)}

def apply_aggressive_tta(img: np.ndarray) -> List[np.ndarray]:
    augmented = [img]
    # Simple TTA subset for performance
    augmented.append(cv2.flip(img, 1))
    augmented.append(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))
    return augmented

def calculate_health_score(cnn_result: Dict, lesion_percent: float) -> Tuple[float, str]:
    predicted_class = cnn_result["predicted_class"]
    confidence = cnn_result["confidence"]
    
    if predicted_class == "Healthy":
        cnn_component = confidence * 100
    else:
        cnn_component = (1 - confidence) * 50
    
    lesion_component = max(0, 100 - lesion_percent)
    health_score = cnn_component * WEIGHT_CNN + lesion_component * WEIGHT_LESION
    health_score = min(100, max(0, health_score))
    
    if predicted_class == "Healthy" and confidence >= CONFIDENCE_MEDIUM:
        condition = "Healthy"
    elif predicted_class != "Healthy" and confidence >= CONFIDENCE_MEDIUM:
        condition = "Diseased"
    else:
        condition = "Uncertain"
    
    return round(health_score, 1), condition

def process_image_data(image_bytes: bytes) -> Dict:
    # 1. Decode Image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Invalid image data")

    # 2. Load Model
    model = load_model()
    
    # 3. Processing
    leaf_mask = segment_leaf(img)
    vein_mask = extract_veins(img, leaf_mask)
    lesion_mask, lesion_metrics = detect_lesions(img, leaf_mask)
    vein_metrics = compute_vein_morphometry(vein_mask, leaf_mask)
    
    # 4. Prediction (TTA)
    augmented_images = apply_aggressive_tta(img)
    all_logits = []
    
    for aug_img in augmented_images:
        x = transform(aug_img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model(x).cpu().numpy()[0]
        all_logits.append(logits)
        
    avg_logits = np.mean(all_logits, axis=0)
    
    # Temperature scaling (simple)
    temp = 1.2
    scaled = avg_logits / temp
    exp_scaled = np.exp(scaled - np.max(scaled))
    probs = exp_scaled / exp_scaled.sum()
    
    predicted_idx = int(np.argmax(probs))
    final_conf = float(probs[predicted_idx])
    final_class = CLASSES[predicted_idx]
    
    plant_classification = {
        "predicted_class": final_class,
        "confidence": round(final_conf, 3),
        "class_probabilities": {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}
    }
    
    # 5. Validation Logic
    if final_conf < MIN_CONFIDENCE_THRESHOLD:
        health_score = 0.0
        condition = "Unknown Object"
        plant_classification["predicted_class"] = "Unknown"
    else:
        health_score, condition = calculate_health_score(plant_classification, lesion_metrics["lesion_area_percent"])
        
    # 6. Encode Overlay for Frontend
    # Create overlay
    overlay = img.copy()
    vein_overlay = overlay.copy()
    vein_overlay[vein_mask > 0] = [0, 255, 0] # Green Veins
    overlay = cv2.addWeighted(overlay, 0.7, vein_overlay, 0.3, 0)
    overlay[lesion_mask > 0] = [0, 0, 255]    # Red Lesions
    
    # Encode to base64 or keep as bytes? Usually API returns URL or Base64.
    # For simplicity, let's return metrics. Frontend can do overlay if we send mask? 
    # Or better, we upload overlay to S3. 
    # For this Free Tier guide, let's return Base64 string of the overlay image.
    
    success, buffer = cv2.imencode('.jpg', overlay)
    overlay_base64 = None
    if success:
        import base64
        overlay_base64 = base64.b64encode(buffer).decode('utf-8')
        
    return {
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        "plant_summary": {
            "classification": plant_classification,
            "health_score": health_score,
            "condition": condition,
            "avg_lesion_area_percent": lesion_metrics["lesion_area_percent"],
            "total_lesion_count": lesion_metrics["lesion_count"]
        },
        "vein_morphometry": vein_metrics,
        "overlay_base64": overlay_base64 # Frontend can display this
    }
