"""
SmartPlant Enhanced - Rice Leaf Health Analysis System
Production Version 2.0 with Enhanced Accuracy

ENHANCEMENTS:
1. Temperature Scaling - Calibrated confidence scores
2. Model Ensemble - Combines multiple models for better accuracy
3. Aggressive TTA - 8 augmented predictions per image
4. Uses improved model from train_improved.py

Author: SmartPlant Team
"""

import os
import json
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

# Model paths - using best_model_fixed.pth as primary
MODEL_PRIMARY = "best_model_fixed.pth"
MODEL_FALLBACK = "best_model.pth"  # Just as a backup file check
OUTPUT_DIR = "outputs_enhanced"

MIN_IMAGES_REQUIRED = 3
CONFIDENCE_MEDIUM = 0.40

# Enhancement settings
ENABLE_TTA = True
TTA_AUGMENTATIONS = 8
ENABLE_ENSEMBLE = False     # Disabled to force using only the best trusted model
TEMPERATURE = 1.2           # Lowered slightly for fixed model

# Health score weights
WEIGHT_CNN = 0.6
WEIGHT_LESION = 0.4


# ============================================================================
# MODEL LOADING
# ============================================================================
def load_model(model_path: str) -> nn.Module:
    """Load MobileNetV2 model with same architecture as training."""
    model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.last_channel, len(CLASSES))
    )
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


def load_ensemble_models() -> List[nn.Module]:
    """Load multiple models for ensemble."""
    models = []
    
    # Primary model (improved or fixed)
    if os.path.exists(MODEL_PRIMARY):
        print(f"  Loading primary: {MODEL_PRIMARY}")
        models.append(load_model(MODEL_PRIMARY))
    elif os.path.exists(MODEL_FALLBACK):
        print(f"  Loading fallback: {MODEL_FALLBACK}")
        models.append(load_model(MODEL_FALLBACK))
    
    # Add secondary model for ensemble if available
    if ENABLE_ENSEMBLE and os.path.exists(MODEL_FALLBACK) and os.path.exists(MODEL_PRIMARY):
        print(f"  Loading ensemble: {MODEL_FALLBACK}")
        models.append(load_model(MODEL_FALLBACK))
    
    return models


# Transform (NO normalization - matching training)
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor()
])


# ============================================================================
# ENHANCED TTA - 8 AUGMENTATIONS
# ============================================================================
def apply_aggressive_tta(img: np.ndarray) -> List[np.ndarray]:
    """
    Generate 8 augmented versions for aggressive TTA.
    1. Original
    2. Horizontal flip
    3. Vertical flip
    4. Rotate 90 CW
    5. Rotate 90 CCW
    6. Rotate 180
    7. Flip + Rotate 90
    8. Brightness adjusted
    """
    augmented = [img]
    
    # Flips
    augmented.append(cv2.flip(img, 1))  # Horizontal
    augmented.append(cv2.flip(img, 0))  # Vertical
    
    # Rotations
    augmented.append(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))
    augmented.append(cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE))
    augmented.append(cv2.rotate(img, cv2.ROTATE_180))
    
    # Combined
    flipped = cv2.flip(img, 1)
    augmented.append(cv2.rotate(flipped, cv2.ROTATE_90_CLOCKWISE))
    
    # Brightness
    bright = cv2.convertScaleAbs(img, alpha=1.1, beta=10)
    augmented.append(bright)
    
    return augmented[:TTA_AUGMENTATIONS]


# ============================================================================
# TEMPERATURE SCALING
# ============================================================================
def apply_temperature_scaling(logits: np.ndarray, temperature: float = TEMPERATURE) -> np.ndarray:
    """
    Apply temperature scaling to calibrate confidence.
    Higher temperature = more conservative (lower confidence)
    """
    scaled = logits / temperature
    exp_scaled = np.exp(scaled - np.max(scaled))
    return exp_scaled / exp_scaled.sum()


# ============================================================================
# ENSEMBLE CLASSIFICATION
# ============================================================================
def classify_with_ensemble(models: List[nn.Module], img: np.ndarray) -> Dict:
    """
    Classify using ensemble of models + aggressive TTA + temperature scaling.
    """
    if ENABLE_TTA:
        augmented_images = apply_aggressive_tta(img)
    else:
        augmented_images = [img]
    
    all_logits = []
    
    for model in models:
        for aug_img in augmented_images:
            x = transform(aug_img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits = model(x).cpu().numpy()[0]
            all_logits.append(logits)
    
    # Average logits across all predictions
    avg_logits = np.mean(all_logits, axis=0)
    
    # Apply temperature scaling for calibrated confidence
    probs = apply_temperature_scaling(avg_logits)
    
    predicted_idx = int(np.argmax(probs))
    
    return {
        "probabilities": {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))},
        "predicted_class": CLASSES[predicted_idx],
        "confidence": float(probs[predicted_idx]),
        "num_predictions": len(all_logits)
    }


# ============================================================================
# IMAGE PROCESSING
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


def create_overlay(img: np.ndarray, vein_mask: np.ndarray, lesion_mask: np.ndarray) -> np.ndarray:
    overlay = img.copy()
    vein_overlay = overlay.copy()
    vein_overlay[vein_mask > 0] = [0, 255, 0]
    overlay = cv2.addWeighted(overlay, 0.7, vein_overlay, 0.3, 0)
    overlay[lesion_mask > 0] = [0, 0, 255]
    return overlay


# ============================================================================
# HEALTH SCORE
# ============================================================================
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


# ============================================================================
# MAIN ANALYSIS
# ============================================================================
def analyze_plant(image_folder: str, output_dir: str = OUTPUT_DIR) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    
    valid_ext = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(valid_ext)])
    
    if len(image_files) < MIN_IMAGES_REQUIRED:
        return {"status": "error", "message": f"Minimum {MIN_IMAGES_REQUIRED} images required"}
    
    print("=" * 60)
    print("SmartPlant Enhanced v2.0 - Rice Leaf Analysis")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"TTA: {TTA_AUGMENTATIONS} augmentations")
    print(f"Temperature scaling: {TEMPERATURE}")
    print(f"Input: {image_folder} ({len(image_files)} images)")
    print("-" * 60)
    
    # Load ensemble models
    print("Loading models...")
    models = load_ensemble_models()
    print(f"Ensemble size: {len(models)} models")
    print("-" * 60)
    
    leaf_results = []
    overlay_paths = []
    total_lesion = 0
    
    for i, filename in enumerate(image_files):
        img = cv2.imread(os.path.join(image_folder, filename))
        if img is None:
            continue
        
        leaf_mask = segment_leaf(img)
        vein_mask = extract_veins(img, leaf_mask)
        lesion_mask, lesion_metrics = detect_lesions(img, leaf_mask)
        vein_metrics = compute_vein_morphometry(vein_mask, leaf_mask)
        
        # Enhanced classification
        cnn_result = classify_with_ensemble(models, img)
        
        # Save overlay
        overlay = create_overlay(img, vein_mask, lesion_mask)
        overlay_path = os.path.join(output_dir, f"leaf_{i+1}_overlay.png")
        cv2.imwrite(overlay_path, overlay)
        overlay_paths.append(overlay_path)
        
        leaf_results.append({
            "leaf_index": i + 1,
            "filename": filename,
            "classification": cnn_result,
            "lesion_metrics": lesion_metrics,
            "vein_morphometry": vein_metrics
        })
        
        total_lesion += lesion_metrics["lesion_area_percent"]
        
        conf = cnn_result["confidence"] * 100
        preds = cnn_result["num_predictions"]
        print(f"  Leaf {i+1}: {cnn_result['predicted_class']} ({conf:.1f}%, {preds} preds) - {filename}")
    
    if not leaf_results:
        return {"status": "error", "message": "No valid images"}
    
    # Aggregate
    num_leaves = len(leaf_results)
    avg_lesion = total_lesion / num_leaves
    
    avg_probs = {}
    for cls in CLASSES:
        avg_probs[cls] = np.mean([r["classification"]["probabilities"][cls] for r in leaf_results])
    
    final_class = max(avg_probs, key=avg_probs.get)
    final_conf = avg_probs[final_class]
    
    plant_classification = {
        "predicted_class": final_class,
        "confidence": round(final_conf, 3),
        "class_probabilities": {k: round(v, 4) for k, v in avg_probs.items()}
    }
    
    # Validation: Check if it's likely a leaf
    # If using temperature scaling, confidence is calibrated.
    # A very low max confidence means the model is unsure (likely OOD/not a leaf)
    MIN_CONFIDENCE_THRESHOLD = 0.35
    
    if final_conf < MIN_CONFIDENCE_THRESHOLD:
        print(f"[WARNING] Low confidence ({final_conf:.2f}). Possibly not a rice leaf.")
        health_score = 0
        condition = "Unknown Object"
        plant_classification["predicted_class"] = "Unknown"
    else:
        health_score, condition = calculate_health_score(plant_classification, avg_lesion)
    
    result = {
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        "system_version": "2.0-enhanced",
        "enhancements": {
            "tta_augmentations": TTA_AUGMENTATIONS,
            "ensemble_models": len(models),
            "temperature_scaling": TEMPERATURE
        },
        "input": {"folder": image_folder, "num_leaves_analyzed": num_leaves},
        "leaf_results": leaf_results,
        "overlay_images": overlay_paths,
        "plant_summary": {
            "classification": plant_classification,
            "health_score": health_score,
            "condition": condition,
            "avg_lesion_area_percent": round(avg_lesion, 2),
            "total_lesion_count": sum(r["lesion_metrics"]["lesion_count"] for r in leaf_results)
        }
    }
    
    report_path = os.path.join(output_dir, "analysis_report.json")
    with open(report_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print("-" * 60)
    print(f"Report: {report_path}")
    
    return result


def print_summary(result: Dict):
    if result.get("status") != "success":
        print(f"\n[ERROR] {result.get('message')}")
        return
    
    summary = result["plant_summary"]
    cls = summary["classification"]
    enh = result.get("enhancements", {})
    
    print("\n" + "=" * 60)
    print("PLANT HEALTH SUMMARY (Enhanced)")
    print("=" * 60)
    print(f"  Leaves Analyzed    : {result['input']['num_leaves_analyzed']}")
    print(f"  Predicted Class    : {cls['predicted_class']}")
    print(f"  Confidence         : {cls['confidence']*100:.1f}%")
    print(f"  Health Score       : {summary['health_score']}/100")
    print(f"  Condition          : {summary['condition']}")
    print(f"  Avg Lesion Area    : {summary['avg_lesion_area_percent']:.1f}%")
    print("-" * 60)
    print(f"  TTA Augmentations  : {enh.get('tta_augmentations', 'N/A')}")
    print(f"  Ensemble Models    : {enh.get('ensemble_models', 1)}")
    print(f"  Temperature        : {enh.get('temperature_scaling', 1.0)}")
    print("-" * 60)
    print("Class Probabilities:")
    for c, p in cls['class_probabilities'].items():
        bar = "█" * int(p * 20)
        print(f"    {c:12s}: {p*100:5.1f}% {bar}")
    print("=" * 60)
    
    if summary["condition"] == "Healthy":
        print("✓ Plant appears healthy")
    elif summary["condition"] == "Diseased":
        print(f"⚠ Possible disease: {cls['predicted_class']}")
    else:
        print("? Analysis inconclusive")
    print("=" * 60)


if __name__ == "__main__":
    result = analyze_plant("sample_leaves")
    print_summary(result)
