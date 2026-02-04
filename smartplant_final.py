"""
SmartPlant - Rice Leaf Disease Detection & Morphometry Analysis
Final Production Script

Model: best_model_fixed.pth
Features:
- Disease classification (BrownSpot, Healthy, Hispa, LeafBlast)
- Vein detection using morphological top-hat filtering
- Lesion detection with refined HSV ranges
- Vein morphometry analysis (length, density, continuity)
- Visual overlay generation
- Health score calculation
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

# ============== CONFIGURATION ==============
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["BrownSpot", "Healthy", "Hispa", "LeafBlast"]
MODEL_PATH = "best_model_fixed.pth"
OUTPUT_DIR = "outputs_final"


# ============== MODEL LOADING ==============
def load_model(model_path):
    """Load the trained MobileNetV2 model"""
    model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    
    # Architecture must match train_fix_overfitting.py
    model.classifier[1] = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.last_channel, len(CLASSES))
    )

    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


# Transform WITHOUT normalization (matches original training)
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor()
])


# ============== IMAGE PROCESSING ==============
def rice_leaf_mask(img):
    """Extract rice leaf region using HSV color segmentation"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Green range for rice leaves
    mask = cv2.inRange(hsv, (25, 40, 40), (95, 255, 255))
    
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def extract_rice_vein(img):
    """
    Extract vein pattern using morphological top-hat filtering.
    Optimized for parallel longitudinal veins in rice leaves.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    leaf_mask = rice_leaf_mask(img)

    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Top-hat for vertical veins
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    tophat_v = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, kernel_v)

    # Top-hat for horizontal veins
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    tophat_h = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, kernel_h)

    # Combine both directions
    combined = cv2.add(tophat_v, tophat_h)

    # Threshold to binary
    _, vein = cv2.threshold(combined, 20, 255, cv2.THRESH_BINARY)
    
    # Apply leaf mask
    vein = cv2.bitwise_and(vein, vein, mask=leaf_mask)

    # Clean up noise
    kernel_clean = np.ones((2, 2), np.uint8)
    vein = cv2.morphologyEx(vein, cv2.MORPH_OPEN, kernel_clean)
    vein = cv2.morphologyEx(vein, cv2.MORPH_CLOSE, kernel_clean)
    
    return vein


def lesion_mask(img, leaf_mask):
    """
    Detect disease lesions (brown spots, tan areas).
    Uses refined HSV ranges to avoid false positives on healthy tissue.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Brown disease spots (BrownSpot, LeafBlast)
    brown_spots = cv2.inRange(hsv, (5, 30, 40), (22, 200, 200))
    
    # Tan/yellowish disease areas
    tan_areas = cv2.inRange(hsv, (18, 40, 120), (30, 150, 220))

    # Combine lesion types
    lesion = cv2.bitwise_or(brown_spots, tan_areas)
    
    # Apply leaf mask
    lesion = cv2.bitwise_and(lesion, lesion, mask=leaf_mask)

    # Morphological cleanup
    kernel = np.ones((5, 5), np.uint8)
    lesion = cv2.morphologyEx(lesion, cv2.MORPH_OPEN, kernel)
    lesion = cv2.morphologyEx(lesion, cv2.MORPH_CLOSE, kernel)

    # Filter small regions (noise)
    contours, _ = cv2.findContours(lesion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered = np.zeros_like(lesion)
    for cnt in contours:
        if cv2.contourArea(cnt) > 100:  # Minimum area threshold
            cv2.drawContours(filtered, [cnt], -1, 255, -1)

    return filtered


# ============== MORPHOMETRY ANALYSIS ==============
def calculate_morphometry(vein_mask, leaf_mask):
    """
    Calculate vein morphometry metrics:
    - vein_length: Total length of vein skeleton (pixels)
    - vein_density_percent: Vein coverage percentage of leaf area
    - vein_continuity: Ratio indicating vein structure coherence
    """
    # Skeletonize vein pattern
    skel = skeletonize(vein_mask > 0)
    
    vein_length = int(np.sum(skel))
    vein_area = np.count_nonzero(vein_mask)
    leaf_area = np.count_nonzero(leaf_mask)

    continuity = vein_length / (vein_area + 1e-6)
    vein_density = (vein_area / (leaf_area + 1e-6)) * 100

    return {
        "vein_length": vein_length,
        "vein_density_percent": round(float(vein_density), 2),
        "vein_continuity": round(float(continuity), 3)
    }


def calculate_lesion_metrics(lesion_mask, leaf_mask):
    """Calculate lesion-related metrics"""
    lesion_area = np.count_nonzero(lesion_mask)
    leaf_area = np.count_nonzero(leaf_mask)
    
    lesion_percent = (lesion_area / (leaf_area + 1e-6)) * 100
    
    # Count number of lesion spots
    contours, _ = cv2.findContours(lesion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return {
        "lesion_count": len(contours),
        "lesion_area_percent": round(float(lesion_percent), 2)
    }


# ============== VISUALIZATION ==============
def create_overlay(img, vein, lesion):
    """
    Create visual overlay:
    - Green: Vein pattern
    - Red: Disease lesions
    """
    overlay = img.copy()
    overlay[vein > 0] = [0, 255, 0]    # Green for veins
    overlay[lesion > 0] = [0, 0, 255]  # Red for lesions
    return overlay


def save_overlay(overlay, out_path):
    """Save overlay image to file"""
    cv2.imwrite(out_path, overlay)


# ============== MAIN ANALYSIS ==============
def analyze_single_leaf(model, img):
    """Analyze a single leaf image"""
    # Classification
    x = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1).cpu().numpy()[0]
    
    prediction = {CLASSES[j]: float(probs[j]) for j in range(len(CLASSES))}
    predicted_class = CLASSES[np.argmax(probs)]
    confidence = float(np.max(probs))
    
    # Morphometry
    leaf_mask = rice_leaf_mask(img)
    vein = extract_rice_vein(img)
    lesion = lesion_mask(img, leaf_mask)
    
    vein_metrics = calculate_morphometry(vein, leaf_mask)
    lesion_metrics = calculate_lesion_metrics(lesion, leaf_mask)
    
    # Overlay
    overlay = create_overlay(img, vein, lesion)
    
    return {
        "prediction": prediction,
        "predicted_class": predicted_class,
        "confidence": round(confidence, 3),
        "vein_metrics": vein_metrics,
        "lesion_metrics": lesion_metrics,
        "overlay": overlay
    }


def analyze_plant(image_folder, out_dir=OUTPUT_DIR):
    """
    Analyze all leaf images in a folder.
    Returns comprehensive plant health analysis.
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # Load model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    
    print(f"Loading model: {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    
    print(f"Analyzing images in: {image_folder}")
    print("-" * 50)
    
    results = []
    overlays = []
    
    image_files = sorted([f for f in os.listdir(image_folder) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    for i, name in enumerate(image_files):
        img_path = os.path.join(image_folder, name)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"  [SKIP] Cannot read: {name}")
            continue
        
        # Analyze
        result = analyze_single_leaf(model, img)
        
        # Save overlay
        overlay_path = os.path.join(out_dir, f"overlay_leaf_{i+1}.png")
        save_overlay(result["overlay"], overlay_path)
        overlays.append(overlay_path)
        
        # Store result (without overlay image data)
        leaf_result = {
            "filename": name,
            "prediction": result["prediction"],
            "predicted_class": result["predicted_class"],
            "confidence": result["confidence"],
            "vein_metrics": result["vein_metrics"],
            "lesion_metrics": result["lesion_metrics"]
        }
        results.append(leaf_result)
        
        print(f"  Leaf {i+1}: {result['predicted_class']} ({result['confidence']*100:.1f}%) - {name}")
    
    # Aggregate results
    if not results:
        return {"error": "No valid images found"}
    
    # Calculate average probabilities
    avg_probs = np.mean([[r["prediction"][c] for c in CLASSES] for r in results], axis=0)
    final_class_idx = int(np.argmax(avg_probs))
    final_class = CLASSES[final_class_idx]
    final_confidence = float(avg_probs[final_class_idx])
    
    # Health score (probability of Healthy class)
    healthy_idx = CLASSES.index("Healthy")
    health_score = float(avg_probs[healthy_idx]) * 100
    
    # Average morphometry
    avg_vein_density = np.mean([r["vein_metrics"]["vein_density_percent"] for r in results])
    avg_lesion_area = np.mean([r["lesion_metrics"]["lesion_area_percent"] for r in results])
    total_lesions = sum([r["lesion_metrics"]["lesion_count"] for r in results])
    
    final_result = {
        "timestamp": datetime.now().isoformat(),
        "num_leaves": len(results),
        "leaf_results": results,
        "overlay_images": overlays,
        "summary": {
            "predicted_class": final_class,
            "confidence": round(final_confidence, 3),
            "health_score": round(health_score, 2),
            "plant_condition": "Healthy" if final_class == "Healthy" else "Diseased",
            "avg_vein_density_percent": round(avg_vein_density, 2),
            "avg_lesion_area_percent": round(avg_lesion_area, 2),
            "total_lesion_count": total_lesions
        }
    }
    
    # Save JSON report
    report_path = os.path.join(out_dir, "analysis_report.json")
    with open(report_path, 'w') as f:
        json.dump(final_result, f, indent=2)
    print(f"\nReport saved: {report_path}")
    
    return final_result


# ============== ENTRY POINT ==============
if __name__ == "__main__":
    result = analyze_plant("sample_leaves")
    
    print("\n" + "=" * 50)
    print("PLANT HEALTH SUMMARY")
    print("=" * 50)
    print(f"Leaves analyzed: {result['num_leaves']}")
    print(f"Predicted class: {result['summary']['predicted_class']}")
    print(f"Confidence: {result['summary']['confidence']*100:.1f}%")
    print(f"Health score: {result['summary']['health_score']:.1f}%")
    print(f"Condition: {result['summary']['plant_condition']}")
    print(f"Avg vein density: {result['summary']['avg_vein_density_percent']:.1f}%")
    print(f"Avg lesion area: {result['summary']['avg_lesion_area_percent']:.1f}%")
    print(f"Total lesions: {result['summary']['total_lesion_count']}")
    print("=" * 50)
