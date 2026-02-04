"""
================================================================================
SmartPlant - Rice Leaf Disease Detection & Health Analysis System
Production-Ready Version 1.0
================================================================================

System Overview:
---------------
SmartPlant analyzes multiple images of rice leaves from a single plant to 
estimate plant health condition using computer vision and deep learning.
The system produces both numerical results and visual explanations.

Architecture:
- Classification: MobileNetV2-based CNN for disease detection
- Vein Analysis: Orientation-based edge detection for rice leaf venation
- Lesion Detection: HSV color-space segmentation
- Health Scoring: Heuristic combining CNN confidence and lesion extent

Classes: BrownSpot, Healthy, Hispa, LeafBlast

Author: SmartPlant Team
License: MIT
================================================================================
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
from typing import Dict, List, Optional, Tuple

# ============================================================================
# CONFIGURATION
# ============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["BrownSpot", "Healthy", "Hispa", "LeafBlast"]
MODEL_PATH = "best_model_fixed.pth"
OUTPUT_DIR = "outputs_production"

# Minimum number of leaf images required for reliable analysis
MIN_IMAGES_REQUIRED = 3

# Test-Time Augmentation (TTA) settings
ENABLE_TTA = True           # Enable TTA for confidence boost
TTA_AUGMENTATIONS = 5       # Number of augmented predictions per image

# Confidence thresholds for decision transparency
CONFIDENCE_HIGH = 0.60      # High confidence threshold
CONFIDENCE_MEDIUM = 0.40    # Medium confidence threshold
CONFIDENCE_LOW = 0.25       # Low confidence (uncertain)

# Health score weights (must sum to 1.0)
WEIGHT_CNN_CONFIDENCE = 0.6     # Weight for CNN classification confidence
WEIGHT_LESION_ABSENCE = 0.4     # Weight for lesion-based scoring


# ============================================================================
# MODEL LOADING
# ============================================================================
def load_model(model_path: str) -> nn.Module:
    """
    Load the trained MobileNetV2-based CNN model.
    
    Architecture matches train_fix_overfitting.py:
    - Base: MobileNetV2 pretrained on ImageNet
    - Classifier: Dropout(0.5) -> Linear(1280, 4)
    
    Args:
        model_path: Path to the saved model weights (.pth file)
        
    Returns:
        Loaded and evaluation-ready PyTorch model
    """
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


# Image transform (matches training - NO normalization)
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor()
])


# ============================================================================
# IMAGE PROCESSING - LEAF SEGMENTATION
# ============================================================================
def segment_leaf(img: np.ndarray) -> np.ndarray:
    """
    Segment rice leaf region from background using HSV color thresholding.
    
    Rice leaves typically appear in green to yellow-green hue range.
    Background is assumed to be non-green (white, gray, or other colors).
    
    Args:
        img: BGR image array
        
    Returns:
        Binary mask where 255 = leaf region, 0 = background
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Green to yellow-green range for rice leaves
    mask = cv2.inRange(hsv, (20, 30, 30), (100, 255, 255))
    
    # Morphological operations to clean up mask
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return mask


# ============================================================================
# VEIN EXTRACTION - RICE LEAF SPECIFIC
# ============================================================================
def extract_veins(img: np.ndarray, leaf_mask: np.ndarray) -> np.ndarray:
    """
    Extract vein pattern from rice leaf using orientation-based morphology.
    
    Rice leaves exhibit parallel longitudinal venation, which is incompatible
    with traditional vessel detection methods (e.g., Frangi filter).
    
    Method:
    1. Convert to grayscale and enhance contrast (CLAHE)
    2. Apply morphological top-hat with vertical/horizontal kernels
    3. Combine orientations and threshold
    4. Apply leaf mask to remove background artifacts
    
    Args:
        img: BGR image array
        leaf_mask: Binary mask of leaf region
        
    Returns:
        Binary mask of detected vein structures
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Top-hat for vertical structures (longitudinal veins)
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    tophat_v = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, kernel_v)
    
    # Top-hat for horizontal structures (lateral veins)
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    tophat_h = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, kernel_h)
    
    # Combine orientations
    combined = cv2.add(tophat_v, tophat_h)
    
    # Threshold to binary
    _, vein_mask = cv2.threshold(combined, 20, 255, cv2.THRESH_BINARY)
    
    # Apply leaf mask
    vein_mask = cv2.bitwise_and(vein_mask, vein_mask, mask=leaf_mask)
    
    # Clean up noise
    kernel_clean = np.ones((2, 2), np.uint8)
    vein_mask = cv2.morphologyEx(vein_mask, cv2.MORPH_OPEN, kernel_clean)
    vein_mask = cv2.morphologyEx(vein_mask, cv2.MORPH_CLOSE, kernel_clean)
    
    return vein_mask


def compute_vein_morphometry(vein_mask: np.ndarray, leaf_mask: np.ndarray) -> Dict:
    """
    Compute vein morphometry features for descriptive analysis.
    
    Note: These features are for visualization and description only.
    They do NOT influence disease classification or health scoring.
    
    Features:
    - vein_length: Total skeleton length in pixels
    - vein_density_percent: Vein area as percentage of leaf area
    - vein_continuity: Ratio of skeleton to vein area (structure coherence)
    
    Args:
        vein_mask: Binary mask of veins
        leaf_mask: Binary mask of leaf region
        
    Returns:
        Dictionary of morphometry metrics
    """
    # Skeletonization for visualization and length calculation
    skeleton = skeletonize(vein_mask > 0)
    
    vein_length = int(np.sum(skeleton))
    vein_area = np.count_nonzero(vein_mask)
    leaf_area = np.count_nonzero(leaf_mask)
    
    # Avoid division by zero
    vein_density = (vein_area / max(leaf_area, 1)) * 100
    vein_continuity = vein_length / max(vein_area, 1)
    
    return {
        "vein_length_px": vein_length,
        "vein_density_percent": round(vein_density, 2),
        "vein_continuity": round(vein_continuity, 3)
    }


# ============================================================================
# LESION DETECTION
# ============================================================================
def detect_lesions(img: np.ndarray, leaf_mask: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    Detect disease lesions using HSV color-based segmentation.
    
    Lesions typically appear as:
    - Brown spots (BrownSpot, LeafBlast)
    - Tan/yellowish discoloration
    
    Note: Lesion detection supports visual interpretation but does NOT
    directly determine disease classification (that's the CNN's role).
    
    Args:
        img: BGR image array
        leaf_mask: Binary mask of leaf region
        
    Returns:
        Tuple of (lesion_mask, lesion_metrics)
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Brown disease spots
    brown_spots = cv2.inRange(hsv, (5, 30, 40), (22, 200, 200))
    
    # Tan/yellowish areas
    tan_areas = cv2.inRange(hsv, (18, 40, 120), (30, 150, 220))
    
    # Combine lesion types
    lesion_mask = cv2.bitwise_or(brown_spots, tan_areas)
    
    # Apply leaf mask
    lesion_mask = cv2.bitwise_and(lesion_mask, lesion_mask, mask=leaf_mask)
    
    # Morphological cleanup
    kernel = np.ones((5, 5), np.uint8)
    lesion_mask = cv2.morphologyEx(lesion_mask, cv2.MORPH_OPEN, kernel)
    lesion_mask = cv2.morphologyEx(lesion_mask, cv2.MORPH_CLOSE, kernel)
    
    # Filter small noise regions
    contours, _ = cv2.findContours(lesion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_mask = np.zeros_like(lesion_mask)
    lesion_count = 0
    
    for cnt in contours:
        if cv2.contourArea(cnt) > 100:  # Minimum area threshold
            cv2.drawContours(filtered_mask, [cnt], -1, 255, -1)
            lesion_count += 1
    
    # Calculate metrics
    lesion_area = np.count_nonzero(filtered_mask)
    leaf_area = np.count_nonzero(leaf_mask)
    lesion_percent = (lesion_area / max(leaf_area, 1)) * 100
    
    metrics = {
        "lesion_count": lesion_count,
        "lesion_area_percent": round(lesion_percent, 2)
    }
    
    return filtered_mask, metrics


# ============================================================================
# CNN CLASSIFICATION WITH TEST-TIME AUGMENTATION (TTA)
# ============================================================================
def apply_tta_augmentations(img: np.ndarray) -> List[np.ndarray]:
    """
    Generate augmented versions of an image for TTA.
    
    Augmentations:
    1. Original image
    2. Horizontal flip
    3. Vertical flip
    4. Rotate 90 degrees
    5. Rotate -90 degrees
    
    Args:
        img: BGR image array
        
    Returns:
        List of augmented images
    """
    augmented = [img]  # Original
    
    # Horizontal flip
    augmented.append(cv2.flip(img, 1))
    
    # Vertical flip
    augmented.append(cv2.flip(img, 0))
    
    # Rotate 90 degrees clockwise
    augmented.append(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))
    
    # Rotate 90 degrees counter-clockwise
    augmented.append(cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE))
    
    return augmented[:TTA_AUGMENTATIONS]


def classify_leaf(model: nn.Module, img: np.ndarray) -> Dict:
    """
    Classify a single leaf image using the CNN model with optional TTA.
    
    When TTA is enabled, the image is augmented multiple times and
    predictions are averaged for more stable, higher-confidence results.
    
    The CNN is the SOLE decision-making component for disease classification.
    
    Args:
        model: Loaded PyTorch model
        img: BGR image array
        
    Returns:
        Dictionary with class probabilities and prediction
    """
    if ENABLE_TTA:
        # Test-Time Augmentation: predict on multiple versions
        augmented_images = apply_tta_augmentations(img)
        all_probs = []
        
        for aug_img in augmented_images:
            x = transform(aug_img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            all_probs.append(probs)
        
        # Average predictions across all augmentations
        probs = np.mean(all_probs, axis=0)
    else:
        # Standard single prediction
        x = transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    
    predicted_idx = int(np.argmax(probs))
    
    return {
        "probabilities": {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))},
        "predicted_class": CLASSES[predicted_idx],
        "confidence": float(probs[predicted_idx])
    }


# ============================================================================
# HEALTH SCORE CALCULATION
# ============================================================================
def calculate_health_score(
    cnn_result: Dict,
    lesion_percent: float
) -> Tuple[float, str]:
    """
    Calculate health score as a heuristic indicator of plant condition.
    
    IMPORTANT: The health score is NOT a medical or biological diagnosis.
    It is a relative, comparative indicator designed for user interpretability.
    
    Formula:
    health_score = (CNN_component * WEIGHT_CNN) + (Lesion_component * WEIGHT_LESION)
    
    Where:
    - CNN_component: Based on classification result
      - If Healthy: confidence * 100
      - If Diseased: (1 - confidence) * 50 (penalized)
    - Lesion_component: (100 - lesion_percent)
    
    Args:
        cnn_result: Classification result from CNN
        lesion_percent: Average lesion area percentage
        
    Returns:
        Tuple of (health_score, condition_label)
    """
    predicted_class = cnn_result["predicted_class"]
    confidence = cnn_result["confidence"]
    
    # CNN component
    if predicted_class == "Healthy":
        cnn_component = confidence * 100
    else:
        # Penalize diseased predictions
        cnn_component = (1 - confidence) * 50
    
    # Lesion component (inverse - less lesion = higher score)
    lesion_component = max(0, 100 - lesion_percent)
    
    # Weighted combination
    health_score = (
        cnn_component * WEIGHT_CNN_CONFIDENCE +
        lesion_component * WEIGHT_LESION_ABSENCE
    )
    
    # Normalize to 0-100
    health_score = min(100, max(0, health_score))
    
    # Determine condition label
    if predicted_class == "Healthy" and confidence >= CONFIDENCE_MEDIUM:
        condition = "Healthy"
    elif predicted_class != "Healthy" and confidence >= CONFIDENCE_MEDIUM:
        condition = "Diseased"
    else:
        condition = "Uncertain"
    
    return round(health_score, 1), condition


# ============================================================================
# VISUALIZATION
# ============================================================================
def create_overlay(img: np.ndarray, vein_mask: np.ndarray, lesion_mask: np.ndarray) -> np.ndarray:
    """
    Create visual overlay for decision transparency.
    
    Colors:
    - Green: Detected vein structures
    - Red: Detected lesion regions
    
    These overlays serve as visual explanations to help users understand
    the basis of the system's analysis.
    
    Args:
        img: Original BGR image
        vein_mask: Binary mask of veins
        lesion_mask: Binary mask of lesions
        
    Returns:
        BGR image with overlays applied
    """
    overlay = img.copy()
    
    # Green overlay for veins (with transparency)
    vein_overlay = overlay.copy()
    vein_overlay[vein_mask > 0] = [0, 255, 0]
    overlay = cv2.addWeighted(overlay, 0.7, vein_overlay, 0.3, 0)
    
    # Red overlay for lesions (more prominent)
    overlay[lesion_mask > 0] = [0, 0, 255]
    
    return overlay


# ============================================================================
# MAIN ANALYSIS PIPELINE
# ============================================================================
def analyze_plant(
    image_folder: str,
    output_dir: str = OUTPUT_DIR,
    min_images: int = MIN_IMAGES_REQUIRED
) -> Dict:
    """
    Analyze all leaf images from a single plant.
    
    Pipeline:
    1. Validate input (minimum images requirement)
    2. Load CNN model
    3. Process each leaf image:
       - Segment leaf region
       - Extract veins and compute morphometry
       - Detect lesions
       - Classify using CNN
    4. Aggregate results at plant level
    5. Calculate health score
    6. Generate visual overlays
    7. Save comprehensive report
    
    Args:
        image_folder: Path to folder containing leaf images
        output_dir: Path for output files
        min_images: Minimum required images (default: 3)
        
    Returns:
        Complete analysis result dictionary
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Validate model exists
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    
    # Get image files
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = sorted([
        f for f in os.listdir(image_folder)
        if f.lower().endswith(valid_extensions)
    ])
    
    # Validate minimum images
    if len(image_files) < min_images:
        return {
            "status": "error",
            "message": f"Minimum {min_images} leaf images required. Found: {len(image_files)}",
            "images_found": len(image_files)
        }
    
    print("=" * 60)
    print("SmartPlant - Rice Leaf Health Analysis System")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_PATH}")
    print(f"Input: {image_folder} ({len(image_files)} images)")
    print("-" * 60)
    
    # Load model
    model = load_model(MODEL_PATH)
    
    # Process each leaf
    leaf_results = []
    overlay_paths = []
    total_lesion_percent = 0
    
    for i, filename in enumerate(image_files):
        img_path = os.path.join(image_folder, filename)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"  [SKIP] Cannot read: {filename}")
            continue
        
        # Image processing pipeline
        leaf_mask = segment_leaf(img)
        vein_mask = extract_veins(img, leaf_mask)
        lesion_mask, lesion_metrics = detect_lesions(img, leaf_mask)
        
        # CNN classification (sole decision maker)
        cnn_result = classify_leaf(model, img)
        
        # Vein morphometry (descriptive only)
        vein_metrics = compute_vein_morphometry(vein_mask, leaf_mask)
        
        # Create and save overlay
        overlay = create_overlay(img, vein_mask, lesion_mask)
        overlay_path = os.path.join(output_dir, f"leaf_{i+1}_overlay.png")
        cv2.imwrite(overlay_path, overlay)
        overlay_paths.append(overlay_path)
        
        # Accumulate results
        leaf_results.append({
            "leaf_index": i + 1,
            "filename": filename,
            "classification": cnn_result,
            "lesion_metrics": lesion_metrics,
            "vein_morphometry": vein_metrics
        })
        
        total_lesion_percent += lesion_metrics["lesion_area_percent"]
        
        # Print progress
        conf_pct = cnn_result["confidence"] * 100
        print(f"  Leaf {i+1}: {cnn_result['predicted_class']} ({conf_pct:.1f}%) - {filename}")
    
    if not leaf_results:
        return {"status": "error", "message": "No valid images could be processed"}
    
    # Aggregate plant-level results
    num_leaves = len(leaf_results)
    avg_lesion_percent = total_lesion_percent / num_leaves
    
    # Average class probabilities across all leaves
    avg_probs = {}
    for cls in CLASSES:
        avg_probs[cls] = np.mean([
            r["classification"]["probabilities"][cls] for r in leaf_results
        ])
    
    # Final classification (maximum averaged probability)
    final_class = max(avg_probs, key=avg_probs.get)
    final_confidence = avg_probs[final_class]
    
    plant_classification = {
        "predicted_class": final_class,
        "confidence": round(final_confidence, 3),
        "class_probabilities": {k: round(v, 4) for k, v in avg_probs.items()}
    }
    
    # Health score calculation
    health_score, condition = calculate_health_score(
        plant_classification,
        avg_lesion_percent
    )
    
    # Build final result
    result = {
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        "system_version": "1.0",
        "input": {
            "folder": image_folder,
            "num_leaves_analyzed": num_leaves
        },
        "leaf_results": leaf_results,
        "overlay_images": overlay_paths,
        "plant_summary": {
            "classification": plant_classification,
            "health_score": health_score,
            "condition": condition,
            "avg_lesion_area_percent": round(avg_lesion_percent, 2),
            "total_lesion_count": sum(r["lesion_metrics"]["lesion_count"] for r in leaf_results)
        },
        "interpretation": {
            "classification_note": "Based solely on CNN deep learning analysis",
            "health_score_note": "Heuristic indicator (0-100), not a biological diagnosis",
            "overlay_note": "Green=veins (descriptive), Red=lesions (visual support)"
        }
    }
    
    # Save JSON report
    report_path = os.path.join(output_dir, "analysis_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print("-" * 60)
    print(f"Report saved: {report_path}")
    
    return result


# ============================================================================
# ENTRY POINT
# ============================================================================
def print_summary(result: Dict):
    """Print human-readable summary of analysis results."""
    if result.get("status") != "success":
        print(f"\n[ERROR] {result.get('message', 'Unknown error')}")
        return
    
    summary = result["plant_summary"]
    classification = summary["classification"]
    
    print("\n" + "=" * 60)
    print("PLANT HEALTH ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"  Leaves Analyzed    : {result['input']['num_leaves_analyzed']}")
    print(f"  Predicted Class    : {classification['predicted_class']}")
    print(f"  Confidence         : {classification['confidence']*100:.1f}%")
    print(f"  Health Score       : {summary['health_score']}/100")
    print(f"  Condition          : {summary['condition']}")
    print(f"  Avg Lesion Area    : {summary['avg_lesion_area_percent']:.1f}%")
    print(f"  Total Lesions      : {summary['total_lesion_count']}")
    print("-" * 60)
    print("Class Probabilities:")
    for cls, prob in classification['class_probabilities'].items():
        bar = "█" * int(prob * 20)
        print(f"    {cls:12s}: {prob*100:5.1f}% {bar}")
    print("=" * 60)
    
    # Interpretation guidance
    if summary["condition"] == "Healthy":
        print("✓ Plant appears healthy based on analysis.")
    elif summary["condition"] == "Diseased":
        print(f"⚠ Possible disease detected: {classification['predicted_class']}")
        print("  Recommend further inspection by agricultural expert.")
    else:
        print("? Analysis inconclusive. Consider:")
        print("  - Capturing additional leaf images")
        print("  - Ensuring good lighting conditions")
        print("  - Consulting with agricultural expert")
    print("=" * 60)


if __name__ == "__main__":
    result = analyze_plant("sample_leaves")
    print_summary(result)
