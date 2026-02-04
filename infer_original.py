import os
import json
import cv2
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as T
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from skimage.morphology import skeletonize

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["BrownSpot", "Healthy", "Hispa", "LeafBlast"]


def load_original_model(model_path):
    """Load the original best_model_fixed.pth with correct architecture"""
    model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    
    # Architecture from train_fix_overfitting.py
    model.classifier[1] = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.last_channel, len(CLASSES))
    )

    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


# Transform WITHOUT normalization - matches original training
transform_original = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor()
])


def rice_leaf_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (25, 40, 40), (95, 255, 255))
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def extract_rice_vein(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    leaf_mask = rice_leaf_mask(img)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    tophat = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, kernel_v)

    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    tophat_h = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, kernel_h)

    combined = cv2.add(tophat, tophat_h)
    _, vein = cv2.threshold(combined, 20, 255, cv2.THRESH_BINARY)
    vein = cv2.bitwise_and(vein, vein, mask=leaf_mask)

    kernel_clean = np.ones((2, 2), np.uint8)
    vein = cv2.morphologyEx(vein, cv2.MORPH_OPEN, kernel_clean)
    vein = cv2.morphologyEx(vein, cv2.MORPH_CLOSE, kernel_clean)
    return vein


def vein_morphometry_visual_only(vein_mask, leaf_mask):
    skel = skeletonize(vein_mask > 0)
    length = int(np.sum(skel))
    vein_area = np.count_nonzero(vein_mask)
    leaf_area = np.count_nonzero(leaf_mask)

    continuity = length / (vein_area + 1e-6)
    vein_density = (vein_area / (leaf_area + 1e-6)) * 100

    return {
        "vein_length": length,
        "vein_density_percent": round(float(vein_density), 2),
        "vein_continuity": round(float(continuity), 3)
    }


def lesion_mask(img, leaf_mask):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    brown_spots = cv2.inRange(hsv, (5, 30, 40), (22, 200, 200))
    tan_areas = cv2.inRange(hsv, (18, 40, 120), (30, 150, 220))

    lesion = cv2.bitwise_or(brown_spots, tan_areas)
    lesion = cv2.bitwise_and(lesion, lesion, mask=leaf_mask)

    kernel = np.ones((5, 5), np.uint8)
    lesion = cv2.morphologyEx(lesion, cv2.MORPH_OPEN, kernel)
    lesion = cv2.morphologyEx(lesion, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(lesion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered = np.zeros_like(lesion)
    for cnt in contours:
        if cv2.contourArea(cnt) > 100:
            cv2.drawContours(filtered, [cnt], -1, 255, -1)

    return filtered


def save_overlay(img, vein, lesion, out_path):
    overlay = img.copy()
    overlay[vein > 0] = [0, 255, 0]
    overlay[lesion > 0] = [0, 0, 255]
    cv2.imwrite(out_path, overlay)


def analyze_plant_original(model, image_folder, out_dir="outputs_original"):
    os.makedirs(out_dir, exist_ok=True)
    preds, morphs, overlays = [], [], []

    print(f"Analyzing images in: {image_folder}")

    for i, name in enumerate(sorted(os.listdir(image_folder))):
        img = cv2.imread(os.path.join(image_folder, name))
        if img is None:
            continue

        # Use original transform (no normalization)
        x = transform_original(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1).cpu().numpy()[0]

        preds.append({CLASSES[j]: float(probs[j]) for j in range(len(CLASSES))})

        leaf_mask = rice_leaf_mask(img)
        vein = extract_rice_vein(img)
        lesion = lesion_mask(img, leaf_mask)
        morph = vein_morphometry_visual_only(vein, leaf_mask)
        morphs.append(morph)

        out_path = os.path.join(out_dir, f"overlay_leaf_{i + 1}.png")
        save_overlay(img, vein, lesion, out_path)
        overlays.append(out_path)

        pred_class = CLASSES[np.argmax(probs)]
        pred_conf = max(probs) * 100
        print(f"  Leaf {i+1}: {pred_class} ({pred_conf:.1f}%)")

    avg_probs = np.mean([[p[c] for c in CLASSES] for p in preds], axis=0)
    idx = int(np.argmax(avg_probs))

    healthy_idx = CLASSES.index("Healthy")
    health_score = round(float(avg_probs[healthy_idx]) * 100, 2)

    return {
        "num_leaves": len(preds),
        "leaf_predictions": preds,
        "morphometry_visual": morphs,
        "overlay_images": overlays,
        "final_result": {
            "predicted_class": CLASSES[idx],
            "confidence": round(float(avg_probs[idx]), 3),
            "health_score": health_score,
            "plant_condition": "Healthy" if CLASSES[idx] == "Healthy" else "Diseased"
        }
    }


if __name__ == "__main__":
    model_path = "best_model_fixed.pth"
    
    if os.path.exists(model_path):
        print("Loading original model (best_model_fixed.pth)...")
        model = load_original_model(model_path)
        result = analyze_plant_original(model, "sample_leaves")
        print("\n" + "="*50)
        print(json.dumps(result, indent=2))
    else:
        print(f"Model '{model_path}' not found!")
