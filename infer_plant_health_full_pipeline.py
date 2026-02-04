import os
import cv2
import json
import torch
import numpy as np
from torchvision import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from skimage.filters import frangi
from skimage.morphology import skeletonize
from skimage.color import rgb2gray

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["BrownSpot", "Healthy", "Hispa", "LeafBlast"]

# ---------------- CNN MODEL ----------------

def load_cnn_model(model_path):
    model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = torch.nn.Sequential(
        torch.nn.Dropout(0.5),
        torch.nn.Linear(model.last_channel, len(CLASSES))
    )
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

cnn_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def cnn_predict(model, img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = cnn_transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    return dict(zip(CLASSES, probs))

# ---------------- VEIN MORPHOMETRY ----------------

def apply_clahe(gray):
    clahe = cv2.createCLAHE(2.0, (8, 8))
    return clahe.apply(gray)

def extract_vein_mask(gray):
    enhanced = apply_clahe(gray)
    vessel = frangi(enhanced / 255.0)
    mask = vessel > np.percentile(vessel, 90)
    return mask.astype(np.uint8)

def skeleton_and_branch(mask):
    skel = skeletonize(mask > 0)
    skel = skel.astype(np.uint8)

    kernel = np.array([[1,1,1],[1,10,1],[1,1,1]])
    neighbors = cv2.filter2D(skel, -1, kernel)
    branch_points = np.logical_and(skel == 1, neighbors >= 13)

    return skel, branch_points.astype(np.uint8)

def calculate_morphometry(skel, branch_points):
    length = np.sum(skel)
    branches = np.sum(branch_points)
    density = branches / (length + 1e-6)
    return {
        "vein_length": int(length),
        "branch_count": int(branches),
        "branch_density": round(float(density), 5)
    }

# ---------------- HSV LESION ----------------

def hsv_lesion_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([10, 40, 40])
    upper = np.array([35, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    lesion_ratio = np.sum(mask > 0) / mask.size
    return mask, round(float(lesion_ratio), 4)

# ---------------- OVERLAY ----------------

def create_overlay(img, vein_mask, lesion_mask):
    overlay = img.copy()
    overlay[vein_mask > 0] = [0, 255, 0]
    overlay[lesion_mask > 0] = [255, 0, 0]
    return overlay

# ---------------- AGGREGATION ----------------

def aggregate(leaves_cnn, leaves_morph):
    top = [max(l, key=l.get) for l in leaves_cnn]
    majority = max(set(top), key=top.count)

    avg_prob = {
        c: float(np.mean([l[c] for l in leaves_cnn]))
        for c in CLASSES
    }

    avg_density = np.mean([m["branch_density"] for m in leaves_morph])

    score = int(avg_prob.get("Healthy", 0) * 70 + avg_density * 1000)
    score = min(max(score, 0), 100)

    return {
        "plant_condition": "Healthy" if majority == "Healthy" else "Diseased",
        "predicted_class": majority,
        "confidence": round(avg_prob[majority], 3),
        "health_score": score,
        "avg_branch_density": round(float(avg_density), 5)
    }

# ---------------- MAIN PIPELINE ----------------

def analyze_plant(model_path, image_folder):
    model = load_cnn_model(model_path)

    cnn_results = []
    morph_results = []

    for img_name in sorted(os.listdir(image_folder)):
        path = os.path.join(image_folder, img_name)
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        cnn_out = cnn_predict(model, path)

        vein_mask = extract_vein_mask(gray)
        skel, branch = skeleton_and_branch(vein_mask)
        morph = calculate_morphometry(skel, branch)

        lesion_mask, lesion_ratio = hsv_lesion_mask(img)
        morph["lesion_ratio"] = lesion_ratio

        cnn_results.append(cnn_out)
        morph_results.append(morph)

    final = aggregate(cnn_results, morph_results)

    return {
        "num_leaves": len(cnn_results),
        "leaf_predictions": cnn_results,
        "morphometry": morph_results,
        "final_result": final
    }

# ---------------- RUN ----------------

def to_python_type(obj):
    if isinstance(obj, dict):
        return {k: to_python_type(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_python_type(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj

if __name__ == "__main__":
    result = analyze_plant(
        model_path="best_model_fixed.pth",
        image_folder="sample_leaves"
    )
    result = to_python_type(result)
    print(json.dumps(result, indent=2))

