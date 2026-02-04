import os
import cv2
import json
import torch
import numpy as np
from torchvision import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from skimage.filters import frangi
from skimage.morphology import skeletonize

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["BrownSpot", "Healthy", "Hispa", "LeafBlast"]

# ================= UTIL =================

def to_python_type(obj):
    if isinstance(obj, dict):
        return {k: to_python_type(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_python_type(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj

# ================= CNN =================

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

# ================= VEIN MORPHOMETRY =================

def apply_clahe(gray):
    return cv2.createCLAHE(2.0, (8, 8)).apply(gray)

def extract_vein_mask(gray):
    enhanced = apply_clahe(gray)
    vesselness = frangi(enhanced / 255.0)
    mask = vesselness > np.percentile(vesselness, 90)
    return mask.astype(np.uint8)

def skeleton_and_branch(mask):
    skel = skeletonize(mask > 0).astype(np.uint8)
    kernel = np.array([[1,1,1],[1,10,1],[1,1,1]])
    neighbors = cv2.filter2D(skel, -1, kernel)
    branch = np.logical_and(skel == 1, neighbors >= 13)
    return skel, branch.astype(np.uint8)

def calculate_morphometry(skel, branch):
    length = int(np.sum(skel))
    branches = int(np.sum(branch))
    density = branches / (length + 1e-6)
    return length, branches, float(density)

# ================= HSV LESION =================

def lesion_mask_and_ratio(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (10,40,40), (35,255,255))
    ratio = np.sum(mask > 0) / mask.size
    return mask, float(ratio)

# ================= OVERLAY =================

def save_overlay(img, vein_mask, lesion_mask, out_path):
    overlay = img.copy()
    overlay[vein_mask > 0] = [0, 255, 0]
    overlay[lesion_mask > 0] = [0, 0, 255]
    cv2.imwrite(out_path, overlay)

# ================= RULE-BASED AGGREGATION =================

def aggregate_results(cnn_outs, morph_outs, lesion_outs):
    avg_prob = {c: float(np.mean([o[c] for o in cnn_outs])) for c in CLASSES}
    predicted_class = max(avg_prob, key=avg_prob.get)

    disease_conf = 1.0 - avg_prob.get("Healthy", 0.0)
    avg_density = float(np.mean([m["branch_density"] for m in morph_outs]))
    avg_lesion = float(np.mean(lesion_outs))

    if disease_conf < 0.35 and avg_density >= 0.03 and avg_lesion < 0.05:
        status = "Healthy"
    elif disease_conf < 0.5:
        status = "Early Stress / Uncertain"
    else:
        status = "Diseased"

    base_score = 70 if avg_density >= 0.03 else 55
    score = base_score + (avg_density * 500) - (avg_lesion * 100) - (disease_conf * 40)
    score = int(min(max(score, 0), 100))

    return {
        "plant_condition": status,
        "predicted_class": predicted_class,
        "confidence": round(avg_prob[predicted_class], 3),
        "health_score": score,
        "avg_branch_density": round(avg_density, 5),
        "avg_lesion_ratio": round(avg_lesion, 4)
    }

# ================= MAIN PIPELINE =================

def analyze_plant(model_path, image_folder):
    model = load_cnn_model(model_path)

    cnn_results = []
    morph_results = []
    lesion_results = []
    overlay_paths = []

    os.makedirs("outputs", exist_ok=True)

    for idx, fname in enumerate(sorted(os.listdir(image_folder))):
        img_path = os.path.join(image_folder, fname)
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        cnn_results.append(cnn_predict(model, img_path))

        vein_mask = extract_vein_mask(gray)
        skel, branch = skeleton_and_branch(vein_mask)
        length, branches, density = calculate_morphometry(skel, branch)

        morph_results.append({
            "vein_length": length,
            "branch_count": branches,
            "branch_density": density
        })

        lesion_mask, lesion_ratio = lesion_mask_and_ratio(img)
        lesion_results.append(lesion_ratio)

        overlay_path = f"outputs/overlay_leaf_{idx+1}.png"
        save_overlay(img, vein_mask, lesion_mask, overlay_path)
        overlay_paths.append(overlay_path)

    final_result = aggregate_results(cnn_results, morph_results, lesion_results)

    return to_python_type({
        "num_leaves": len(cnn_results),
        "leaf_predictions": cnn_results,
        "morphometry": morph_results,
        "overlay_images": overlay_paths,
        "final_result": final_result
    })

# ================= RUN =================

if __name__ == "__main__":
    result = analyze_plant(
        model_path="best_model_fixed.pth",
        image_folder="sample_leaves"
    )
    print(json.dumps(result, indent=2))
