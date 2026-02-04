import torch
import cv2
import numpy as np
import os
from torchvision import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from collections import Counter

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["BrownSpot", "Healthy", "Hispa", "LeafBlast"]

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def load_cnn_model(model_path):
    model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = torch.nn.Linear(model.last_channel, len(CLASSES))
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def cnn_predict(model, image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    return dict(zip(CLASSES, probs))

def vein_morphometry_score(image_path):
    img = cv2.imread(image_path, 0)
    img = cv2.resize(img, (512, 512))

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(img)

    edges = cv2.Canny(enhanced, 50, 150)
    vein_density = np.sum(edges > 0) / edges.size

    if vein_density > 0.035:
        status = "Good"
    elif vein_density > 0.02:
        status = "Moderate"
    else:
        status = "Poor"

    return {
        "branch_density": round(float(vein_density), 4),
        "status": status
    }

def aggregate_results(cnn_outputs, vein_outputs):
    top_classes = [max(o, key=o.get) for o in cnn_outputs]
    majority_class = Counter(top_classes).most_common(1)[0][0]

    avg_probs = {
        cls: float(np.mean([o[cls] for o in cnn_outputs]))
        for cls in CLASSES
    }

    avg_density = np.mean([v["branch_density"] for v in vein_outputs])

    if avg_density > 0.03:
        vein_health = "Good"
    elif avg_density > 0.02:
        vein_health = "Moderate"
    else:
        vein_health = "Poor"

    health_score = int(
        avg_probs.get("Healthy", 0) * 70 +
        (avg_density * 1000)
    )
    health_score = min(max(health_score, 0), 100)

    return {
        "plant_condition": "Healthy" if majority_class == "Healthy" else "Diseased",
        "predicted_disease": majority_class,
        "confidence": round(avg_probs[majority_class], 2),
        "health_score": health_score,
        "vein_health": vein_health,
        "avg_branch_density": round(float(avg_density), 4)
    }

def analyze_plant(model_path, image_folder):
    model = load_cnn_model(model_path)

    cnn_outputs = []
    vein_outputs = []

    for img_name in sorted(os.listdir(image_folder)):
        img_path = os.path.join(image_folder, img_name)
        cnn_outputs.append(cnn_predict(model, img_path))
        vein_outputs.append(vein_morphometry_score(img_path))

    final_result = aggregate_results(cnn_outputs, vein_outputs)

    return {
        "num_leaves": len(cnn_outputs),
        "leaf_predictions": cnn_outputs,
        "vein_analysis": vein_outputs,
        "final_result": final_result
    }

if __name__ == "__main__":
    result = analyze_plant(
        model_path="best_model_fixed.pth",
        image_folder="sample_leaves"
    )

    import json
    print(json.dumps(result, indent=2))
