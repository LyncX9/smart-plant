"""
Model Comparison Script
Compare all available .pth models on sample_leaves dataset
"""

import os
import torch
import torch.nn as nn
import numpy as np
import cv2
import torchvision.transforms as T
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights, efficientnet_b0, EfficientNet_B0_Weights

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["BrownSpot", "Healthy", "Hispa", "LeafBlast"]

MODELS = {
    "best_model.pth": "mobilenet_original",
    "best_model_fixed.pth": "mobilenet_fixed",
    "best_model_finetuned.pth": "mobilenet_finetuned",
    "best_model_v2.pth": "mobilenet_v2_simple",
    "best_model_optimized.pth": "efficientnet_b0",
}

transform_no_norm = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor()
])

transform_with_norm = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_mobilenet_original(path):
    model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.last_channel, len(CLASSES))
    state = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model, transform_no_norm


def load_mobilenet_fixed(path):
    model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.last_channel, len(CLASSES))
    )
    state = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model, transform_no_norm


def load_mobilenet_finetuned(path):
    model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.last_channel, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, len(CLASSES))
    )
    state = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model, transform_with_norm


def load_mobilenet_v2_simple(path):
    model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.last_channel, len(CLASSES))
    )
    state = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model, transform_with_norm


def load_efficientnet_b0(path):
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, len(CLASSES))
    )
    state = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model, transform_with_norm


LOADERS = {
    "mobilenet_original": load_mobilenet_original,
    "mobilenet_fixed": load_mobilenet_fixed,
    "mobilenet_finetuned": load_mobilenet_finetuned,
    "mobilenet_v2_simple": load_mobilenet_v2_simple,
    "efficientnet_b0": load_efficientnet_b0,
}


def evaluate_model(model, trans, image_folder):
    predictions = []
    confidences = []
    
    for name in sorted(os.listdir(image_folder)):
        img_path = os.path.join(image_folder, name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        x = trans(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        
        pred_class = CLASSES[np.argmax(probs)]
        confidence = float(np.max(probs))
        
        predictions.append(pred_class)
        confidences.append(confidence)
    
    avg_conf = np.mean(confidences)
    healthy_count = predictions.count("Healthy")
    
    return avg_conf, healthy_count, len(predictions), predictions, confidences


def main():
    image_folder = "sample_leaves"
    
    print("=" * 70)
    print("MODEL COMPARISON - SmartPlant Rice Leaf Disease Detection")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Test folder: {image_folder}")
    print("-" * 70)
    
    results = []
    
    for model_file, model_type in MODELS.items():
        if not os.path.exists(model_file):
            print(f"[SKIP] {model_file} not found")
            continue
        
        try:
            loader = LOADERS[model_type]
            model, trans = loader(model_file)
            
            avg_conf, healthy_count, total, preds, confs = evaluate_model(model, trans, image_folder)
            
            results.append({
                "model": model_file,
                "type": model_type,
                "avg_confidence": avg_conf,
                "healthy_detected": healthy_count,
                "total_images": total,
                "predictions": preds,
                "confidences": confs
            })
            
            print(f"\n{model_file} ({model_type})")
            print(f"  Avg Confidence: {avg_conf*100:.1f}%")
            print(f"  Healthy Detected: {healthy_count}/{total}")
            print(f"  Predictions: {preds}")
            
        except Exception as e:
            print(f"[ERROR] {model_file}: {e}")
    
    print("\n" + "=" * 70)
    print("RANKING (by Average Confidence)")
    print("=" * 70)
    
    results_sorted = sorted(results, key=lambda x: x["avg_confidence"], reverse=True)
    
    for i, r in enumerate(results_sorted):
        print(f"{i+1}. {r['model']:30s} - Conf: {r['avg_confidence']*100:5.1f}% | Healthy: {r['healthy_detected']}/{r['total_images']}")
    
    if results_sorted:
        best = results_sorted[0]
        print("\n" + "=" * 70)
        print(f"BEST MODEL: {best['model']}")
        print(f"  Type: {best['type']}")
        print(f"  Average Confidence: {best['avg_confidence']*100:.1f}%")
        print(f"  Healthy Leaves Detected: {best['healthy_detected']}/{best['total_images']}")
        print("=" * 70)
    
    # Save to JSON for reliable viewing
    import json
    output = {
        "ranking": [
            {
                "rank": i+1,
                "model": r["model"],
                "type": r["type"],
                "avg_confidence_percent": round(r["avg_confidence"]*100, 1),
                "healthy_detected": r["healthy_detected"],
                "total_images": r["total_images"],
                "per_leaf_predictions": r["predictions"],
                "per_leaf_confidences": [round(c*100, 1) for c in r["confidences"]]
            }
            for i, r in enumerate(results_sorted)
        ],
        "best_model": results_sorted[0]["model"] if results_sorted else None
    }
    with open("model_comparison.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nResults saved to: model_comparison.json")


if __name__ == "__main__":
    main()
