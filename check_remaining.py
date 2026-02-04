"""
Test best_model_optimized.pth and best_model_finetuned.pth on VALIDATION set
"""
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights, efficientnet_b0, EfficientNet_B0_Weights
from sklearn.metrics import classification_report, confusion_matrix

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["BrownSpot", "Healthy", "Hispa", "LeafBlast"]

def load_model_optimized(path):
    """EfficientNet-B0 with complex classifier"""
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
    return model

def load_model_finetuned(path):
    """MobileNetV2 with deeper classifier"""
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
    return model

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def evaluate(model, val_loader):
    all_preds, all_labels, all_confs = [], [], []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_confs.extend(confs.cpu().numpy())
    
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels)) * 100
    avg_conf = np.mean(all_confs) * 100
    return accuracy, avg_conf, all_preds, all_labels

def check_verdict(acc, conf):
    if conf > 70 and acc < 50:
        return "OVERFITTING"
    elif conf > 50 and acc > 50:
        return "GOOD"
    elif conf < 40 and acc < 40:
        return "UNDERFITTING"
    else:
        return "ACCEPTABLE"

def main():
    val_dataset = datasets.ImageFolder("RiceLeafs/validation", transform=val_transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print("=" * 60)
    print("Testing Models on Validation Set (671 samples)")
    print("=" * 60)
    
    results = []
    
    # Test best_model_optimized.pth
    print("\n1. best_model_optimized.pth (EfficientNet-B0)")
    try:
        model1 = load_model_optimized("best_model_optimized.pth")
        acc1, conf1, preds1, labels1 = evaluate(model1, val_loader)
        verdict1 = check_verdict(acc1, conf1)
        print(f"   Accuracy: {acc1:.1f}%")
        print(f"   Confidence: {conf1:.1f}%")
        print(f"   Verdict: {verdict1}")
        results.append(("best_model_optimized.pth", acc1, conf1, verdict1))
    except Exception as e:
        print(f"   ERROR: {e}")
    
    # Test best_model_finetuned.pth
    print("\n2. best_model_finetuned.pth (MobileNetV2 deeper)")
    try:
        model2 = load_model_finetuned("best_model_finetuned.pth")
        acc2, conf2, preds2, labels2 = evaluate(model2, val_loader)
        verdict2 = check_verdict(acc2, conf2)
        print(f"   Accuracy: {acc2:.1f}%")
        print(f"   Confidence: {conf2:.1f}%")
        print(f"   Verdict: {verdict2}")
        results.append(("best_model_finetuned.pth", acc2, conf2, verdict2))
    except Exception as e:
        print(f"   ERROR: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, acc, conf, verdict in results:
        print(f"{name}: Acc={acc:.1f}%, Conf={conf:.1f}%, {verdict}")

if __name__ == "__main__":
    main()
