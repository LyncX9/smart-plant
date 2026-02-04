"""
Test best_model.pth on VALIDATION set to check for overfitting
"""
import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from sklearn.metrics import classification_report, confusion_matrix

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["BrownSpot", "Healthy", "Hispa", "LeafBlast"]

def load_model_original(path):
    model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.last_channel, len(CLASSES))
    state = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model

def load_model_fixed(path):
    model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.last_channel, len(CLASSES))
    )
    state = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def evaluate_on_validation(model, val_loader):
    all_preds = []
    all_labels = []
    all_confs = []
    
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

def main():
    val_dataset = datasets.ImageFolder("RiceLeafs/validation", transform=val_transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print("=" * 70)
    print("OVERFITTING CHECK - Testing on VALIDATION set")
    print("=" * 70)
    print(f"Validation samples: {len(val_dataset)}")
    print("-" * 70)
    
    # Test best_model.pth
    print("\n1. best_model.pth (original):")
    model1 = load_model_original("best_model.pth")
    acc1, conf1, preds1, labels1 = evaluate_on_validation(model1, val_loader)
    print(f"   Validation Accuracy: {acc1:.1f}%")
    print(f"   Average Confidence: {conf1:.1f}%")
    
    # Test best_model_fixed.pth
    print("\n2. best_model_fixed.pth (with dropout):")
    model2 = load_model_fixed("best_model_fixed.pth")
    acc2, conf2, preds2, labels2 = evaluate_on_validation(model2, val_loader)
    print(f"   Validation Accuracy: {acc2:.1f}%")
    print(f"   Average Confidence: {conf2:.1f}%")
    
    print("\n" + "=" * 70)
    print("ANALYSIS:")
    print("=" * 70)
    
    if acc1 > acc2:
        print(f"best_model.pth is BETTER on validation ({acc1:.1f}% vs {acc2:.1f}%)")
        if conf1 > 80 and acc1 > 70:
            print("High confidence + high accuracy = NOT overfitting")
        elif conf1 > 80 and acc1 < 50:
            print("High confidence + low accuracy = OVERFITTING!")
    else:
        print(f"best_model_fixed.pth is BETTER on validation ({acc2:.1f}% vs {acc1:.1f}%)")
    
    print("\n--- Classification Report (best_model.pth) ---")
    print(classification_report(labels1, preds1, target_names=CLASSES))
    
    print("\n--- Confusion Matrix (best_model.pth) ---")
    print(confusion_matrix(labels1, preds1))

if __name__ == "__main__":
    main()
