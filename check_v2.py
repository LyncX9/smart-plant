"""
Test best_model_v2.pth on VALIDATION set
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

def load_model_v2(path):
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

# WITH normalization (as trained)
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def evaluate(model, val_loader):
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
    
    print("=" * 60)
    print("Testing best_model_v2.pth on Validation Set")
    print("=" * 60)
    print(f"Validation samples: {len(val_dataset)}")
    
    model = load_model_v2("best_model_v2.pth")
    acc, conf, preds, labels = evaluate(model, val_loader)
    
    print(f"\nValidation Accuracy: {acc:.1f}%")
    print(f"Average Confidence: {conf:.1f}%")
    
    # Overfitting check
    print("\n" + "-" * 60)
    if conf > 70 and acc < 50:
        print("VERDICT: OVERFITTING (high confidence, low accuracy)")
    elif conf > 50 and acc > 50:
        print("VERDICT: GOOD (balanced confidence and accuracy)")
    elif conf < 40 and acc < 40:
        print("VERDICT: UNDERFITTING (low confidence, low accuracy)")
    else:
        print("VERDICT: ACCEPTABLE (conservative predictions)")
    
    print("\n--- Classification Report ---")
    print(classification_report(labels, preds, target_names=CLASSES, zero_division=0))
    
    print("\n--- Confusion Matrix ---")
    print(confusion_matrix(labels, preds))

if __name__ == "__main__":
    main()
