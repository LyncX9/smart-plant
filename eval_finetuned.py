"""
Evaluasi khusus untuk best_model_finetuned.pth
Model ini punya custom classifier architecture
"""

import torch
import numpy as np
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v2
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, f1_score

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Transform sama seperti saat training
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    print("\n📦 Loading datasets...")
    train_dataset = datasets.ImageFolder("RiceLeafs/train", transform=val_transform)
    val_dataset = datasets.ImageFolder("RiceLeafs/validation", transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    num_classes = len(val_dataset.classes)
    class_names = val_dataset.classes
    print(f"Classes: {class_names}")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Build model dengan arsitektur yang sama seperti train_finetune.py
    print("\n🔧 Building model with custom classifier architecture...")
    model = mobilenet_v2(weights=None)
    
    # Custom classifier yang sama dengan train_finetune.py
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.4),
        torch.nn.Linear(model.last_channel, 256),  # 1280 -> 256
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(256, num_classes)  # 256 -> 4
    )
    
    # Load weights
    print("📥 Loading best_model_finetuned.pth...")
    state_dict = torch.load("best_model_finetuned.pth", map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print("✅ Model loaded successfully!")
    
    # Evaluate on training set
    print("\n📊 Evaluating on TRAIN set...")
    train_acc, train_preds, train_labels, train_conf = evaluate(model, train_loader, device)
    train_f1 = f1_score(train_labels, train_preds, average='macro')
    
    # Evaluate on validation set
    print("📊 Evaluating on VALIDATION set...")
    val_acc, val_preds, val_labels, val_conf = evaluate(model, val_loader, device)
    val_f1 = f1_score(val_labels, val_preds, average='macro')
    val_f1_per_class = f1_score(val_labels, val_preds, average=None)
    
    # Overfitting check
    overfit_gap = train_acc - val_acc
    if overfit_gap > 15:
        fit_status = "⚠️ OVERFITTING"
    elif overfit_gap < -5:
        fit_status = "⚠️ UNDERFITTING"
    elif val_acc < 40:
        fit_status = "⚠️ LOW ACCURACY"
    else:
        fit_status = "✅ GOOD FIT"
    
    # Print results
    print("\n" + "="*60)
    print("📈 HASIL EVALUASI: best_model_finetuned.pth")
    print("="*60)
    print(f"\n{'Metrik':<25} {'Train':<15} {'Validation':<15}")
    print("-"*55)
    print(f"{'Accuracy':<25} {train_acc:<15.2f} {val_acc:<15.2f}")
    print(f"{'F1 Macro Score':<25} {train_f1*100:<15.2f} {val_f1*100:<15.2f}")
    print(f"{'Avg Confidence':<25} {np.mean(train_conf)*100:<15.2f} {np.mean(val_conf)*100:<15.2f}")
    
    print(f"\n🔍 Overfit Gap: {overfit_gap:.2f}%")
    print(f"📋 Status: {fit_status}")
    
    print(f"\n📊 Per-Class F1 Score (Validation):")
    for i, name in enumerate(class_names):
        print(f"   {name:<15}: {val_f1_per_class[i]*100:.2f}%")
    
    print("\n📋 Confusion Matrix (Validation):")
    cm = confusion_matrix(val_labels, val_preds)
    print(f"\n{'':>15}", end="")
    for name in class_names:
        print(f"{name[:8]:>10}", end="")
    print()
    for i, name in enumerate(class_names):
        print(f"{name:<15}", end="")
        for j in range(len(class_names)):
            print(f"{cm[i][j]:>10}", end="")
        print()
    
    print("\n📋 Classification Report (Validation):")
    print(classification_report(val_labels, val_preds, target_names=class_names, digits=4))
    
    # Compare with best_model.pth
    print("\n" + "="*60)
    print("📊 PERBANDINGAN DENGAN best_model.pth")
    print("="*60)
    print(f"\n{'Model':<30} {'Val Acc':<12} {'F1 Macro':<12} {'Status':<15}")
    print("-"*70)
    print(f"{'best_model.pth':<30} {'42.92':<12} {'20.21':<12} {'✅ GOOD FIT':<15}")
    print(f"{'best_model_finetuned.pth':<30} {val_acc:<12.2f} {val_f1*100:<12.2f} {fit_status:<15}")
    
    if val_acc > 42.92:
        print(f"\n🏆 WINNER: best_model_finetuned.pth (+{val_acc - 42.92:.2f}% accuracy)")
    else:
        print(f"\n🏆 WINNER: best_model.pth (+{42.92 - val_acc:.2f}% accuracy)")
    
    return val_acc, val_f1 * 100


def evaluate(model, loader, device):
    """Evaluate model on a dataloader."""
    all_preds = []
    all_labels = []
    all_confidences = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            confidences, preds = torch.max(probs, 1)
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())
    
    accuracy = 100.0 * correct / total
    return accuracy, all_preds, all_labels, all_confidences


if __name__ == "__main__":
    main()
