"""
Quick Optimized Training Script for best_model_fixed.pth
Target: Improve accuracy from 37% to 45-55% without overfitting

Techniques:
- Class-balanced sampling (WeightedRandomSampler)
- Cosine Annealing LR with warm restarts
- Label smoothing
- Moderate augmentation (not too aggressive)
- Gradient clipping
- Early stopping with patience

Estimated time: 15-30 minutes on CPU
"""

import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter
from tqdm import tqdm

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Moderate augmentation - not too aggressive
    train_transform = transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        # NO normalization - matching original model
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    train_dataset = datasets.ImageFolder("RiceLeafs/train", transform=train_transform)
    val_dataset = datasets.ImageFolder("RiceLeafs/validation", transform=val_transform)
    
    print(f"Classes: {train_dataset.classes}")
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Class-balanced sampling
    labels = [label for _, label in train_dataset.samples]
    class_counts = Counter(labels)
    num_classes = len(class_counts)
    print(f"Class distribution: {dict(class_counts)}")
    
    # Calculate sample weights for balanced sampling
    class_weights_sampling = {c: len(labels) / count for c, count in class_counts.items()}
    sample_weights = [class_weights_sampling[label] for label in labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # Class weights for loss
    total = sum(class_counts.values())
    class_weights = torch.tensor(
        [total / (num_classes * class_counts[i]) for i in range(num_classes)],
        dtype=torch.float
    ).to(device)
    print(f"Class weights: {class_weights}")
    
    # Model - same architecture as best_model_fixed.pth
    model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.last_channel, num_classes)
    )
    model = model.to(device)
    
    # Loss with label smoothing
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    
    # Phase 1: Train classifier only
    print("\n" + "=" * 60)
    print("PHASE 1: Train Classifier (10 epochs)")
    print("=" * 60)
    
    for param in model.features.parameters():
        param.requires_grad = False
    
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)
    
    best_acc = 0.0
    patience = 5
    patience_counter = 0
    
    for epoch in range(10):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f"Epoch {epoch+1}/10 | Train: {train_acc:.1f}% | Val: {val_acc:.1f}% | LR: {scheduler.get_last_lr()[0]:.6f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model_improved.pth")
            print("  -> Saved!")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stop Phase 1")
                break
    
    print(f"Phase 1 Best: {best_acc:.1f}%")
    
    # Phase 2: Fine-tune backbone (last 4 layers)
    print("\n" + "=" * 60)
    print("PHASE 2: Fine-tune Backbone (15 epochs)")
    print("=" * 60)
    
    model.load_state_dict(torch.load("best_model_improved.pth", map_location=device))
    
    # Unfreeze last 4 layers of features
    for i, layer in enumerate(model.features):
        if i >= 14:
            for param in layer.parameters():
                param.requires_grad = True
    
    optimizer = torch.optim.AdamW([
        {'params': model.classifier.parameters(), 'lr': 5e-4},
        {'params': filter(lambda p: p.requires_grad, model.features.parameters()), 'lr': 5e-5}
    ], weight_decay=0.01)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)
    
    best_acc_p2 = best_acc
    patience_counter = 0
    
    for epoch in range(15):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, preds, labels = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f"Epoch {epoch+1}/15 | Train: {train_acc:.1f}% | Val: {val_acc:.1f}%")
        
        if val_acc > best_acc_p2:
            best_acc_p2 = val_acc
            torch.save(model.state_dict(), "best_model_improved.pth")
            print("  -> Saved!")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 7:
                print("Early stop Phase 2")
                break
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Best Validation Accuracy: {best_acc_p2:.1f}%")
    
    model.load_state_dict(torch.load("best_model_improved.pth", map_location=device))
    _, _, preds, labels = validate(model, val_loader, criterion, device)
    
    # Calculate average confidence
    model.eval()
    all_confs = []
    with torch.no_grad():
        for images, _ in val_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            confs, _ = torch.max(probs, 1)
            all_confs.extend(confs.cpu().numpy())
    avg_conf = np.mean(all_confs) * 100
    print(f"Average Confidence: {avg_conf:.1f}%")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(labels, preds))
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=train_dataset.classes))


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    
    for images, targets in tqdm(loader, desc="Train", leave=False):
        images, targets = images.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (preds == targets).sum().item()
    
    return running_loss / len(loader), 100.0 * correct / total


def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (preds == targets).sum().item()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
    
    return running_loss / len(loader), 100.0 * correct / total, all_preds, all_labels


if __name__ == "__main__":
    main()
