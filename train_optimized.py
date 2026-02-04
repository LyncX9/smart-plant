import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights, efficientnet_b0, EfficientNet_B0_Weights
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter
from tqdm import tqdm
import os
import random

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()


def mixup_data(x, y, alpha=0.4):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_transform_phase2 = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder("RiceLeafs/train", transform=train_transform)
    val_dataset = datasets.ImageFolder("RiceLeafs/validation", transform=val_transform)

    print(f"Classes: {train_dataset.classes}")
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    labels = [label for _, label in train_dataset.samples]
    class_counts = Counter(labels)
    num_classes = len(class_counts)
    print(f"Class distribution: {dict(class_counts)}")

    sample_weights = []
    for _, label in train_dataset.samples:
        weight = 1.0 / class_counts[label]
        sample_weights.append(weight)
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True
    )

    train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)

    total_samples = sum(class_counts.values())
    class_weights = torch.tensor(
        [total_samples / (num_classes * class_counts[i]) for i in range(num_classes)],
        dtype=torch.float
    ).to(device)
    print(f"Class weights: {class_weights}")

    print("\nLoading EfficientNet-B0 (stronger than MobileNetV2)...")
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

    for param in model.features.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.classifier[1].in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes)
    )

    model = model.to(device)

    criterion = FocalLoss(alpha=class_weights, gamma=2.0)

    print("\n========== PHASE 1: Train Classifier (20 epochs) ==========")
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-5)

    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(20):
        train_loss, train_acc = train_epoch_simple(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch+1}/20 | Train: {train_acc:.2f}% | Val: {val_acc:.2f}% | LR: {scheduler.get_last_lr()[0]:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model_optimized.pth")
            print("  -> Saved!")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 7:
                print("Early stop Phase 1")
                break

    print(f"Phase 1 Best: {best_val_acc:.2f}%")

    print("\n========== PHASE 2: Fine-tune Backbone (30 epochs) ==========")
    
    # Reload dataset with stronger augmentation for Phase 2
    train_dataset_p2 = datasets.ImageFolder("RiceLeafs/train", transform=train_transform_phase2)
    train_loader = DataLoader(train_dataset_p2, batch_size=16, sampler=sampler, num_workers=0, pin_memory=True)
    
    model.load_state_dict(torch.load("best_model_optimized.pth", map_location=device))

    for i, block in enumerate(model.features):
        if i >= 6:
            for param in block.parameters():
                param.requires_grad = True

    optimizer = torch.optim.AdamW([
        {'params': model.classifier.parameters(), 'lr': 5e-4},
        {'params': filter(lambda p: p.requires_grad, model.features.parameters()), 'lr': 5e-5}
    ], weight_decay=0.01)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)

    patience_counter = 0
    best_val_acc_p2 = best_val_acc

    for epoch in range(30):
        train_loss, train_acc = train_epoch_mixup(model, train_loader, criterion, optimizer, None, device, use_mixup=True)
        val_loss, val_acc, all_preds, all_labels = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch+1}/30 | Train: {train_acc:.2f}% | Val: {val_acc:.2f}%")

        if val_acc > best_val_acc_p2:
            best_val_acc_p2 = val_acc
            torch.save(model.state_dict(), "best_model_optimized.pth")
            print("  -> Saved!")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print("Early stop Phase 2")
                break

    print(f"\n========== FINAL ==========")
    print(f"Best Val Accuracy: {best_val_acc_p2:.2f}%")

    model.load_state_dict(torch.load("best_model_optimized.pth", map_location=device))
    _, _, all_preds, all_labels = validate(model, val_loader, criterion, device)

    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=val_dataset.classes, digits=4))


def train_epoch_simple(model, loader, criterion, optimizer, device):
    """Simple training without Mixup - for Phase 1"""
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, targets in tqdm(loader, desc="Train"):
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (preds == targets).sum().item()

    return running_loss / len(loader), 100.0 * correct / total


def train_epoch_mixup(model, loader, criterion, optimizer, scheduler, device, use_mixup=True):
    """Training with optional Mixup - for Phase 2"""
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, targets in tqdm(loader, desc="Train"):
        images, targets = images.to(device), targets.to(device)

        if use_mixup:
            mixed_images, targets_a, targets_b, lam = mixup_data(images, targets, alpha=0.3)
            optimizer.zero_grad()
            outputs = model(mixed_images)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            lam = 1.0
            targets_a, targets_b = targets, targets

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if scheduler:
            scheduler.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        total += targets.size(0)
        if use_mixup:
            correct += (lam * (preds == targets_a).float() + (1 - lam) * (preds == targets_b).float()).sum().item()
        else:
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
