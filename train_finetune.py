import torch
import numpy as np
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter
from tqdm import tqdm
import os

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
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
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)

    labels = [label for _, label in train_dataset.samples]
    class_counts = Counter(labels)
    num_classes = len(class_counts)

    total_samples = sum(class_counts.values())
    class_weights = torch.tensor(
        [total_samples / (num_classes * class_counts[i]) for i in range(num_classes)],
        dtype=torch.float
    ).to(device)
    print(f"Class weights: {class_weights}")

    weights = MobileNet_V2_Weights.DEFAULT
    model = mobilenet_v2(weights=weights)

    for param in model.features.parameters():
        param.requires_grad = False

    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.4),
        torch.nn.Linear(model.last_channel, 256),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(256, num_classes)
    )

    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    print("\n========== PHASE 1: Train Classifier Only ==========")
    optimizer = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=1e-3,
        weight_decay=1e-2
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)

    best_val_acc = 0.0
    patience = 5
    patience_counter = 0

    for epoch in range(10):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch+1}/10 | Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | LR: {scheduler.get_last_lr()[0]:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model_finetuned.pth")
            print("  -> Best model saved!")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping Phase 1")
            break

    print(f"\nPhase 1 Best Val Acc: {best_val_acc:.2f}%")

    print("\n========== PHASE 2: Fine-tune Backbone ==========")
    model.load_state_dict(torch.load("best_model_finetuned.pth", map_location=device))

    for i, layer in enumerate(model.features):
        if i >= 14:
            for param in layer.parameters():
                param.requires_grad = True

    optimizer = torch.optim.AdamW([
        {'params': model.classifier.parameters(), 'lr': 1e-4},
        {'params': filter(lambda p: p.requires_grad, model.features.parameters()), 'lr': 1e-5}
    ], weight_decay=1e-2)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)

    patience_counter = 0
    best_val_acc_phase2 = best_val_acc

    for epoch in range(20):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, all_preds, all_labels = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch+1}/20 | Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc_phase2:
            best_val_acc_phase2 = val_acc
            torch.save(model.state_dict(), "best_model_finetuned.pth")
            print("  -> Best model saved!")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= 7:
            print("Early stopping Phase 2")
            break

    print(f"\n========== FINAL RESULTS ==========")
    print(f"Best Validation Accuracy: {best_val_acc_phase2:.2f}%")

    model.load_state_dict(torch.load("best_model_finetuned.pth", map_location=device))
    _, _, all_preds, all_labels = validate(model, val_loader, criterion, device)

    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    print("\nClassification Report:")
    print(classification_report(
        all_labels,
        all_preds,
        target_names=val_dataset.classes,
        digits=4
    ))


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(loader, desc="Training")
    for images, targets in loop:
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

        loop.set_postfix(loss=loss.item(), acc=100.0 * correct / total)

    return running_loss / len(loader), 100.0 * correct / total


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

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
