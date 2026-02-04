import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter
from tqdm import tqdm

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    total_samples = sum(class_counts.values())
    class_weights = torch.tensor(
        [total_samples / (num_classes * class_counts[i]) for i in range(num_classes)],
        dtype=torch.float
    ).to(device)
    print(f"Class weights: {class_weights}")

    model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

    model.classifier[1] = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.last_channel, num_classes)
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    print("\n========== PHASE 1: Train Classifier Only (15 epochs) ==========")
    
    for param in model.features.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_val_acc = 0.0

    for epoch in range(15):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch+1}/15 | Train: {train_acc:.2f}% | Val: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model_v2.pth")
            print("  -> Saved!")

    print(f"Phase 1 Best: {best_val_acc:.2f}%")

    print("\n========== PHASE 2: Fine-tune All Layers (20 epochs) ==========")
    model.load_state_dict(torch.load("best_model_v2.pth", map_location=device))

    for param in model.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam([
        {'params': model.features.parameters(), 'lr': 1e-5},
        {'params': model.classifier.parameters(), 'lr': 1e-4}
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    best_val_acc_p2 = best_val_acc

    for epoch in range(20):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, all_preds, all_labels = validate(model, val_loader, criterion, device)
        scheduler.step(val_acc)

        print(f"Epoch {epoch+1}/20 | Train: {train_acc:.2f}% | Val: {val_acc:.2f}%")

        if val_acc > best_val_acc_p2:
            best_val_acc_p2 = val_acc
            torch.save(model.state_dict(), "best_model_v2.pth")
            print("  -> Saved!")

    print(f"\n========== FINAL ==========")
    print(f"Best Val Accuracy: {best_val_acc_p2:.2f}%")

    model.load_state_dict(torch.load("best_model_v2.pth", map_location=device))
    _, _, all_preds, all_labels = validate(model, val_loader, criterion, device)

    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=val_dataset.classes, digits=4))


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, targets in tqdm(loader, desc="Train"):
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
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
