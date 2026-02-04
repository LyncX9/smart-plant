import torch
import numpy as np
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter
from tqdm import tqdm

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor()
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_dataset = datasets.ImageFolder("RiceLeafs/train", transform=train_transform)
    val_dataset = datasets.ImageFolder("RiceLeafs/validation", transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    labels = [label for _, label in train_dataset.samples]
    class_counts = Counter(labels)
    num_classes = len(class_counts)

    class_weights = torch.tensor(
        [1.0 / class_counts[i] for i in range(num_classes)],
        dtype=torch.float
    ).to(device)

    weights = MobileNet_V2_Weights.DEFAULT
    model = mobilenet_v2(weights=weights)

    for param in model.features.parameters():
        param.requires_grad = False

    model.classifier[1] = torch.nn.Sequential(
        torch.nn.Dropout(0.5),
        torch.nn.Linear(model.last_channel, num_classes)
    )

    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
        weight_decay=1e-4
    )

    epochs = 10
    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        train_correct = 0
        train_total = 0

        train_loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}] Train")
        for images, targets in train_loop:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            train_total += targets.size(0)
            train_correct += (preds == targets).sum().item()

            train_loop.set_postfix(acc=100.0 * train_correct / train_total)

        model.eval()
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            val_loop = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{epochs}] Val")
            for images, targets in val_loop:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)

                val_total += targets.size(0)
                val_correct += (preds == targets).sum().item()

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(targets.cpu().numpy())

        val_acc = 100.0 * val_correct / val_total
        print(f"Epoch {epoch+1} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model_fixed.pth")
            print("Best model updated")

    print("\n=== FINAL EVALUATION ===")
    print(confusion_matrix(all_labels, all_preds))
    print(classification_report(
        all_labels,
        all_preds,
        target_names=val_dataset.classes,
        digits=4
    ))

if __name__ == "__main__":
    main()
