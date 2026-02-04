import torch
import numpy as np
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter
from tqdm import tqdm

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_base_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_aug_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    class CustomTrainDataset(datasets.ImageFolder):
        def __init__(self, root):
            super().__init__(root, transform=None)
            self.base_transform = train_base_transform
            self.aug_transform = train_aug_transform
            labels = [label for _, label in self.samples]
            counts = Counter(labels)
            mean_count = np.mean(list(counts.values()))
            self.minor_classes = [c for c, n in counts.items() if n < mean_count]

        def __getitem__(self, index):
            path, label = self.samples[index]
            image = self.loader(path)
            if label in self.minor_classes:
                image = self.aug_transform(image)
            else:
                image = self.base_transform(image)
            return image, label

    train_dataset = CustomTrainDataset("RiceLeafs/train")
    val_dataset = datasets.ImageFolder("RiceLeafs/validation", transform=val_transform)

    labels = [label for _, label in train_dataset.samples]
    class_counts = np.bincount(labels)
    num_classes = len(class_counts)

    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

    sample_weights = [class_weights[label] for label in labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        sampler=sampler,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

    weights = MobileNet_V2_Weights.DEFAULT
    model = mobilenet_v2(weights=weights)
    model.classifier[1] = torch.nn.Linear(model.last_channel, num_classes)
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    epochs = 5
    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        train_correct = 0
        train_total = 0
        train_loss = 0.0

        train_loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}] Train")
        for images, targets in train_loop:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_total += targets.size(0)
            train_correct += (preds == targets).sum().item()

            train_loop.set_postfix(
                loss=train_loss / (train_loop.n + 1),
                acc=100.0 * train_correct / train_total
            )

        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0

        with torch.no_grad():
            val_loop = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{epochs}] Val")
            for images, targets in val_loop:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)

                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_total += targets.size(0)
                val_correct += (preds == targets).sum().item()

                val_loop.set_postfix(
                    loss=val_loss / (val_loop.n + 1),
                    acc=100.0 * val_correct / val_total
                )

        val_acc = 100.0 * val_correct / val_total
        print(f"Epoch {epoch+1} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print("Best model updated and saved")

    print("\nTraining finished. Best Validation Accuracy:", best_val_acc)

    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\n=== CONFUSION MATRIX ===")
    print(confusion_matrix(all_labels, all_preds))

    print("\n=== PRECISION, RECALL, F1 ===")
    print(classification_report(
        all_labels,
        all_preds,
        target_names=val_dataset.classes,
        digits=4
    ))

if __name__ == "__main__":
    main()
