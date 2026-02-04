import torch
import numpy as np
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torch.utils.data import DataLoader, WeightedRandomSampler
from collections import Counter
from tqdm import tqdm

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    augment_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    class CustomDataset(datasets.ImageFolder):
        def __init__(self, root):
            super().__init__(root, transform=None)
            self.base_transform = base_transform
            self.augment_transform = augment_transform
            labels = [label for _, label in self.samples]
            counts = Counter(labels)
            mean_count = np.mean(list(counts.values()))
            self.minor_classes = [c for c, n in counts.items() if n < mean_count]

        def __getitem__(self, index):
            path, label = self.samples[index]
            image = self.loader(path)
            if label in self.minor_classes:
                image = self.augment_transform(image)
            else:
                image = self.base_transform(image)
            return image, label

    train_dataset = CustomDataset("RiceLeafs/train")

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
        num_workers=0,
        pin_memory=False
    )

    weights = MobileNet_V2_Weights.DEFAULT
    model = mobilenet_v2(weights=weights)
    model.classifier[1] = torch.nn.Linear(model.last_channel, num_classes)
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    epochs = 5

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]")

        for images, targets in loop:
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            loop.set_postfix(
                loss=running_loss / (loop.n + 1),
                acc=100.0 * correct / total
            )

        print(f"Epoch {epoch+1} | Loss: {running_loss/len(train_loader):.4f} | Acc: {100.0*correct/total:.2f}%")

if __name__ == "__main__":
    main()
