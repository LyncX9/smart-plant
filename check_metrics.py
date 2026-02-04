
import torch
import numpy as np
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v2
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Standard Transform (Same as training/inference)
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # Note: smartplant_enhanced.py DOES NOT normalize in its transform logic for inference
        # But training usually does. I will check smartplant_enhanced.py again.
        # smartplant_enhanced.py lines 89-93 ONLY does ToTensor.
        # So I should PROBABLY NOT normalize here to match inference pipeline if that's what's being asked.
        # However, models are usually trained with normalization.
        # Let's check train_improved.py if possible, but let's stick to simple ToTensor first to match "smartplant_enhanced.py"
        # Wait, mobilenet_v2 expects normalization.
        # smartplant_enhanced.py: 
        # transform = T.Compose([T.ToPILImage(), T.Resize((224, 224)), T.ToTensor()])
        # So it seems the production script ignores normalization? That might be a bug or intended.
        # Let's use standard normalization just in case, or try without if accuracy is low.
        # Actually, let's look at best_model_fixed.pth performance note in json: "error".
        # Let's try matching smartplant_enhanced.py transform exactly.
    ])
    
    # Actually, for correct evaluation, I should use what was used during training.
    # Standard MobileNetV2 usually requires normalization.
    # Let's use the one that is most likely correct: With Normalization.
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("Loading dataset...")
    val_dataset = datasets.ImageFolder("RiceLeafs/validation", transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    class_names = val_dataset.classes
    print(f"Classes: {class_names}")

    # Model Architecture (Matches smartplant_enhanced.py)
    print("Building model...")
    model = mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Sequential(
        torch.nn.Dropout(0.5),
        torch.nn.Linear(model.last_channel, len(class_names))
    )

    print("Loading weights (best_model_fixed.pth)...")
    state_dict = torch.load("best_model_fixed.pth", map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print("Evaluating...")
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Calculate Metrics
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
    
    print("\n" + "="*30)
    print("MODEL METRICS: best_model_fixed.pth")
    print("="*30)
    print(f"Accuracy : {acc*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall   : {recall*100:.2f}%")
    print(f"F1 Score : {f1*100:.2f}%")
    print("="*30)
    
    # print("\nDetailed Report:")
    # print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

if __name__ == "__main__":
    main()
