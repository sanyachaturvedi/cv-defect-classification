import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Import local modules
from src.dataset import get_test_loader
from src.model import get_model

def main():
    # Configuration
    class_names = ['crack', 'hole', 'normal', 'rust', 'scratch']
    num_classes = len(class_names)
    checkpoint_path = os.path.join("outputs", "best_model.pt")
    plot_path = os.path.join("outputs", "confusion_matrix.png")
    
    # Device Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")
    
    # Load Data Loader
    test_loader = get_test_loader(batch_size=32, num_workers=0)
    if test_loader is None:
        print("Error: Test loader could not be initialized.")
        return

    # Initialize Model
    model = get_model(num_classes=num_classes)
    
    # Load Checkpoint
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Successfully loaded checkpoint: {checkpoint_path}")
    else:
        print(f"Warning: No checkpoint found at {checkpoint_path}. Running evaluation on initialized weights.")

    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    print("Running inference on test set...")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calculation: Overall Accuracy
    overall_accuracy = np.mean(all_preds == all_labels) * 100
    print(f"\nEvaluation Results:")
    print(f"  Overall Accuracy: {overall_accuracy:.2f}%")

    # Calculation: Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    
    # Calculation: Per-class Accuracy
    print("\n  Per-class Accuracy:")
    for i, class_name in enumerate(class_names):
        # Find indices where the true label is i
        idx = np.where(all_labels == i)[0]
        if len(idx) > 0:
            class_acc = np.mean(all_preds[idx] == all_labels[idx]) * 100
            print(f"    {class_name:<10}: {class_acc:.2f}% ({len(idx)} samples)")
        else:
            print(f"    {class_name:<10}: N/A (0 samples)")

    # Plot and Save Confusion Matrix
    print("\nGenerating confusion matrix plot...")
    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title(f"Confusion Matrix (Accuracy: {overall_accuracy:.2f}%)")
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Confusion matrix saved to: {plot_path}")
    print("\nEvaluation complete.")

if __name__ == "__main__":
    main()
