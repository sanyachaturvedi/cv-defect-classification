import os
import csv
import torch
from torch.utils.data import Subset
from src.dataset import get_test_loader
from src.model import get_model

def main():
    # Configuration
    checkpoint_path = os.path.join("outputs", "best_model.pt")
    output_csv = os.path.join("outputs", "predictions.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Data Loader
    # Using batch_size=1 or larger; shuffle MUST be False for path alignment
    test_loader = get_test_loader(batch_size=32, num_workers=0)
    if test_loader is None:
        print("Error: Test loader could not be initialized.")
        return

    # Extract dataset and class information
    dataset = test_loader.dataset
    
    # Handle potentially wrapped Subset
    if isinstance(dataset, Subset):
        base_dataset = dataset.dataset
        indices = dataset.indices
        # Get paths only for the subset samples in the correct order
        image_paths = [base_dataset.samples[i][0] for i in indices]
        class_to_idx = base_dataset.class_to_idx
    else:
        image_paths = [s[0] for s in dataset.samples]
        class_to_idx = dataset.class_to_idx

    # Invert class_to_idx mapping
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)

    # Initialize and Load Model
    model = get_model(num_classes=num_classes)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"Warning: No checkpoint found at {checkpoint_path}. Using random weights.")

    model.to(device)
    model.eval()

    all_predictions = []

    print(f"Running inference on {len(image_paths)} images...")
    
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            # Map indices to class names
            batch_predictions = [idx_to_class[p.item()] for p in preds]
            all_predictions.extend(batch_predictions)

    # Save results to CSV
    os.makedirs("outputs", exist_ok=True)
    with open(output_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'predicted_label'])
        
        for full_path, label in zip(image_paths, all_predictions):
            # Generate relative path starting from 'test/' (stripping data/raw/)
            # Expected format: test/class_name/image.png
            rel_path = os.path.relpath(full_path, start=os.path.join("data", "raw"))
            rel_path = rel_path.replace("\\", "/") # Ensure forward slashes for portability
            
            writer.writerow([rel_path, label])

    print(f"Predictions saved to: {output_csv}")
    print("Inference complete.")

if __name__ == "__main__":
    main()
