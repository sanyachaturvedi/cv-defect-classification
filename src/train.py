import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Import local modules
from src.dataset import get_train_loader, get_test_loader
from src.model import get_model

def main(args):
    # Print the chosen configuration
    print("\n" + "="*45)
    print("Industrial Defect Classification - Configuration:")
    for arg, value in vars(args).items():
        print(f"  {arg:<20}: {value}")
    print("="*45 + "\n")

    # Device Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # Save path
    save_path = os.path.join("outputs", "best_model.pt")
    os.makedirs("outputs", exist_ok=True)
    
    # Data Loaders (Windows safety: num_workers=0)
    train_loader = get_train_loader(
        batch_size=args.batch_size, 
        num_workers=0, 
        subset_size=args.train_subset_size, 
        seed=args.seed
    )
    test_loader = get_test_loader(
        batch_size=args.batch_size, 
        num_workers=0, 
        subset_size=args.test_subset_size, 
        seed=args.seed
    )
    
    if train_loader is None or test_loader is None:
        print("Error: Could not initialize data loaders. Please ensure dataset is correctly placed.")
        return

    # Model, Loss, and Optimizer
    # Note: num_classes is hardcoded to 5 based on project requirements
    model = get_model(num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    best_acc = 0.0
    
    print("\nStarting Training Pipeline...\n")
    
    for epoch in range(args.epochs):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            train_bar.set_postfix(loss=loss.item())
            
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # --- Evaluation Phase ---
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        test_acc = 100 * correct / total
        
        # --- Logging and Checkpointing ---
        print(f"Epoch [{epoch+1}/{args.epochs}] - Loss: {epoch_loss:.4f} - Test Acc: {test_acc:.2f}%")
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), save_path)
            print(f"  --> Saved best model! (Acc: {best_acc:.2f}%)")
        print("-" * 30)

    print(f"\nTraining Complete. Best Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Classification Training Pipeline")
    
    # Training Parameters
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs (default: 5)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    
    # Dataset Parameters
    parser.add_argument("--train-subset-size", type=int, default=None, help="Optional subset size for training (for debugging)")
    parser.add_argument("--test-subset-size", type=int, default=None, help="Optional subset size for testing (for debugging)")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducible subset sampling (default: 42)")
    
    args = parser.parse_args()
    main(args)
