import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# Configuration
TRAIN_DIR = os.path.join("data", "raw", "train")
TEST_DIR = os.path.join("data", "raw", "test")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_train_loader(batch_size=32, num_workers=4, subset_size=None, seed=42):
    """
    Creates a DataLoader for the training dataset with an optional random subset.
    """
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    if not os.path.exists(TRAIN_DIR):
        print(f"Warning: Train directory {TRAIN_DIR} not found.")
        return None

    train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transform)

    if subset_size is not None:
        print(f"Training class mapping: {train_dataset.class_to_idx}")
        print(f"Total training samples before subset: {len(train_dataset)}")

        # Randomly sample subset indices
        num_train = len(train_dataset)
        actual_subset_size = min(subset_size, num_train)
        indices = torch.randperm(num_train, generator=torch.Generator().manual_seed(seed))[:actual_subset_size]
        train_dataset = Subset(train_dataset, indices)
        
        print(f"Total training samples used after subset: {len(train_dataset)}")
    else:
        print(f"Total training samples: {len(train_dataset)}")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader

def get_test_loader(batch_size=32, num_workers=4, subset_size=None, seed=42):
    """
    Creates a DataLoader for the testing dataset with an optional random subset.
    """
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    if not os.path.exists(TEST_DIR):
        print(f"Warning: Test directory {TEST_DIR} not found.")
        return None

    test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=test_transform)

    if subset_size is not None:
        print(f"Testing class mapping: {test_dataset.class_to_idx}")
        print(f"Total testing samples before subset: {len(test_dataset)}")

        # Randomly sample subset indices
        num_test = len(test_dataset)
        actual_subset_size = min(subset_size, num_test)
        indices = torch.randperm(num_test, generator=torch.Generator().manual_seed(seed))[:actual_subset_size]
        test_dataset = Subset(test_dataset, indices)
        
        print(f"Total testing samples used after subset: {len(test_dataset)}")
    else:
        print(f"Total testing samples: {len(test_dataset)}")
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return test_loader

if __name__ == "__main__":
    # Quick sanity check
    train_loader = get_train_loader(subset_size=2000)
    test_loader = get_test_loader(subset_size=500)
