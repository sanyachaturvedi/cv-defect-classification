import torch
import torch.nn as nn
from torchvision import models

def get_model(num_classes=5):
    """
    Loads a pretrained ResNet18 model and replaces the final fully connected layer.
    """
    # Load pretrained ResNet18 weights
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    
    # Replace the final fully connected (fc) layer
    # ResNet18's fc layer input features are usually 512
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

if __name__ == "__main__":
    # Create the model
    num_classes = 5
    model = get_model(num_classes=num_classes)
    
    # Print model architecture summary (optional, but requested)
    # print(model) 
    
    # Create a dummy input tensor: (batch_size, channels, height, width)
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Run a forward pass
    output = model(dummy_input)
    
    # Print results
    print(f"Model created for {num_classes} classes.")
    print(f"Dummy input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Check if output dimension matches num_classes
    assert output.shape[1] == num_classes, f"Expected output dim {num_classes}, got {output.shape[1]}"
    print("Success: Model and dummy forward pass verified.")
