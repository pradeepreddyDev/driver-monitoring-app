import torch
import torchvision.models as models
import os

# Create the directory for saving models if it doesn't exist
os.makedirs("./models/behavior_detection", exist_ok=True)

# Load the pre-trained ResNet18 model from torchvision
resnet18 = models.resnet18(pretrained=True)

# Save the model's state dictionary (weights only)
torch.save(resnet18.state_dict(), "./models/behavior_detection/resnet18.pth")
