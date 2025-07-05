# import torch
# import torch.nn as nn
# from torchvision import models
# from torchvision.models import ResNet34_Weights

# def create_hybrid_cnn(num_classes=6, freeze_base=True):
#     """
#     Creates a hybrid CNN model using a pre-trained ResNet-34 base.

#     Args:
#         num_classes (int): The number of output classes (6 for our TBI types).
#         freeze_base (bool): If True, freezes the weights of the pre-trained base.
    
#     Returns:
#         A PyTorch model.
#     """
#     # 1. Load a pre-trained ResNet-34 model
#     model = models.resnet34(weights=ResNet34_Weights.DEFAULT)

#     # 2. Freeze the parameters of the base model if specified
#     if freeze_base:
#         for param in model.parameters():
#             param.requires_grad = False

#     # 3. Replace the final fully connected layer (the classifier)
#     # Get the number of input features from the original classifier
#     num_ftrs = model.fc.in_features

#     # Create a new classifier head
#     model.fc = nn.Sequential(
#         nn.Linear(num_ftrs, 256),
#         nn.ReLU(),
#         nn.Dropout(0.5),
#         nn.Linear(256, num_classes)
#     )

#     return model

# # Example of how to instantiate the model (for testing purposes)
# if __name__ == '__main__':
#     model = create_hybrid_cnn()
#     # Create a dummy input tensor to test the model
#     dummy_input = torch.randn(1, 3, 224, 224) # (batch_size, channels, height, width)
#     output = model(dummy_input)
#     print("Model created successfully!")
#     print("Output shape:", output.shape) # Should be [1, 6]
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet34_Weights

def create_hybrid_cnn(num_classes=6, freeze_base=True, pretrained=True):
    """
    Creates a hybrid CNN model using a pre-trained ResNet-34 base.

    Args:
        num_classes (int): The number of output classes (6 for our TBI types).
        freeze_base (bool): If True, freezes the weights of the pre-trained base.
    
    Returns:
        A PyTorch model.
    """
    # 1. Load a pre-trained ResNet-34 model with its default weights
    model = models.resnet34(weights=ResNet34_Weights.DEFAULT)

    # 2. Freeze the parameters of the base model to leverage pre-trained features
    if freeze_base:
        for param in model.parameters():
            param.requires_grad = False

    # 3. Replace the final fully connected layer (the classifier)
    # Get the number of input features from the original classifier
    num_ftrs = model.fc.in_features

    # Create a new classifier head tailored to our task
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes)
    )

    return model

# Example of how to instantiate the model (for testing purposes)
if __name__ == '__main__':
    model = create_hybrid_cnn()
    # Create a dummy input tensor to test the model
    dummy_input = torch.randn(1, 3, 224, 224) # (batch_size, channels, height, width)
    output = model(dummy_input)
    print("Model created successfully!")
    print("Output shape:", output.shape) # Should be [1, 6] for our 6 classes