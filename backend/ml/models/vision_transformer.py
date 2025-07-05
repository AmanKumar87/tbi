import torch.nn as nn
import timm

def create_vision_transformer(num_classes=6, pretrained=True):
    """
    Creates a hybrid Vision Transformer model using a pre-trained base from timm.
    """
    # Load a pre-trained ViT model
    model = timm.create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=num_classes)
    
    print("Vision Transformer model created successfully.")
    print("Classifier head:", model.head)
    
    return model