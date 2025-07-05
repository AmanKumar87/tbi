import torch.nn as nn
import timm

def create_high_accuracy_hybrid(num_classes=6, pretrained=True):
    """
    Creates a high-accuracy hybrid model using a pre-trained EfficientNetV2-M base.
    """
    # Load a pre-trained EfficientNetV2 model
    # tf_efficientnetv2_m' is a strong, accurate model
    model = timm.create_model('tf_efficientnetv2_m', pretrained=pretrained, num_classes=num_classes)
    
    print("High-accuracy EfficientNetV2 model created successfully.")
    print("Classifier head:", model.classifier)
    
    return model