import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from itertools import cycle
import argparse
import os
import sys
from tqdm import tqdm

# Add project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from backend.ml.utils.data_loader import TbiDataset, val_test_transforms, CLASS_NAMES
from backend.ml.models.hybrid_cnn import create_hybrid_cnn
from backend.ml.models.vision_transformer import create_vision_transformer
from backend.ml.models.high_accuracy_hybrid import create_high_accuracy_hybrid

def generate_roc_curve(model_name, weights_path, data_dir, output_path):
    """
    Generates and saves a multi-class ROC curve plot for a given model.
    """
    # 1. Setup device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_classes = len(CLASS_NAMES)
    model_map = {
        'hybrid_cnn': create_hybrid_cnn,
        'vision_transformer': create_vision_transformer,
        'high_accuracy_hybrid': create_high_accuracy_hybrid
    }

    if model_name not in model_map:
        raise ValueError("Unsupported model name. Choose from 'hybrid_cnn', 'vision_transformer', 'high_accuracy_hybrid'.")

    # Load the model structure and weights
    model = model_map[model_name](num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model '{model_name}' loaded successfully from '{weights_path}'.")

    # 2. Prepare DataLoader
    # We use the validation/test transforms for evaluation
    dataset = TbiDataset(data_dir=data_dir, transform=val_test_transforms)
    # Use a small subset for quick testing, or full dataset for final evaluation
    # subset_indices = np.random.choice(len(dataset), 500, replace=False) 
    # subset = torch.utils.data.Subset(dataset, subset_indices)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    print(f"Evaluating on {len(dataset)} images from '{data_dir}'.")

    # 3. Get model predictions (scores) and true labels
    y_true = []
    y_score = []

    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Getting predictions"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            # Use softmax to get probabilities
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            y_score.extend(probabilities.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    
    # Binarize the true labels for multi-class ROC
    y_true_binarized = np.eye(num_classes)[y_true]

    # 4. Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 5. Plot all ROC curves
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(12, 10))
    
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple'])
    for i, color in zip(range(num_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of class {CLASS_NAMES[i]} (AUC = {roc_auc[i]:0.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title(f'Multi-class ROC Curve for {model_name.replace("_", " ").title()}', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"ROC curve plot saved to '{output_path}'")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate ROC curve for a trained TBI model.")
    parser.add_argument('--model_name', type=str, required=True, help="Model to evaluate. Choose from 'hybrid_cnn', 'vision_transformer', 'high_accuracy_hybrid'.")
    parser.add_argument('--weights_path', type=str, required=True, help="Path to the saved model weights (.pth file).")
    parser.add_argument('--data_dir', type=str, required=True, help="Directory containing the validation/test data.")
    parser.add_argument('--output_path', type=str, required=True, help="Path to save the output ROC curve image (e.g., 'roc_cnn.png').")
    
    args = parser.parse_args()
    generate_roc_curve(args.model_name, args.weights_path, args.data_dir, args.output_path)