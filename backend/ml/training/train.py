# # import torch
# # import torch.nn as nn
# # import torch.optim as optim
# # from torch.utils.data import DataLoader, random_split
# # from tqdm import tqdm
# # import argparse
# # import os
# # import sys

# # # Add the project root to the Python path to allow for absolute imports
# # sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# # from ml.utils.data_loader import TbiDataset, train_transforms, val_test_transforms, BATCH_SIZE
# # from ml.models.hybrid_cnn import create_hybrid_cnn

# # def train(data_dir, model_name, epochs):
# #     """
# #     Main function to train the model.
# #     """
# #     # 1. Setup device (use GPU if available)
# #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #     print(f"Using device: {device}")

# #     # 2. Create Datasets and DataLoaders
# #     # Use training transforms for the main dataset
# #     full_dataset = TbiDataset(data_dir=data_dir, transform=train_transforms)

# #     # Split dataset into training and validation sets (80-20 split)
# #     train_size = int(0.8 * len(full_dataset))
# #     val_size = len(full_dataset) - train_size
# #     train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# #     # Important: Apply validation transforms to the validation set
# #     val_dataset.dataset.transform = val_test_transforms

# #     train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# #     val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
# #     print(f"Training on {len(train_dataset)} images, validating on {len(val_dataset)} images.")

# #     # 3. Initialize Model, Loss Function, and Optimizer
# #     if model_name.lower() == 'cnn':
# #         model = create_hybrid_cnn()
# #     else:
# #         raise ValueError("Unsupported model type provided.")
        
# #     model.to(device)
# #     criterion = nn.CrossEntropyLoss()
# #     optimizer = optim.Adam(model.fc.parameters(), lr=0.001) # Only optimize the classifier head

# #     # 4. Training Loop
# #     for epoch in range(epochs):
# #         # --- Training Phase ---
# #         model.train()
# #         running_loss = 0.0
# #         correct_predictions = 0
# #         total_predictions = 0

# #         for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Training]"):
# #             inputs, labels = inputs.to(device), labels.to(device)

# #             optimizer.zero_grad()
# #             outputs = model(inputs)
# #             loss = criterion(outputs, labels)
# #             loss.backward()
# #             optimizer.step()

# #             running_loss += loss.item() * inputs.size(0)
# #             _, predicted = torch.max(outputs.data, 1)
# #             total_predictions += labels.size(0)
# #             correct_predictions += (predicted == labels).sum().item()

# #         epoch_loss = running_loss / len(train_loader.dataset)
# #         epoch_acc = (correct_predictions / total_predictions) * 100

# #         # --- Validation Phase ---
# #         model.eval()
# #         val_loss = 0.0
# #         correct_predictions_val = 0
# #         total_predictions_val = 0
# #         with torch.no_grad():
# #             for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Validation]"):
# #                 inputs, labels = inputs.to(device), labels.to(device)
# #                 outputs = model(inputs)
# #                 loss = criterion(outputs, labels)
                
# #                 val_loss += loss.item() * inputs.size(0)
# #                 _, predicted = torch.max(outputs.data, 1)
# #                 total_predictions_val += labels.size(0)
# #                 correct_predictions_val += (predicted == labels).sum().item()
        
# #         val_loss /= len(val_loader.dataset)
# #         val_acc = (correct_predictions_val / total_predictions_val) * 100

# #         print(f"Epoch {epoch+1}/{epochs} -> Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

# #     # 5. Save the trained model
# #     weights_dir = os.path.join(os.path.dirname(__file__), '..', 'weights')
# #     os.makedirs(weights_dir, exist_ok=True)
# #     save_path = os.path.join(weights_dir, 'cnn_model.pth')
# #     torch.save(model.state_dict(), save_path)
# #     print(f"Training complete. Model saved to {save_path}")

# # if __name__ == '__main__':
# #     parser = argparse.ArgumentParser(description="Train a model for TBI classification.")
# #     parser.add_argument('--data_dir', type=str, required=True, help="Directory containing the training data.")
# #     parser.add_argument('--model', type=str, default='cnn', help="Model to train ('cnn').")
# #     parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs.")
    
# #     args = parser.parse_args()
# # #     train(args.data_dir, args.model, args.epochs)
# # import torch
# # import torch.nn as nn
# # import torch.optim as optim
# # from torch.utils.data import DataLoader, random_split
# # from tqdm import tqdm
# # import argparse
# # import os
# # import sys

# # # Add the project root to the Python path to allow for absolute imports
# # sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# # from ml.utils.data_loader import TbiDataset, train_transforms, val_test_transforms, BATCH_SIZE
# # from ml.models.hybrid_cnn import create_hybrid_cnn
# # from ml.models.scratch_cnn import create_scratch_cnn
# # from ml.models.scratch_vit import create_scratch_vit
# # from ml.models.hybrid_vit import create_hybrid_vit
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, random_split
# from tqdm import tqdm
# import argparse
# import os

# from ..utils.data_loader import TbiDataset, train_transforms, val_test_transforms, BATCH_SIZE
# from ..models.hybrid_cnn import create_hybrid_cnn
# from ..models.vision_transformer import create_vision_transformer
# from ..models.high_accuracy_hybrid import create_high_accuracy_hybrid

# def train(data_dir, model_name, epochs):
#     """
#     Main function to train the specified model.
#     """
#     # 1. Setup device (use GPU if available)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     # 2. Create Datasets and DataLoaders
#     full_dataset = TbiDataset(data_dir=data_dir, transform=train_transforms)

#     train_size = int(0.8 * len(full_dataset))
#     val_size = len(full_dataset) - train_size
#     train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
#     val_dataset.dataset.transform = val_test_transforms

#     train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
#     val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
#     print(f"Training on {len(train_dataset)} images, validating on {len(val_dataset)} images.")

#     # 3. Initialize Model, Loss Function, and Optimizer
#     criterion = nn.CrossEntropyLoss()

#     # if model_name.lower() == 'cnn':
#     #     print("Loading Hybrid CNN for fine-tuning.")
#     #     model = create_hybrid_cnn()
#     #     for name, child in model.named_children():
#     #         if name in ["layer3", "layer4"]:
#     #             for param in child.parameters():
#     #                 param.requires_grad = True
#     #     for param in model.fc.parameters():
#     #         param.requires_grad = True
#     #     params_to_update = [
#     #         {'params': model.fc.parameters(), 'lr': 0.001},
#     #         {'params': model.layer4.parameters(), 'lr': 0.0001},
#     #         {'params': model.layer3.parameters(), 'lr': 0.0001}
#     #     ]
#     #     optimizer = optim.Adam(params_to_update)

#     # elif model_name.lower() == 'scratch_cnn':
#     #     print("Loading CNN built from scratch.")
#     #     model = create_scratch_cnn()
#     #     optimizer = optim.Adam(model.parameters(), lr=0.001)

#     # elif model_name.lower() == 'scratch_vit':
#     #     print("Loading Vision Transformer built from scratch.")
#     #     model = create_scratch_vit()
#     #     optimizer = optim.Adam(model.parameters(), lr=0.001)

#     # elif model_name.lower() == 'hybrid_vit':
#     #     print("Loading Hybrid (fine-tuning) Vision Transformer.")
#     #     model = create_hybrid_vit()
#     #     optimizer = optim.Adam(model.parameters(), lr=0.0001)

#     # else:
#     #     raise ValueError("Unsupported model type provided. Choose 'cnn', 'scratch_cnn', 'scratch_vit', or 'hybrid_vit'.")

#     # model.to(device)
#     criterion = nn.CrossEntropyLoss()

#     if model_name.lower() == 'hybrid_cnn':
#         print("Loading Hybrid CNN for fine-tuning.")
#         model = create_hybrid_cnn()
#         params_to_update = [
#             {'params': model.fc.parameters(), 'lr': 0.001},
#             {'params': model.layer4.parameters(), 'lr': 0.0001},
#             {'params': model.layer3.parameters(), 'lr': 0.0001}
#         ]
#         optimizer = optim.Adam(params_to_update)

#     elif model_name.lower() == 'vision_transformer':
#         print("Loading Hybrid (fine-tuning) Vision Transformer.")
#         model = create_vision_transformer()
#         optimizer = optim.Adam(model.parameters(), lr=0.0001)

#     elif model_name.lower() == 'high_accuracy_hybrid':
#         print("Loading High-Accuracy Hybrid (EfficientNetV2).")
#         model = create_high_accuracy_hybrid()
#         optimizer = optim.Adam(model.parameters(), lr=0.0001)

#     else:
#         raise ValueError("Unsupported model type provided. Choose 'hybrid_cnn', 'vision_transformer', or 'high_accuracy_hybrid'.")

#     model.to(device)
#     # 4. Training Loop
#     for epoch in range(epochs):
#         # --- Training Phase ---
#         model.train()
#         running_loss = 0.0
#         correct_predictions = 0
#         total_predictions = 0

#         for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Training]"):
#             inputs, labels = inputs.to(device), labels.to(device)

#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item() * inputs.size(0)
#             _, predicted = torch.max(outputs.data, 1)
#             total_predictions += labels.size(0)
#             correct_predictions += (predicted == labels).sum().item()

#         epoch_loss = running_loss / len(train_loader.dataset)
#         epoch_acc = (correct_predictions / total_predictions) * 100

#         # --- Validation Phase ---
#         model.eval()
#         val_loss = 0.0
#         correct_predictions_val = 0
#         total_predictions_val = 0
#         with torch.no_grad():
#             for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Validation]"):
#                 inputs, labels = inputs.to(device), labels.to(device)
#                 outputs = model(inputs)
#                 loss = criterion(outputs, labels)

#                 val_loss += loss.item() * inputs.size(0)
#                 _, predicted = torch.max(outputs.data, 1)
#                 total_predictions_val += labels.size(0)
#                 correct_predictions_val += (predicted == labels).sum().item()

#         val_loss /= len(val_loader.dataset)
#         val_acc = (correct_predictions_val / total_predictions_val) * 100

#         print(f"Epoch {epoch+1}/{epochs} -> Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

#     # 5. Save the trained model
#     weights_dir = os.path.join(os.path.dirname(__file__), '..', 'weights')
#     os.makedirs(weights_dir, exist_ok=True)
#     # Save with a dynamic name based on the model type
#     save_path = os.path.join(weights_dir, f'{model_name}_model.pth')
#     torch.save(model.state_dict(), save_path)
#     print(f"Training complete. Model saved to {save_path}")

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="Train a model for TBI classification.")
#     parser.add_argument('--data_dir', type=str, required=True, help="Directory containing the training data.")
#     parser.add_argument('--model', type=str, default='cnn', help="Model to train ('cnn', 'scratch_cnn', 'scratch_vit', or 'hybrid_vit').")
#     parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs.")

#     args = parser.parse_args()
#     train(args.data_dir, args.model, args.epochs)
# parser.add_argument('--model', type=str, default='hybrid_cnn', help="Model to train ('hybrid_cnn', 'vision_transformer', 'high_accuracy_hybrid').")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import argparse
import os

# Relative imports to correctly find the other modules
from ..utils.data_loader import TbiDataset, train_transforms, val_test_transforms, BATCH_SIZE
from ..models.hybrid_cnn import create_hybrid_cnn
from ..models.vision_transformer import create_vision_transformer
from ..models.high_accuracy_hybrid import create_high_accuracy_hybrid

def train(data_dir, model_name, epochs):
    """
    Main function to train the specified model.
    """
    # 1. Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Create Datasets and DataLoaders
    full_dataset = TbiDataset(data_dir=data_dir, transform=train_transforms)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    val_dataset.dataset.transform = val_test_transforms

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Training on {len(train_dataset)} images, validating on {len(val_dataset)} images.")

    # 3. Initialize Model, Loss Function, and Optimizer
    criterion = nn.CrossEntropyLoss()

    if model_name.lower() == 'hybrid_cnn':
        print("Loading Hybrid CNN for fine-tuning.")
        model = create_hybrid_cnn()
        params_to_update = [
            {'params': model.fc.parameters(), 'lr': 0.001},
            {'params': model.layer4.parameters(), 'lr': 0.0001},
            {'params': model.layer3.parameters(), 'lr': 0.0001}
        ]
        optimizer = optim.Adam(params_to_update)

    elif model_name.lower() == 'vision_transformer':
        print("Loading Hybrid (fine-tuning) Vision Transformer.")
        model = create_vision_transformer()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

    elif model_name.lower() == 'high_accuracy_hybrid':
        print("Loading High-Accuracy Hybrid (EfficientNetV2).")
        model = create_high_accuracy_hybrid()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

    else:
        raise ValueError("Unsupported model type provided. Choose 'hybrid_cnn', 'vision_transformer', or 'high_accuracy_hybrid'.")

    model.to(device)

    # 4. Training Loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Training]"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = (correct_predictions / total_predictions) * 100

        # Validation Phase
        model.eval()
        val_loss = 0.0
        correct_predictions_val = 0
        total_predictions_val = 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Validation]"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_predictions_val += labels.size(0)
                correct_predictions_val += (predicted == labels).sum().item()

        val_loss /= len(val_dataset)
        val_acc = (correct_predictions_val / total_predictions_val) * 100

        print(f"Epoch {epoch+1}/{epochs} -> Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    # 5. Save the trained model
    weights_dir = os.path.join(os.path.dirname(__file__), '..', 'weights')
    os.makedirs(weights_dir, exist_ok=True)
    save_path = os.path.join(weights_dir, f'{model_name}_model.pth')
    torch.save(model.state_dict(), save_path)
    print(f"Training complete. Model saved to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a model for TBI classification.")
    parser.add_argument('--data_dir', type=str, required=True, help="Directory containing the training data.")
    parser.add_argument('--model', type=str, default='hybrid_cnn', help="Model to train ('hybrid_cnn', 'vision_transformer', 'high_accuracy_hybrid').")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs.")

    args = parser.parse_args()
    train(args.data_dir, args.model, args.epochs)