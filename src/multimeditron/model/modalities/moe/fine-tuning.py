from transformers import PreTrainedModel, PretrainedConfig
from torchvision import models, transforms, datasets
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel
from gating import GatingNetwork, GatingNetworkConfig

import os
import shutil
import json

def prepare_data(source_dir, target_dir, modality):
    with open(source_dir, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line_data = json.loads(line.strip())  # Remove whitespace and parse JSON
            jsonl_image_path = line_data['modalities'][0]['value']
            image_path = os.path.join('/mloscratch/users/tagemoua/XR-glob', jsonl_image_path)
            #jsonl_image_path = jsonl_image_path.split('/')
            
            target_path = os.path.join(target_dir, modality)
            # Extracting the study ID (assuming it's part of the image path or metadata)
            study_id = jsonl_image_path[-2] # Replace with actual field if needed
            
            # Create a target directory based on modality
            target_path = os.path.join(target_dir, modality)
            
            # Get the base image filename and create a new filename
            base_name = os.path.basename(image_path)
            new_image_name = f"{study_id}_{base_name}"
            new_image_path = os.path.join(target_path, new_image_name)
            os.makedirs(target_path, exist_ok=True)
            

            print(f"Copying {image_path} to {new_image_path}")
            try:
                shutil.copy(image_path, new_image_path)
                print("Image copied successfully")
            except Exception as e:
                print(f"An error occurred: {e}")
                if image_path.lower().endswith(('.png')):
                    try:
                        new_image_path = image_path.replace('.jjpg')
                        shutil.copy(new_image_path, target_path)
                        print("Image copied successfully")
                    except Exception as e:
                        print(f"An error occurred: {e}")

# Function to create a subset of image files

def create_subset(input_dirs, output_dir, subset_size=100):
    """
    Create a subset of image files from multiple directories.

    :param input_dirs: List of input directories containing image files.
    :param output_dir: Target directory to save the subset.
    :param subset_size: Number of images to take from each input directory.
    """
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

    for input_dir in input_dirs:
        if not os.path.exists(input_dir):
            print(f"Directory {input_dir} does not exist. Skipping.")
            continue
        
        # List all files in the current directory
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        
        # Limit to the subset size
        subset_files = files[:subset_size]

        # Create a subfolder in the output directory for this class (same name as input folder)
        class_name = os.path.basename(input_dir)
        class_output_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_output_dir, exist_ok=True)

        # Copy the subset files to the output directory
        for file in subset_files:
            src_path = os.path.join(input_dir, file)
            dst_path = os.path.join(class_output_dir, file)
            shutil.copyfile(src_path, dst_path)

        print(f"Copied {len(subset_files)} files from {input_dir} to {class_output_dir}")
    

if __name__ == "__main__":
    # prepare x-ray data
    #prepare_data("/mloscratch/users/tagemoua/XR-glob/XR-glob.jsonl", "/mloscratch/users/tagemoua/MultiMeditron/processed_data", "X-ray")
    # Create a subset of the dataset for equal representation across classes
    input_dirs = ['/mloscratch/users/tagemoua/MultiMeditron/processed_data/Mri', 
                  '/mloscratch/users/tagemoua/MultiMeditron/processed_data/Ct', 
                  '/mloscratch/users/tagemoua/MultiMeditron/processed_data/General', 
                  '/mloscratch/users/tagemoua/MultiMeditron/processed_data/X-ray', 
                  '/mloscratch/users/tagemoua/MultiMeditron/processed_data/Ultrasound']
    output_dir = '/mloscratch/users/tagemoua/MultiMeditron/processed_data_subset_new'
    #create_subset(input_dirs, output_dir, subset_size=3000)
    new_data = '/mloscratch/users/tagemoua/MultiMeditron/processed_data_subset_new'
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset preparation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    data = datasets.ImageFolder(root=new_data, transform=transform)

    # Split dataset into training and validation sets
    train_size = int(0.8 * len(data))
    val_size = len(data) - train_size
    train_data, val_data = torch.utils.data.random_split(data, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # Load model and move to device
    model = GatingNetwork.from_pretrained("gating/checkpoint", num_classes=5)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Training the model with early stopping based on loss and accuracy
    num_epochs = 20
    best_accuracy = 0.0
    prev_loss = float('inf')  # Initialize with a high value

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Forward pass
            _, _, outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate accuracy for this batch
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Compute epoch loss and accuracy
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

        # Early stopping condition
        if epoch > 0 and epoch_loss > prev_loss and epoch_accuracy <= best_accuracy and best_accuracy > 90:
            print(f"Stopping early at epoch {epoch+1} due to no improvement in accuracy and an increase in loss.")
            break

        # Update best accuracy and previous loss for comparison
        best_accuracy = max(best_accuracy, epoch_accuracy)
        prev_loss = epoch_loss

        # Validation step
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                _, _, outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = 100 * correct / total
        print(f"Validation Loss: {val_loss/len(val_loader):.4f}, Accuracy: {val_accuracy:.2f}%")

    # Save the trained model
    model.save_pretrained("gating/checkpoint")
