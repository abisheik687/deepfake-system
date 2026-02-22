
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
import glob
import argparse
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# KAVACH-AI Day 2: Simple Classifier Training
# Binary Classification: Real vs Fake

class DeepfakeDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def train_model(data_dir, epochs=5, batch_size=32, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # Gather data
    real_paths = glob.glob(os.path.join(data_dir, "real", "*.jpg"))
    fake_paths = glob.glob(os.path.join(data_dir, "fake", "*.jpg"))
    
    if not real_paths or not fake_paths:
        print("Dataset not found! Please run extract_faces.py first.")
        # Create dummy data for testing the script flow if no data exists
        print("Creating dummy data for verification...")
        os.makedirs(os.path.join(data_dir, "real"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "fake"), exist_ok=True)
        # Create 10 dummy images
        for i in range(10):
            Image.new('RGB', (224, 224), color='green').save(os.path.join(data_dir, "real", f"dummy_{i}.jpg"))
            Image.new('RGB', (224, 224), color='red').save(os.path.join(data_dir, "fake", f"dummy_{i}.jpg"))
        real_paths = glob.glob(os.path.join(data_dir, "real", "*.jpg"))
        fake_paths = glob.glob(os.path.join(data_dir, "fake", "*.jpg"))

    file_paths = real_paths + fake_paths
    labels = [0] * len(real_paths) + [1] * len(fake_paths) # 0=Real, 1=Fake

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        file_paths, labels, test_size=0.2, random_state=42
    )

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = DeepfakeDataset(train_paths, train_labels, transform=transform)
    val_dataset = DeepfakeDataset(val_paths, val_labels, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Model: EfficientNet-B0 (Pretrained)
    print("Loading EfficientNet-B0...")
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    
    # Modify last layer for binary classification
    # EfficientNet classifier is 'classifier' -> sequence -> dropout -> linear
    try:
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, 1)
    except:
        # Fallback if structure differs
        num_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_features, 1)

    model = model.to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.float().to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_acc = 100 * correct / total
        print(f"Train Loss: {running_loss/len(train_loader):.4f}, Acc: {train_acc:.2f}%")
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.float().to(device).unsqueeze(1)
                outputs = model(inputs)
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        print(f"Val Acc: {val_acc:.2f}%")
        
    # Save Model
    os.makedirs("models/checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "models/checkpoints/day2_model.pth")
    print("Model saved to models/checkpoints/day2_model.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/processed/faces", help="Path to face crops")
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    
    train_model(args.data_dir, epochs=args.epochs)
