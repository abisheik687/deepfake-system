
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import argparse
from tqdm import tqdm
from backend.models.temporal import TemporalDeepfakeDetector

# KAVACH-AI Day 5: Temporal Model Training
# Trains LSTM on sequences of features

class DummySequenceDataset(Dataset):
    """
    Simulates sequences of features extracted from video frames.
    Real implementation would load pre-extracted .npy feature files.
    """
    def __init__(self, count=100, seq_len=30, feature_dim=1280):
        self.count = count
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        
        # Generate dummy data
        # Real videos: High variance in features
        # Fake videos: Temporal inconsistencies (simulated randomness)
        self.data = torch.randn(count, seq_len, feature_dim) 
        self.labels = torch.randint(0, 2, (count,)).float()

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def train_temporal_model(epochs=5, batch_size=16, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device} (Simulated Data)")

    # Dataset
    train_dataset = DummySequenceDataset(count=200)
    val_dataset = DummySequenceDataset(count=50)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Model
    model = TemporalDeepfakeDetector(input_dim=1280).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f"Train Loss: {running_loss/len(train_loader):.4f}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        print(f"Val Loss: {val_loss/len(val_loader):.4f}, Acc: {100*correct/total:.2f}%")
        
    # Save Model
    os.makedirs("models/checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "models/checkpoints/day5_temporal_model.pth")
    print("Model saved to models/checkpoints/day5_temporal_model.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    
    train_temporal_model(epochs=args.epochs)
