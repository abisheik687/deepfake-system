
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import glob
import argparse
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from backend.features.audio_extraction import AudioFeatureExtractor

# KAVACH-AI Day 3: Audio Classifier Training
# Binary Classification: Real vs Fake Audio

class AudioDataset(Dataset):
    def __init__(self, file_paths, labels, extractor):
        self.file_paths = file_paths
        self.labels = labels
        self.extractor = extractor

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        audio_path = self.file_paths[idx]
        features = self.extractor.get_features(audio_path)
        label = self.labels[idx]
        return features, label

class SimpleAudioCNN(nn.Module):
    """
    Simple 2D CNN for Spectrogram classification.
    Input: (Batch, 1, 128, Time)
    """
    def __init__(self):
        super(SimpleAudioCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Adaptive pool to handle variable time lengths if needed
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def train_audio_model(data_dir, epochs=5, batch_size=32, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # Gather data
    real_paths = glob.glob(os.path.join(data_dir, "real", "*.wav"))
    fake_paths = glob.glob(os.path.join(data_dir, "fake", "*.wav"))
    
    if not real_paths or not fake_paths:
        print("Dataset not found! Please verify audio extraction.")
        # Create dummy data for testing
        print("Creating dummy data for verification...")
        os.makedirs(os.path.join(data_dir, "real"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "fake"), exist_ok=True)
        # We need actual .wav files for librosa, dummy empty files won't work well
        # Skipping dummy creation for audio as it's more complex than images
        print("Skipping dummy creation. Ensure .wav files exist in data/processed/audio/real and /fake")
        return

    file_paths = real_paths + fake_paths
    labels = [0] * len(real_paths) + [1] * len(fake_paths)

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        file_paths, labels, test_size=0.2, random_state=42
    )

    extractor = AudioFeatureExtractor()
    
    train_dataset = AudioDataset(train_paths, train_labels, extractor=extractor)
    val_dataset = AudioDataset(val_paths, val_labels, extractor=extractor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = SimpleAudioCNN().to(device)
    
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
            
        train_acc = 100 * correct / (total + 1e-8)
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
        
        val_acc = 100 * val_correct / (val_total + 1e-8)
        print(f"Val Acc: {val_acc:.2f}%")
        
    # Save Model
    os.makedirs("models/checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "models/checkpoints/day3_audio_model.pth")
    print("Model saved to models/checkpoints/day3_audio_model.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/processed/audio", help="Path to audio files")
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    
    train_audio_model(args.data_dir, epochs=args.epochs)
