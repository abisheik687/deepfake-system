
import torch
import torch.nn as nn

class TemporalDeepfakeDetector(nn.Module):
    """
    KAVACH-AI Day 5: Temporal Deepfake Detector
    Uses an LSTM/GRU to analyze sequences of frame features.
    Input: Sequence of feature vectors (e.g., from EfficientNet/ResNet).
    Shape: (Batch, Sequence_Length, Feature_Dim)
    """
    def __init__(self, input_dim=1280, hidden_dim=256, num_layers=2, num_classes=1):
        super(TemporalDeepfakeDetector, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM Layer
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            dropout=0.5 if num_layers > 1 else 0
        )
        
        # Fully Connected Layer
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch, seq_len, hidden_size)
        
        # Decode the hidden state of the last time step
        out = out[:, -1, :]
        
        out = self.fc(out)
        return out
