import torch
import torch.nn as nn
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.chaotic_oscillator import ChaoticOscillator

# Original CombinedModel
class CombinedModel(nn.Module):
    def __init__(self, n_points, k=0.05, s=1.0, lstm_hidden_size=128, lstm_num_layers=2,
                 lstm_dropout=0.3, attention_heads=4, attention_dropout=0.1, conv_channels=[32, 64, 128],
                 kernel_size=3, dense_hidden_size=64, dense_dropout=0.3, oscillator_units=41, 
                 oscillator_count=8):
        super(CombinedModel, self).__init__()
        # LSTM Branch
        self.lstm1 = nn.LSTM(input_size=1, hidden_size=lstm_hidden_size, batch_first=True, 
                            bidirectional=True, num_layers=lstm_num_layers)
        self.dropout1 = nn.Dropout(lstm_dropout)
        self.lstm2 = nn.LSTM(input_size=lstm_hidden_size*2, hidden_size=lstm_hidden_size//2, 
                            batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(lstm_dropout)
        self.lstm3 = nn.LSTM(input_size=lstm_hidden_size, hidden_size=lstm_hidden_size//4, 
                            batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(lstm_hidden_size//2, lstm_hidden_size//4)
        self.bn1 = nn.BatchNorm1d(lstm_hidden_size//4)
        self.fc2 = nn.Linear(lstm_hidden_size//4, lstm_hidden_size//8)
        self.fc3 = nn.Linear(lstm_hidden_size//8, lstm_hidden_size//16)

        # Hybrid Branch
        self.n_points = n_points
        self.chaotic_oscillators = nn.ModuleList([
            ChaoticOscillator(units=oscillator_units, k=k, bifurcation_type=i, s=s) 
            for i in range(oscillator_count)
        ])
        self.hybrid_fc1 = nn.Linear(oscillator_units*oscillator_count, dense_hidden_size)
        self.hybrid_dropout1 = nn.Dropout(dense_dropout)
        self.hybrid_fc2 = nn.Linear(dense_hidden_size, dense_hidden_size)
        self.hybrid_dropout2 = nn.Dropout(dense_dropout)

        # Progressive Branch
        self.prog_conv1 = nn.Sequential(
            nn.Conv1d(1, conv_channels[0], kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(conv_channels[0]),
            nn.ReLU(),
            nn.Dropout(dense_dropout)
        )
        self.prog_conv2 = nn.Sequential(
            nn.Conv1d(conv_channels[0], conv_channels[1], kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(conv_channels[1]),
            nn.ReLU(),
            nn.Dropout(dense_dropout)
        )
        self.prog_conv3 = nn.Sequential(
            nn.Conv1d(conv_channels[1], conv_channels[2], kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(conv_channels[2]),
            nn.ReLU(),
            nn.Dropout(dense_dropout)
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=conv_channels[2], num_heads=attention_heads, 
                                             dropout=attention_dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(conv_channels[2])
        
        # Progressive dense layers
        self.prog_fc1 = nn.Linear(conv_channels[2] * n_points, dense_hidden_size*2)
        self.prog_bn1 = nn.BatchNorm1d(dense_hidden_size*2)
        self.prog_fc2 = nn.Linear(dense_hidden_size*2, dense_hidden_size)
        self.prog_dropout = nn.Dropout(dense_dropout)

        # Combined Branch
        combined_input_size = lstm_hidden_size//16 + dense_hidden_size*2  # LSTM + Hybrid + Progressive
        self.combined_fc1 = nn.Linear(combined_input_size, dense_hidden_size)
        self.combined_dropout1 = nn.Dropout(dense_dropout)
        self.combined_fc2 = nn.Linear(dense_hidden_size, dense_hidden_size//2)
        self.combined_dropout2 = nn.Dropout(dense_dropout)
        self.combined_fc3 = nn.Linear(dense_hidden_size//2, dense_hidden_size//4)
        self.out = nn.Linear(dense_hidden_size//4, 1)

        # Residual connection
        self.residual_fc = nn.Linear(n_points, 1)
        
    def forward(self, x):
        # LSTM Branch
        lstm_out, _ = self.lstm1(x)
        lstm_out = self.dropout1(lstm_out)
        lstm_out, _ = self.lstm2(lstm_out)
        lstm_out = self.dropout2(lstm_out)
        lstm_out, _ = self.lstm3(lstm_out)
        lstm_last = lstm_out[:, -1, :]
        lstm_fc = self.fc1(lstm_last)
        lstm_fc = self.bn1(lstm_fc)
        lstm_fc = torch.relu(self.fc2(lstm_fc))
        lstm_fc = torch.relu(self.fc3(lstm_fc))

        # Hybrid Branch
        flat_input = x.view(x.size(0), -1)
        stimulus = flat_input[:, -1].unsqueeze(1)
        hybrid_input = torch.cat([flat_input, stimulus], dim=1)
        oscillator_outputs = [osc(hybrid_input) for osc in self.chaotic_oscillators]
        hybrid_out = torch.cat(oscillator_outputs, dim=1)
        hybrid_out = torch.relu(self.hybrid_fc1(hybrid_out))
        hybrid_out = self.hybrid_dropout1(hybrid_out)
        hybrid_out = torch.relu(self.hybrid_fc2(hybrid_out))
        hybrid_out = self.hybrid_dropout2(hybrid_out)

        # Progressive Branch
        # Reshape input for conv1d
        prog_input = x.transpose(1, 2)  # Shape: [batch, 1, sequence_length]
        
        # Progressive convolution layers with residual connections
        prog_out = self.prog_conv1(prog_input)
        prog_out = self.prog_conv2(prog_out)
        prog_out = self.prog_conv3(prog_out)  # Shape: [batch, 128, sequence_length]
        
        # Self-attention mechanism
        prog_out = prog_out.permute(0, 2, 1)  # Shape: [batch, sequence_length, channels]
        prog_out, _ = self.attention(prog_out, prog_out, prog_out)
        prog_out = self.layer_norm(prog_out)
        prog_out = prog_out.permute(0, 2, 1)  # Shape: [batch, channels, sequence_length]
        
        # Flatten and dense layers
        prog_out = prog_out.reshape(prog_out.size(0), -1)
        prog_out = torch.relu(self.prog_fc1(prog_out))
        prog_out = self.prog_bn1(prog_out)
        prog_out = self.prog_dropout(prog_out)
        prog_out = torch.relu(self.prog_fc2(prog_out))
        
        # Residual connection
        residual = self.residual_fc(x.squeeze(-1))

        # Combined Branch with all three components
        combined = torch.cat([lstm_fc, hybrid_out, prog_out], dim=1)
        combined = torch.relu(self.combined_fc1(combined))
        combined = self.combined_dropout1(combined)
        combined = torch.relu(self.combined_fc2(combined))
        combined = self.combined_dropout2(combined)
        combined = torch.relu(self.combined_fc3(combined))
        
        # Final output with residual connection
        output = self.out(combined) + residual
        return output

# Simple fixed point model as an alternative
class FixedPointModel(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_initialized = False
        
        # Layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        
        # Batch normalization for stability
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
    def initialize_weights(self):
        """Initialize weights using He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
        self.weight_initialized = True
    
    def forward(self, x):
        if not self.weight_initialized:
            self.initialize_weights()
        
        # First layer with batch norm and ReLU
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        
        # Second layer with batch norm and ReLU
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout(x)
        
        # Output layer with sigmoid activation
        x = self.fc3(x)
        # Use sigmoid from math_utils
        from utils.math_utils import sigmoid
        x = sigmoid(x)
        
        return x
