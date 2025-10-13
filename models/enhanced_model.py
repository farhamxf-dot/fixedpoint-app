import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.chaotic_oscillator import ChaoticOscillator
from models.transformer_blocks import TransformerAttentionBlock, ResidualBlock

# New improved CombinedModel with enhancements for better performance
class EnhancedCombinedModel(nn.Module):
    def __init__(self, n_points, k=0.05, s=1.0, lstm_hidden_size=128, lstm_num_layers=2,
                 lstm_dropout=0.3, attention_heads=8, attention_dropout=0.1, conv_channels=[64, 128, 256],
                 kernel_size=3, transformer_dim=256, transformer_heads=8, transformer_blocks=3,
                 dense_hidden_size=128, dense_dropout=0.3, oscillator_units=41, oscillator_count=12):
        super(EnhancedCombinedModel, self).__init__()
        
        # Store parameters
        self.n_points = n_points
        self.dropout_rate = dense_dropout
        
        # LSTM Branch with configurable capacity and bidirectional
        self.lstm1 = nn.LSTM(input_size=1, hidden_size=lstm_hidden_size, 
                            batch_first=True, bidirectional=True, 
                            num_layers=lstm_num_layers, dropout=lstm_dropout if lstm_num_layers > 1 else 0)
        self.dropout1 = nn.Dropout(lstm_dropout)
        
        # Second LSTM layer with reduced size
        lstm2_hidden = lstm_hidden_size // 2
        self.lstm2 = nn.LSTM(input_size=lstm_hidden_size*2, hidden_size=lstm2_hidden, 
                            batch_first=True, bidirectional=True, 
                            num_layers=lstm_num_layers, dropout=lstm_dropout if lstm_num_layers > 1 else 0)
        self.dropout2 = nn.Dropout(lstm_dropout)
        
        # Third LSTM layer for deeper feature extraction
        lstm3_hidden = lstm2_hidden // 2
        self.lstm3 = nn.LSTM(input_size=lstm2_hidden*2, hidden_size=lstm3_hidden, 
                            batch_first=True, bidirectional=True)
        
        # Configurable multi-head attention for LSTM output
        self.lstm_attn = nn.MultiheadAttention(embed_dim=lstm3_hidden*2, 
                                              num_heads=attention_heads, 
                                              dropout=attention_dropout)
        self.lstm_norm = nn.LayerNorm(lstm3_hidden*2)
        
        # Dense layers for LSTM branch
        self.fc1 = nn.Linear(lstm3_hidden*2, dense_hidden_size)
        self.bn1 = nn.BatchNorm1d(dense_hidden_size)
        self.fc2 = nn.Linear(dense_hidden_size, dense_hidden_size)
        self.fc3 = nn.Linear(dense_hidden_size, dense_hidden_size//2)
        
        # Chaotic oscillator array
        self.chaotic_oscillators = nn.ModuleList([
            ChaoticOscillator(units=oscillator_units, k=k, s=s)
            for _ in range(oscillator_count)
        ])
        
        # Hybrid branch processing
        hybrid_size = oscillator_units * oscillator_count
        self.hybrid_fc1 = nn.Linear(hybrid_size, dense_hidden_size)
        self.hybrid_norm1 = nn.LayerNorm(dense_hidden_size)
        self.hybrid_dropout1 = nn.Dropout(dense_dropout)
        self.hybrid_fc2 = nn.Linear(dense_hidden_size, dense_hidden_size//2)
        self.hybrid_norm2 = nn.LayerNorm(dense_hidden_size//2)
        self.hybrid_dropout2 = nn.Dropout(dense_dropout)
        
        # Convolutional branch with configurable channels and kernel size
        conv_layers = []
        in_channels = 1
        for out_channels in conv_channels:
            conv_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(2, padding=0)
            ])
            in_channels = out_channels
        self.conv_branch = nn.Sequential(*conv_layers)
        
        # Transformer blocks with configurable dimension and heads
        self.transformer_blocks = nn.ModuleList([
            TransformerAttentionBlock(transformer_dim, transformer_heads, dense_dropout) 
            for _ in range(transformer_blocks)
        ])
        
        # Progressive path
        conv_output_size = self._get_conv_output_size(torch.zeros(1, 1, n_points))
        self.prog_fc1 = nn.Linear(conv_output_size, dense_hidden_size*2)
        self.prog_bn1 = nn.BatchNorm1d(dense_hidden_size*2)
        self.prog_dropout = nn.Dropout(dense_dropout)
        self.prog_fc2 = nn.Linear(dense_hidden_size*2, dense_hidden_size)
        self.prog_fc3 = nn.Linear(dense_hidden_size, dense_hidden_size//2)
        
        # Residual connection
        self.residual_fc = nn.Linear(n_points, 1)
        
        # Combined processing
        combined_input_size = (dense_hidden_size//2) * 3  # from lstm_fc, hybrid_out, and prog_out
        self.combined_fc1 = nn.Linear(combined_input_size, dense_hidden_size*2)
        self.combined_norm1 = nn.LayerNorm(dense_hidden_size*2)
        self.combined_dropout1 = nn.Dropout(dense_dropout)
        self.combined_fc2 = nn.Linear(dense_hidden_size*2, dense_hidden_size)
        self.combined_norm2 = nn.LayerNorm(dense_hidden_size)
        self.combined_dropout2 = nn.Dropout(dense_dropout)
        self.combined_fc3 = nn.Linear(dense_hidden_size, dense_hidden_size//2)
        
        # Output head
        self.out_main = nn.Linear(dense_hidden_size//2, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better training stability."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def _get_conv_output_size(self, x):
        """Helper function to get the output size of the convolutional branch."""
        return self.conv_branch(x).view(x.size(0), -1).size(1)
    
    def forward(self, x):
        """Forward pass through the network."""
        batch_size = x.size(0)
        
        # Ensure input shape is correct [batch_size, sequence_length, 1]
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)
        
        # LSTM branch - ensure input is [batch, seq_len, 1]
        lstm_input = x
        if lstm_input.size(-1) != 1:
            # If the last dimension is not 1, reshape it
            if lstm_input.size(-1) == 3:  # Common error case from the log
                lstm_input = lstm_input.view(batch_size, -1, 1)
            else:
                # Generic reshape for other cases
                lstm_input = lstm_input.reshape(batch_size, -1, 1)
        
        lstm_out, _ = self.lstm1(lstm_input)
        lstm_out = self.dropout1(lstm_out)
        lstm_out, _ = self.lstm2(lstm_out)
        lstm_out = self.dropout2(lstm_out)
        lstm_out, _ = self.lstm3(lstm_out)
        
        # Apply attention to LSTM output
        lstm_out = lstm_out.permute(1, 0, 2)  # [seq_len, batch, hidden]
        attn_out, _ = self.lstm_attn(lstm_out, lstm_out, lstm_out)
        lstm_out = self.lstm_norm(lstm_out + attn_out)
        lstm_out = lstm_out.permute(1, 0, 2)  # [batch, seq_len, hidden]
        
        # Process LSTM features
        lstm_features = self.fc1(lstm_out[:, -1, :])
        lstm_features = self.bn1(lstm_features)
        lstm_features = torch.relu(lstm_features)
        lstm_features = self.fc2(lstm_features)
        lstm_features = torch.relu(lstm_features)
        lstm_features = self.fc3(lstm_features)
        
        # Chaotic oscillator processing
        # Ensure input to oscillators is [batch_size, n_points]
        osc_input = x.reshape(batch_size, -1)  # Flatten to [batch, seq_len*features]
        if osc_input.shape[1] != self.n_points:
            # If dimensions don't match, try to handle it
            if osc_input.shape[1] > self.n_points:
                # Too many points, take the first n_points
                osc_input = osc_input[:, :self.n_points]
            else:
                # Too few points, pad with zeros
                padding = torch.zeros(batch_size, self.n_points - osc_input.shape[1], device=osc_input.device)
                osc_input = torch.cat([osc_input, padding], dim=1)

        oscillator_outputs = []
        for oscillator in self.chaotic_oscillators:
            osc_out = oscillator(osc_input)
            oscillator_outputs.append(osc_out)
        hybrid_features = torch.cat(oscillator_outputs, dim=1)
        
        # Process hybrid features
        hybrid_features = self.hybrid_fc1(hybrid_features)
        hybrid_features = self.hybrid_norm1(hybrid_features)
        hybrid_features = torch.relu(hybrid_features)
        hybrid_features = self.hybrid_dropout1(hybrid_features)
        hybrid_features = self.hybrid_fc2(hybrid_features)
        hybrid_features = self.hybrid_norm2(hybrid_features)
        hybrid_features = self.hybrid_dropout2(hybrid_features)
        
        # Convolutional branch - ensure input is [batch, channels, seq_len]
        conv_input = x
        if conv_input.size(1) == 1 and conv_input.size(2) > 1:
            # Input is [batch, 1, seq_len] - already in the right format
            pass
        elif conv_input.size(1) > 1 and conv_input.size(2) == 1:
            # Input is [batch, seq_len, 1] - need to transpose
            conv_input = conv_input.transpose(1, 2)
        else:
            # Try to reshape to the expected format
            conv_input = conv_input.reshape(batch_size, 1, -1)
        
        conv_out = self.conv_branch(conv_input)
        conv_features = self.prog_fc1(conv_out.view(batch_size, -1))
        conv_features = self.prog_bn1(conv_features)
        conv_features = torch.relu(conv_features)
        conv_features = self.prog_dropout(conv_features)
        conv_features = self.prog_fc2(conv_features)
        conv_features = torch.relu(conv_features)
        conv_features = self.prog_fc3(conv_features)
        
        # Combine features
        combined = torch.cat([lstm_features, hybrid_features, conv_features], dim=1)
        combined = self.combined_fc1(combined)
        combined = self.combined_norm1(combined)
        combined = torch.relu(combined)
        combined = self.combined_dropout1(combined)
        combined = self.combined_fc2(combined)
        combined = self.combined_norm2(combined)
        combined = torch.relu(combined)
        combined = self.combined_dropout2(combined)
        combined = self.combined_fc3(combined)
        
        # Generate output
        main_output = self.out_main(combined)
        
        # Return only the main output for compatibility with the original model
        return main_output
