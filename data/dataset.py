import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import random
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, StandardScaler

class FixedPointDataset(Dataset):    
    def __init__(self, csv_file, poly_degree=1, normalize=True, augment=False, scaler_type="StandardScaler"):
        data = pd.read_csv(csv_file).dropna()
        X = data.iloc[:, :-1].values
        self.y = data.iloc[:, -1].values
        
        if poly_degree > 1:
            poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
            X = poly.fit_transform(X)
        
        if normalize:
            # Choose scaler based on scaler_type
            if scaler_type == "MinMaxScaler":
                scaler = MinMaxScaler(feature_range=(0, 1))
                y_scaler = MinMaxScaler(feature_range=(0, 1))
            else:  # StandardScaler
                scaler = StandardScaler()
                y_scaler = StandardScaler()
            
            X = scaler.fit_transform(X)
            self.y = y_scaler.fit_transform(np.asarray(self.y).reshape(-1, 1)).ravel()
            self.scaler = scaler
            self.y_scaler = y_scaler
        else:
            self.scaler = None
            self.y_scaler = None
            
        # Reshape X for LSTM input [batch, seq_len, features]
        self.X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Data augmentation flag
        self.augment = augment
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        x = self.X[idx].astype(np.float32)
        y = self.y[idx]
        
        # Apply data augmentation if enabled
        if self.augment and random.random() < 0.5:
            # Add small Gaussian noise to input
            noise = np.random.normal(0, 0.01, x.shape).astype(np.float32)
            x = x + noise
            
        return x, y
