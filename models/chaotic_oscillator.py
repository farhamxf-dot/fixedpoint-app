import torch
import torch.nn as nn
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.math_utils import sigmoid_da, OVERRIDE_BIFURCATION_PARAMS

class ChaoticOscillator(nn.Module):
    def __init__(self, units=1, k=0.05, bifurcation_type=0, s=1.0):
        super(ChaoticOscillator, self).__init__()
        self.units = units
        self.k = k
        self.bifurcation_type = bifurcation_type
        self.s = s
        self.weight_initialized = False

    def initialize_weights(self, signals_dim, device):
        self.signals_dim = signals_dim
        self.w_e = nn.Parameter(torch.randn(signals_dim, self.units, device=device))
        self.w_i = nn.Parameter(torch.randn(signals_dim, self.units, device=device))
        self.xi_e = nn.Parameter(torch.zeros(self.units, device=device))
        self.xi_i = nn.Parameter(torch.zeros(self.units, device=device))
        global OVERRIDE_BIFURCATION_PARAMS
        if OVERRIDE_BIFURCATION_PARAMS is not None:
            self.a1, self.a2, self.a3, self.b1, self.b2, self.b3 = OVERRIDE_BIFURCATION_PARAMS
        else:
            params = self.get_bifurcation_params(self.bifurcation_type)
            self.a1, self.a2, self.a3, self.b1, self.b2, self.b3 = params
        self.weight_initialized = True

    def get_bifurcation_params(self, bifurcation_type):
        settings = {
            0: (0.00, 5.00, 5.00, 0.00, -1.00, 1.00),
            1: (-0.50, 0.55, 0.55, -0.50, -0.55, -0.55),
            2: (-0.50, 0.55, 0.55, 0.50, -0.55, -0.55),
            3: (0.50, 0.55, 0.55, 0.50, -0.55, -0.55),
            4: (0.90, 0.90, 0.90, -0.90, -0.90, -0.90),
            5: (0.90, 0.90, 0.90, -0.90, -0.90, -0.90),
            6: (5.00, 5.00, 5.00, -1.00, -1.00, -1.00),
            7: (5.00, 5.00, 5.00, -1.00, -1.00, -1.00),
        }
        return settings.get(bifurcation_type, (0.00, 5.00, 5.00, 0.00, -1.00, 1.00))

    def forward(self, x):
        if not self.weight_initialized:
            self.initialize_weights(x.size(1)-1, x.device)
        signals = x[:, :-1]
        stimulus = x[:, -1].unsqueeze(1)
        e_input = self.a1 * torch.matmul(signals, self.w_e) + self.a2 * self.xi_e - self.a3 * torch.matmul(signals, self.w_i) + stimulus
        i_input = self.b1 * torch.matmul(signals, self.w_i) - self.b2 * self.xi_i - self.b3 * torch.matmul(signals, self.w_e) + stimulus
        E = sigmoid_da(e_input, self.s)
        I = sigmoid_da(i_input, self.s)
        Omega = sigmoid_da(stimulus, self.s)
        return (E - I) * torch.exp(-self.k * (stimulus ** 2)) + Omega
