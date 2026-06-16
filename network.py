"""Dual-head ResNet for AlphaZero chess.

Small, CPU-friendly network with:
- Policy head: outputs move probability distribution over 4672 actions
- Value head: outputs scalar in [-1, 1] for expected outcome from current player's perspective
- Random weight initialization (no pretraining)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

from encoding import NUM_PLANES, NUM_ACTIONS


class ResBlock(nn.Module):
    """Residual block with two conv layers, batch norm, and skip connection."""
    
    def __init__(self, num_filters: int):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class AlphaZeroNet(nn.Module):
    """Dual-head ResNet for chess position evaluation.
    
    Architecture:
        Input conv (20 -> num_filters)
        N residual blocks (num_filters -> num_filters)
        Policy head: Conv 1x1 -> FC -> 4672 logits
        Value head: Conv 1x1 -> FC -> 1 (tanh)
    """
    
    def __init__(self, 
                 num_residual_blocks: int = 4,
                 num_filters: int = 64,
                 num_policy_channels: int = 32,
                 num_value_channels: int = 16,
                 value_fc_size: int = 256):
        super().__init__()
        
        self.num_residual_blocks = num_residual_blocks
        self.num_filters = num_filters
        
        # Input convolution
        self.input_conv = nn.Conv2d(NUM_PLANES, num_filters, kernel_size=3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(num_filters)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResBlock(num_filters) for _ in range(num_residual_blocks)
        ])
        
        # Policy head
        self.policy_conv = nn.Conv2d(num_filters, num_policy_channels, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(num_policy_channels)
        self.policy_fc = nn.Linear(num_policy_channels * 8 * 8, NUM_ACTIONS)
        
        # Value head
        self.value_conv = nn.Conv2d(num_filters, num_value_channels, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(num_value_channels)
        self.value_fc1 = nn.Linear(num_value_channels * 8 * 8, value_fc_size)
        self.value_fc2 = nn.Linear(value_fc_size, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize all weights using Kaiming uniform for conv layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch, 20, 8, 8)
        
        Returns:
            policy_logits: (batch, 4672) raw logits for move probabilities
            value: (batch, 1) scalar in [-1, 1]
        """
        # Input convolution
        x = F.relu(self.input_bn(self.input_conv(x)))
        
        # Residual blocks
        for block in self.res_blocks:
            x = block(x)
        
        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy_logits = self.policy_fc(policy)
        
        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy_logits, value
    
    def predict(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        """Predict policy and value for a single board state.
        
        Args:
            state: (20, 8, 8) numpy array
        
        Returns:
            policy: (4672,) numpy array of probabilities
            value: scalar float in [-1, 1]
        """
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(state).unsqueeze(0).float()
            device = next(self.parameters()).device
            x = x.to(device)
            policy_logits, value = self.forward(x)
            policy = F.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
            value = value.item()
        return policy, value
    
    def predict_batch(self, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict policy and value for a batch of board states.
        
        Args:
            states: (batch, 20, 8, 8) numpy array
        
        Returns:
            policies: (batch, 4672) numpy array of probabilities
            values: (batch,) numpy array of scalars
        """
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(states).float()
            device = next(self.parameters()).device
            x = x.to(device)
            policy_logits, value = self.forward(x)
            policies = F.softmax(policy_logits, dim=1).cpu().numpy()
            values = value.squeeze(-1).cpu().numpy()
        return policies, values


def save_checkpoint(model: AlphaZeroNet, optimizer, path: str, 
                    step: int = 0, extra: Optional[dict] = None):
    """Save model checkpoint."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'step': step,
        'num_residual_blocks': model.num_residual_blocks,
        'num_filters': model.num_filters,
    }
    if extra:
        checkpoint.update(extra)
    torch.save(checkpoint, path)


def load_checkpoint(path: str, model: AlphaZeroNet = None, 
                    optimizer=None) -> dict:
    """Load model checkpoint. Returns the checkpoint dict.
    
    If model is provided, loads state dict into it.
    If optimizer is provided, loads optimizer state dict.
    """
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    
    if model is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None and checkpoint.get('optimizer_state_dict') is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


def create_model_from_config(config) -> AlphaZeroNet:
    """Create an AlphaZeroNet from a Config object."""
    return AlphaZeroNet(
        num_residual_blocks=config.network.num_residual_blocks,
        num_filters=config.network.num_filters,
        num_policy_channels=config.network.num_policy_channels,
        num_value_channels=config.network.num_value_channels,
        value_fc_size=config.network.value_fc_size,
    )