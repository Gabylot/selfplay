"""Training loop for AlphaZero chess engine.

Samples batches from the replay buffer and trains the dual-head network
with separate policy loss (cross-entropy) and value loss (MSE).

Both losses are logged separately to the stats database.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional

from network import AlphaZeroNet, save_checkpoint
from selfplay import ReplayBuffer
from encoding import NUM_ACTIONS


def policy_loss_fn(policy_logits: torch.Tensor, target_policies: torch.Tensor,
                   legal_move_masks: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Compute policy loss using cross-entropy.
    
    Args:
        policy_logits: (batch, 4672) raw logits from network
        target_policies: (batch, 4672) target distribution from MCTS visits
        legal_move_masks: (batch, 4672) binary masks, 1.0 for legal moves
    
    Returns:
        Scalar policy loss
    """
    if legal_move_masks is not None:
        # Mask out illegal moves: set logits for illegal moves to a very negative value
        policy_logits = policy_logits + (1 - legal_move_masks) * (-1e9)
    
    # Cross-entropy with target distribution
    # F.cross_entropy expects log-softmax input and class indices,
    # but with soft targets we need KL divergence
    log_probs = F.log_softmax(policy_logits, dim=1)
    loss = -(target_policies * log_probs).sum(dim=1).mean()
    
    return loss


def value_loss_fn(value_pred: torch.Tensor, target_values: torch.Tensor) -> torch.Tensor:
    """Compute value loss using MSE.
    
    Args:
        value_pred: (batch, 1) predicted values from network
        target_values: (batch,) target values (game outcomes)
    
    Returns:
        Scalar value loss
    """
    target_values = target_values.unsqueeze(1)  # (batch, 1)
    return F.mse_loss(value_pred, target_values)


def train_one_step(model: AlphaZeroNet, optimizer: torch.optim.Optimizer,
                   replay_buffer: ReplayBuffer, batch_size: int, 
                   device: torch.device = None) -> dict:
    """Perform a single training step.
    
    Args:
        model: The neural network
        optimizer: Optimizer
        replay_buffer: Replay buffer to sample from
        batch_size: Batch size
        device: Torch device
    
    Returns:
        Dict with 'policy_loss', 'value_loss', 'total_loss' (all float scalars)
    """
    model.train()
    
    if device is None:
        device = next(model.parameters()).device
    
    # Sample batch from replay buffer
    states, policies, values = replay_buffer.sample_batch(batch_size)
    states = torch.from_numpy(states).float().to(device)
    policies = torch.from_numpy(policies).float().to(device)
    values = torch.from_numpy(values).float().to(device)
    
    # Forward pass
    policy_logits, value_pred = model(states)
    
    # Compute losses
    p_loss = policy_loss_fn(policy_logits, policies)
    v_loss = value_loss_fn(value_pred, values)
    total_loss = p_loss + v_loss
    
    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    return {
        'policy_loss': float(p_loss.item()),
        'value_loss': float(v_loss.item()),
        'total_loss': float(total_loss.item()),
    }


def create_optimizer(model: AlphaZeroNet, lr: float = 0.001, 
                     momentum: float = 0.9, weight_decay: float = 1e-4) -> torch.optim.Optimizer:
    """Create optimizer for the model."""
    return torch.optim.SGD(model.parameters(), lr=float(lr), momentum=float(momentum), 
                           weight_decay=float(weight_decay))
