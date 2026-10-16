"""Training loop for AlphaZero chess engine.

Samples batches from the replay buffer and trains the dual-head network
with separate policy loss (cross-entropy) and value loss (MSE).

Both losses are logged separately to the stats database.

NOTE: policy_loss_fn now receives a legal_move_mask for every sample so the
softmax normalizes only over actually-legal moves in that position, instead
of over the full 4672-action space (which wastes gradient signal modeling
probabilities for moves that could never be played).

NOTE: the replay buffer hands back compact arrays (states/policies float16,
masks uint8).  They are uploaded in that compact form and converted to
float32 on the device, which cuts host->device traffic by ~2.3x.  The four
scalar results are fetched with a single device sync instead of four.
"""

import time

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
                   device: torch.device = None,
                   grad_clip_norm: Optional[float] = 1.0) -> dict:
    """Perform a single training step.
    
    Args:
        model: The neural network
        optimizer: Optimizer
        replay_buffer: Replay buffer to sample from
        batch_size: Batch size
        device: Torch device
        grad_clip_norm: Max gradient norm for clipping (None to disable)
    
    Returns:
        Dict with 'policy_loss', 'value_loss', 'total_loss', 'grad_norm'
        (float scalars) plus timing: 't_sample' (seconds spent building the
        batch on the CPU) and 't_total' (whole step incl. device sync).
    """
    t_start = time.perf_counter()
    model.train()
    
    if device is None:
        device = next(model.parameters()).device
    
    # Sample batch from replay buffer (includes legal-move masks).
    states, policies, values, masks = replay_buffer.sample_batch(batch_size)
    t_sample = time.perf_counter() - t_start

    # Upload in compact dtypes, convert to float32 on the device.
    states = torch.from_numpy(states).to(device).float()
    policies = torch.from_numpy(policies).to(device).float()
    values = torch.from_numpy(values).to(device).float()
    masks = torch.from_numpy(masks).to(device).float()
    
    # Forward pass
    policy_logits, value_pred = model(states)
    
    # Compute losses
    p_loss = policy_loss_fn(policy_logits, policies, legal_move_masks=masks)
    v_loss = value_loss_fn(value_pred, values)
    total_loss = p_loss + v_loss
    
    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()

    if grad_clip_norm is not None:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
    else:
        grad_norm = torch.zeros((), device=device)

    optimizer.step()

    # One device sync for all four scalars (was four separate .item() calls).
    p_f, v_f, t_f, g_f = torch.stack([
        p_loss.detach().float(), v_loss.detach().float(),
        total_loss.detach().float(), grad_norm.detach().float(),
    ]).tolist()
    
    return {
        'policy_loss': p_f,
        'value_loss': v_f,
        'total_loss': t_f,
        'grad_norm': g_f,
        't_sample': t_sample,
        't_total': time.perf_counter() - t_start,
    }


def create_optimizer(model: AlphaZeroNet, lr: float = 0.001, 
                     momentum: float = 0.9, weight_decay: float = 1e-4) -> torch.optim.Optimizer:
    """Create optimizer for the model."""
    return torch.optim.SGD(model.parameters(), lr=float(lr), momentum=float(momentum), 
                           weight_decay=float(weight_decay))