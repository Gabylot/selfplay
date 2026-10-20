"""Training loop for AlphaZero chess engine.

Samples batches from the replay buffer and trains the dual-head network
with separate policy loss (cross-entropy) and value loss (MSE).

Both losses are logged separately to the stats database.

NOTE: policy_loss_fn now receives a legal_move_mask for every sample so the
softmax normalizes only over actually-legal moves in that position, instead
of over the full 4672-action space (which wastes gradient signal modeling
probabilities for moves that could never be played).

NOTE: the replay buffer keeps positions compactly (bit-planes + per-plane
scale, sparse policies, packed legal-move masks).  ``train_one_step`` asks it
for a *compact* batch (~2.7 MB per 1024 positions), uploads that, and expands
it to dense float32 tensors ON THE TRAINING DEVICE (``decode_compact_batch``)
-- the CPU only gathers a few small arrays.  Buffers without
``sample_batch_compact`` still work through the old dense path.  The four
scalar results are fetched with a single device sync.
"""

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional

from network import AlphaZeroNet, save_checkpoint
from selfplay import ReplayBuffer


def _unpack_bits(bits_u8: torch.Tensor, count: int) -> torch.Tensor:
    """Unpack big-endian bit-packed bytes (np.packbits order): (..., nbytes) -> (..., count) int32 0/1."""
    shifts = torch.arange(7, -1, -1, device=bits_u8.device, dtype=torch.int32)
    out = (bits_u8.to(torch.int32).unsqueeze(-1) >> shifts) & 1       # (..., nbytes, 8)
    return out.reshape(*bits_u8.shape[:-1], -1)[..., :count]


def decode_compact_batch(batch: dict, device):
    """Expand a ``ReplayBuffer.sample_batch_compact`` dict into dense float32
    ``(states, policies, values, masks)`` tensors on ``device``."""
    shape = tuple(batch['state_shape'])            # (P, H, W)
    A = int(batch['num_actions'])
    cells = int(np.prod(shape[1:]))

    # states: bit-planes * per-plane scale (or dense f16 fallback)
    if 'st_bits' in batch:
        bits = torch.from_numpy(batch['st_bits']).to(device)
        scale = torch.from_numpy(batch['st_scale']).to(device).float()
        planes = _unpack_bits(bits, cells).float() * scale.unsqueeze(-1)
        states = planes.reshape((-1,) + shape)
    else:
        states = torch.from_numpy(batch['states']).to(device).float()

    # policies: scatter the (index, value) pairs; padding index == A lands in a
    # spare column that is dropped.
    idx = torch.from_numpy(batch['pol_idx'].astype(np.int32)).to(device).long()
    val = torch.from_numpy(batch['pol_val']).to(device).float()
    dense = torch.zeros(idx.shape[0], A + 1, device=device)
    dense.scatter_(1, idx, val)
    policies = dense[:, :A].contiguous()

    masks = _unpack_bits(torch.from_numpy(batch['mask_bits']).to(device), A).float()
    values = torch.from_numpy(batch['values']).to(device).float()
    return states, policies, values, masks


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
        (float scalars) plus timing: 't_sample' (seconds spent gathering the
        batch on the CPU) and 't_total' (whole step incl. device sync).
    """
    t_start = time.perf_counter()
    model.train()
    
    if device is None:
        device = next(model.parameters()).device
    
    if hasattr(replay_buffer, 'sample_batch_compact'):
        # Compact batch: tiny CPU gather, upload, expand on the device.
        batch = replay_buffer.sample_batch_compact(batch_size)
        t_sample = time.perf_counter() - t_start
        states, policies, values, masks = decode_compact_batch(batch, device)
    else:
        # Dense path (buffers without compact sampling): f16/u8 arrays,
        # converted to float32 on the device.
        states, policies, values, masks = replay_buffer.sample_batch(batch_size)
        t_sample = time.perf_counter() - t_start
        states = torch.from_numpy(states).to(device).float()
        policies = torch.from_numpy(policies).to(device).float()
        values = torch.from_numpy(values).to(device).float()
        masks = torch.from_numpy(masks).to(device).float()

    # NOTE: the input encoding and policy indexing are now colour-neutral
    # (frame version 2), so no colour-mirror augmentation is needed or valid;
    # it was removed together with the frame change.

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