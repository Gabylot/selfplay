"""Dual-head ResNet for AlphaZero.

Small, CPU/GPU-friendly network with:
- Policy head: outputs move probability distribution over game.num_actions
  (dimensions from the active game, see game_api.py)
- Value head: outputs scalar in [-1, 1] for expected outcome from current player's perspective
- Random weight initialization (no pretraining)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

import game_api


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
        out = out + residual
        out = F.relu(out)
        return out


class AlphaZeroNet(nn.Module):
    """Dual-head ResNet for game position evaluation.

    These dimensions come from the active game (see game_api.py), so the
    same architecture serves chess (119 planes, 73 move-channels, 8x8),
    connect four (4 planes, 1 channel, 6x7), or any future game.

    Architecture:
        Input conv (game.num_planes -> num_filters)
        N residual blocks (num_filters -> num_filters)
        Policy head: Conv 1x1 -> BN -> ReLU -> Conv 1x1
                     (game.policy_channels) -> game.num_actions logits
        Value head: Conv 1x1 (num_value_channels) -> BN -> ReLU
                    -> FC (value_fc_size) -> ReLU -> FC (1) -> tanh
    """

    def __init__(self,
                 num_residual_blocks: int = 8,
                 num_filters: int = 64,
                 num_policy_channels: int = 32,
                 num_value_channels: int = 32,
                 value_fc_size: int = 128):
        super().__init__()

        self.game_name = game_api.active_game_name()
        self.num_residual_blocks = num_residual_blocks
        self.num_filters = num_filters
        self.num_policy_channels = num_policy_channels
        self.num_value_channels = num_value_channels
        self.value_fc_size = value_fc_size
        self.num_planes = game_api.num_planes()
        self.num_actions = game_api.num_actions()
        self.policy_channels_out = game_api.policy_channels()
        self.board_h = game_api.board_h()
        self.board_w = game_api.board_w()

        self._channels_last = False

        # Input convolution
        self.input_conv = nn.Conv2d(self.num_planes, num_filters, kernel_size=3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(num_filters)

        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResBlock(num_filters) for _ in range(num_residual_blocks)
        ])

        # Policy head — one logit per (cell, move-channel) = num_actions.
        self.policy_conv = nn.Conv2d(num_filters, num_policy_channels, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(num_policy_channels)
        self.policy_logits_conv = nn.Conv2d(num_policy_channels, self.policy_channels_out, kernel_size=1, bias=True)

        # Value head — batch-normalized 1x1 convolution with
        # num_value_channels filters, a rectified FC layer, then a
        # tanh-linear layer of size 1.
        self.value_conv = nn.Conv2d(num_filters, num_value_channels, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(num_value_channels)
        self.value_fc1 = nn.Linear(num_value_channels * self.board_h * self.board_w, value_fc_size)
        self.value_fc2 = nn.Linear(value_fc_size, 1)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights.

        - Kaiming uniform for convs / hidden linear layers.
        - Zero-init the last BN gamma in every residual block so each block
          starts as an identity function (more stable early training).
        - Scale down the final policy / value layers so the net starts near a
          uniform policy and a zero value.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Residual blocks start as identity.
        for block in self.res_blocks:
            nn.init.zeros_(block.bn2.weight)

        # Small final layers -> ~uniform policy, ~0 value at init.
        with torch.no_grad():
            self.policy_logits_conv.weight.mul_(0.1)
            self.value_fc2.weight.mul_(0.1)

    def enable_channels_last(self):
        """Switch to NHWC memory format (often faster for convs on GPU).

        Call once after moving the model to its device. predictBatch() will
        convert inputs automatically.
        """
        self.to(memory_format=torch.channels_last)
        self._channels_last = True
        return self

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, game.num_planes, H, W)

        Returns:
            policy_logits: (batch, game.num_actions) raw logits
            value: (batch, 1) scalar in [-1, 1]
        """
        # Input convolution
        x = F.relu(self.input_bn(self.input_conv(x)))

        # Residual blocks
        for block in self.res_blocks:
            x = block(x)

        # Policy head: rectified BN-conv, then a final 1x1 convolution over
        # game.policy_channels move-channels per cell.  Rearranged from
        # (B, channels, H, W) to the canonical flat (B, H*W*channels)
        # ordering — matching the game's to_index() encoding.
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy_logits = self.policy_logits_conv(policy)
        policy_logits = policy_logits.permute(0, 2, 3, 1)  # (B, H, W, ch)
        policy_logits = policy_logits.reshape(policy_logits.size(0), -1)  # (B, num_actions)

        # Value head: num_value_channels 1x1 conv, rectified FC, tanh-linear.
        # reshape (not view) so it works with channels_last tensors.
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.reshape(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy_logits, value

    def predict(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        """Predict policy and value for a single state.

        Args:
            state: (game.num_planes, H, W) numpy array

        Returns:
            policy: (game.num_actions,) numpy array of probabilities
            value: scalar float in [-1, 1]
        """
        policies, values = self.predictBatch(state[np.newaxis, ...])
        return policies[0], float(values[0])

    def predictBatch(self, states: np.ndarray,
                     amp_dtype: Optional[torch.dtype] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Predict policy and value for a batch of states.

        Args:
            states: (batch, game.num_planes, H, W) numpy array
            amp_dtype: optional autocast dtype (torch.float16 / torch.bfloat16)
                for faster inference. Outputs are always returned in float32.

        Returns:
            policies: (batch, game.num_actions) numpy array of probabilities
            values: (batch,) numpy array of scalars
        """
        self.eval()
        # inference_mode is faster than no_grad (no version counter tracking)
        with torch.inference_mode():
            device = next(self.parameters()).device
            x = torch.from_numpy(states).to(device=device, dtype=torch.float32)
            if self._channels_last:
                x = x.contiguous(memory_format=torch.channels_last)

            if amp_dtype is not None:
                with torch.autocast(device_type=device.type, dtype=amp_dtype):
                    policy_logits, value = self.forward(x)
            else:
                policy_logits, value = self.forward(x)

            policies = F.softmax(policy_logits.float(), dim=1).cpu().numpy()
            values = value.float().squeeze(-1).cpu().numpy()
        return policies, values


def save_checkpoint(model: AlphaZeroNet, optimizer, path: str,
                    step: int = 0, extra: Optional[dict] = None):
    """Save model checkpoint (includes everything needed to rebuild the model)."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'step': step,
        'game_name': model.game_name,
        'num_residual_blocks': model.num_residual_blocks,
        'num_filters': model.num_filters,
        'num_policy_channels': model.num_policy_channels,
        'num_value_channels': model.num_value_channels,
        'value_fc_size': model.value_fc_size,
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
    # weights_only=True: checkpoints contain only tensors / primitives, so we
    # never unpickle arbitrary objects from a checkpoint file.
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)

    if model is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None and checkpoint.get('optimizer_state_dict') is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint


def create_model_from_checkpoint(path: str) -> Tuple[AlphaZeroNet, dict]:
    """Rebuild a model purely from a checkpoint file (no Config needed).

    Raises ValueError if the checkpoint was trained for a different game than
    the currently active one, or if it predates the metadata additions.
    Returns (model, checkpoint_dict).
    """
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)

    required = ('num_residual_blocks', 'num_filters', 'num_policy_channels',
                'num_value_channels', 'value_fc_size')
    missing = [k for k in required if k not in checkpoint]
    if missing:
        raise ValueError(
            f"Checkpoint lacks architecture metadata {missing} (saved by an older "
            f"version). Build the model from Config and use load_checkpoint() instead."
        )

    ckpt_game = checkpoint.get('game_name')
    if ckpt_game is not None and ckpt_game != game_api.active_game_name():
        raise ValueError(
            f"Checkpoint was trained for '{ckpt_game}' but the active game is "
            f"'{game_api.active_game_name()}'."
        )

    model = AlphaZeroNet(
        num_residual_blocks=checkpoint['num_residual_blocks'],
        num_filters=checkpoint['num_filters'],
        num_policy_channels=checkpoint['num_policy_channels'],
        num_value_channels=checkpoint['num_value_channels'],
        value_fc_size=checkpoint['value_fc_size'],
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint


def create_model_from_config(config) -> AlphaZeroNet:
    """Create an AlphaZeroNet from a Config object."""
    return AlphaZeroNet(
        num_residual_blocks=config.network.num_residual_blocks,
        num_filters=config.network.num_filters,
        num_policy_channels=config.network.num_policy_channels,
        num_value_channels=config.network.num_value_channels,
        value_fc_size=config.network.value_fc_size,
    )