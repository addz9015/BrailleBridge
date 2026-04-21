"""
Denoiser Export Interface

EXPORTED INTERFACE for Person B.
Person B calls denoise(noisy_signal) to get clean reconstructions.

This module does NOT depend on training code.
Only imports: torch, numpy, model weights.
"""

import torch
import numpy as np
import os
from typing import Optional

# Default model path (can be overridden)
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "dae_best.pt")


class SimpleDAE(torch.nn.Module):
    """
    Export autoencoder architecture matching models/dae_simple.py.
    """

    def __init__(self):
        super().__init__()

        # Encoder: downsampling via stride
        self.enc1 = torch.nn.Sequential(
            torch.nn.Conv1d(12, 32, kernel_size=5, stride=1, padding=2),
            torch.nn.ReLU()
        )
        self.enc2 = torch.nn.Sequential(
            torch.nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            torch.nn.ReLU()
        )
        self.enc3 = torch.nn.Sequential(
            torch.nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU()
        )

        # Bottleneck
        self.bottleneck = torch.nn.Conv1d(64, 32, kernel_size=3, padding=1)

        # Decoder: upsampling + regular conv
        self.dec1 = torch.nn.Sequential(
            torch.nn.Conv1d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU()
        )
        self.dec2 = torch.nn.Sequential(
            torch.nn.Conv1d(64, 64, kernel_size=5, padding=2),
            torch.nn.ReLU()
        )
        self.dec3 = torch.nn.Conv1d(64, 12, kernel_size=5, padding=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, t = x.shape

        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        b = self.bottleneck(e3)

        d1 = self.dec1(b)
        d1_up = torch.nn.functional.interpolate(d1, scale_factor=2, mode='linear', align_corners=False)

        d2 = self.dec2(d1_up)
        d2_up = torch.nn.functional.interpolate(d2, scale_factor=2, mode='linear', align_corners=False)

        d3 = self.dec3(d2_up)

        if d3.shape[-1] != t:
            d3 = torch.nn.functional.interpolate(d3, size=t, mode='linear', align_corners=False)

        return d3


_model = None
_device = None


def initialize_model(model_path: Optional[str] = None,
                    device: str = 'cpu') -> None:
    """
    Initialize the denoising model.
    
    Args:
        model_path: Path to model weights. If None, uses default.
        device: 'cpu' or 'cuda'
    """
    global _model, _device
    
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at {model_path}. "
                               "Run training first to generate checkpoints/dae_best.pt")
    
    _model = SimpleDAE().to(device)
    _model.load_state_dict(torch.load(model_path, map_location=device))
    _model.eval()
    _device = device


def denoise(noisy_signal: np.ndarray,
           model_path: Optional[str] = None,
           device: str = 'cpu') -> np.ndarray:
    """
    Denoise a signal using the trained autoencoder.
    
    *** MAIN EXPORT INTERFACE - Person B uses this function ***
    
    Args:
        noisy_signal: Input signal, shape (T, 12), dtype float32
                     Values should be approximately in [-1, 1]
        model_path: Optional path to model weights.
                   If None, uses default or previously loaded model.
        device: 'cpu' or 'cuda'
    
    Returns:
        Denoised signal, shape (T, 12), dtype float32
    
    Example:
        >>> from models.denoiser import denoise
        >>> import h5py
        >>> f = h5py.File("data/corpus.h5", "r")
        >>> noisy = f["test/0/noisy"][:]  # (T, 12)
        >>> clean = denoise(noisy)  # (T, 12)
    """
    global _model, _device
    
    # Initialize if needed
    if _model is None:
        initialize_model(model_path, device)
    
    # Validate input
    if not isinstance(noisy_signal, np.ndarray):
        raise TypeError(f"Input must be numpy array, got {type(noisy_signal)}")
    
    if noisy_signal.ndim != 2:
        raise ValueError(f"Input must be 2D (T, 12), got shape {noisy_signal.shape}")
    
    if noisy_signal.shape[1] != 12:
        raise ValueError(f"Input must have 12 channels, got {noisy_signal.shape[1]}")
    
    if noisy_signal.dtype != np.float32:
        noisy_signal = noisy_signal.astype(np.float32)
    
    # Convert to tensor: (T, 12) -> (1, 12, T)
    signal_tensor = torch.from_numpy(noisy_signal).float().to(_device)
    signal_tensor = signal_tensor.transpose(0, 1).unsqueeze(0)
    
    # Forward pass
    with torch.no_grad():
        output = _model(signal_tensor)
    
    # Convert back: (1, 12, T) -> (T, 12)
    output_np = output.squeeze(0).transpose(0, 1).cpu().numpy().astype(np.float32)
    
    return output_np


# Alternative interface for batch processing
def denoise_batch(noisy_signals: list,
                 model_path: Optional[str] = None,
                 device: str = 'cpu',
                 batch_size: int = 8) -> list:
    """
    Denoise multiple signals (batch processing).
    
    Args:
        noisy_signals: List of (T, 12) arrays
        model_path: Optional path to model weights
        device: 'cpu' or 'cuda'
        batch_size: Batch size for processing
    
    Returns:
        List of denoised (T, 12) arrays
    """
    results = []
    
    for i in range(0, len(noisy_signals), batch_size):
        batch = noisy_signals[i:i + batch_size]
        
        for signal in batch:
            denoised = denoise(signal, model_path, device)
            results.append(denoised)
    
    return results
