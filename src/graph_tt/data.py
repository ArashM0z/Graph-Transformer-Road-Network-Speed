"""Traffic-speed sliding-window dataset (METR-LA / PEMS-BAY style)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset


def build_adjacency(distances: np.ndarray, sigma: float | None = None,
                    threshold: float = 0.1) -> torch.Tensor:
    """Build a thresholded Gaussian-kernel adjacency mask from a distance matrix.

    Follows the formulation in Li et al. (2018) DCRNN, eq. (5)–(6):

        W_ij = exp(-d_ij^2 / sigma^2)  if exp(...) >= threshold else 0
        sigma defaults to the std of the off-diagonal distances.
    """
    if distances.shape[0] != distances.shape[1]:
        raise ValueError("distances must be a square matrix")
    n = distances.shape[0]
    iu = np.triu_indices(n, k=1)
    off_diag = distances[iu]
    if sigma is None:
        sigma = float(off_diag.std() + 1e-8)
    w = np.exp(-(distances ** 2) / (sigma ** 2))
    w[w < threshold] = 0.0
    np.fill_diagonal(w, 1.0)
    return torch.from_numpy((w > 0).astype(np.bool_))


class TrafficWindowDataset(Dataset):
    """Sliding-window dataset over a (T, N, F) tensor.

    Produces (encoder_window, decoder_window) pairs with stride 1.
    """

    def __init__(self, values: np.ndarray, encoder_length: int = 12,
                 decoder_length: int = 12, normalize: bool = True):
        if values.ndim != 3:
            raise ValueError("values must have shape (T, N, F)")
        self.encoder_length = encoder_length
        self.decoder_length = decoder_length
        self._n_samples = values.shape[0] - encoder_length - decoder_length + 1
        if self._n_samples <= 0:
            raise ValueError("not enough timesteps for the requested windows")

        v = values.astype(np.float32)
        if normalize:
            mu = v.mean(axis=(0, 1), keepdims=True)
            std = v.std(axis=(0, 1), keepdims=True) + 1e-6
            v = (v - mu) / std
            self.mean = mu
            self.std = std
        else:
            self.mean = np.zeros((1, 1, v.shape[-1]), dtype=np.float32)
            self.std = np.ones((1, 1, v.shape[-1]), dtype=np.float32)
        self.values = v

    def __len__(self) -> int:
        return self._n_samples

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        enc = self.values[i: i + self.encoder_length]
        dec = self.values[
            i + self.encoder_length:
            i + self.encoder_length + self.decoder_length
        ]
        # the prediction target is the speed channel (assume index 0)
        return torch.from_numpy(enc), torch.from_numpy(dec[..., 0])

    def denormalize_target(self, y: torch.Tensor) -> torch.Tensor:
        return y * self.std[0, 0, 0] + self.mean[0, 0, 0]


def load_npz(path: str | Path) -> np.ndarray:
    """Load a (T, N, F) array from a .npz with a 'data' key."""
    return np.load(path, allow_pickle=False)["data"]


def synthesize(n_steps: int = 600, n_nodes: int = 12, n_features: int = 1,
               seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic traffic data: sinusoidal base + per-node phase shift + noise.

    Returns:
        (values, distances): shapes (T, N, F) and (N, N).
    """
    rng = np.random.default_rng(seed)
    phase = rng.uniform(0, 2 * np.pi, n_nodes)
    base = np.zeros((n_steps, n_nodes), dtype=np.float32)
    for n in range(n_nodes):
        t = np.arange(n_steps)
        base[:, n] = 50 + 20 * np.sin(2 * np.pi * t / 96 + phase[n])
    base += rng.normal(0, 3, size=base.shape).astype(np.float32)
    values = base[..., None]
    if n_features > 1:
        extras = rng.normal(0, 1, size=(n_steps, n_nodes, n_features - 1)).astype(np.float32)
        values = np.concatenate([values, extras], axis=-1)
    # toy distances: nodes laid out on a circle
    angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
    pts = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    distances = np.linalg.norm(pts[:, None] - pts[None, :], axis=-1)
    return values, distances
