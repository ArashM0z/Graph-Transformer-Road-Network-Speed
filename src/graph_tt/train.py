"""Training loop with MLflow + W&B dual logging."""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from graph_tt.data import TrafficWindowDataset, build_adjacency, synthesize
from graph_tt.model import GraphTransformerTraffic

log = logging.getLogger(__name__)

try:
    import wandb
    _WANDB = True
except ImportError:  # pragma: no cover - optional
    _WANDB = False


@dataclass
class TrainConfig:
    encoder_length: int = 12
    decoder_length: int = 12
    hidden: int = 64
    n_heads: int = 4
    n_layers: int = 3
    dropout: float = 0.1
    epochs: int = 30
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    seed: int = 20260514
    device: str = "cuda"
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    checkpoint_dir: str = "checkpoints"
    mlflow_tracking_uri: str = "file:./mlruns"
    mlflow_experiment: str = "graph-tt"
    wandb_project: str = "graph-tt"


def masked_mae(pred: torch.Tensor, target: torch.Tensor,
               mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is None:
        return (pred - target).abs().mean()
    return ((pred - target).abs() * mask).sum() / mask.sum().clamp(min=1)


def _seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _split(values: np.ndarray, val_frac: float, test_frac: float
           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = values.shape[0]
    n_val = int(n * val_frac)
    n_test = int(n * test_frac)
    n_train = n - n_val - n_test
    return values[:n_train], values[n_train:n_train + n_val], values[n_train + n_val:]


def _init_wandb(cfg: TrainConfig, run_name: str):
    if not _WANDB or os.environ.get("WANDB_MODE") == "disabled":
        return None
    return wandb.init(project=cfg.wandb_project, name=run_name,
                      config=cfg.__dict__, reinit=True)


def train(values: np.ndarray, distances: np.ndarray, cfg: TrainConfig
          ) -> dict[str, float]:
    _seed_all(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    adj = build_adjacency(distances).to(device)

    train_v, val_v, test_v = _split(values, cfg.val_fraction, cfg.test_fraction)
    train_ds = TrafficWindowDataset(train_v, cfg.encoder_length, cfg.decoder_length)
    val_ds = TrafficWindowDataset(val_v, cfg.encoder_length, cfg.decoder_length)
    test_ds = TrafficWindowDataset(test_v, cfg.encoder_length, cfg.decoder_length)

    common = dict(batch_size=cfg.batch_size, num_workers=0, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)

    n_nodes = values.shape[1]
    input_dim = values.shape[2]
    model = GraphTransformerTraffic(
        n_nodes=n_nodes, input_dim=input_dim,
        hidden=cfg.hidden, n_heads=cfg.n_heads, n_layers=cfg.n_layers,
        dropout=cfg.dropout, encoder_length=cfg.encoder_length,
        decoder_length=cfg.decoder_length,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.mlflow_experiment)
    run_name = f"gtt-{int(time.time())}"
    wb_run = _init_wandb(cfg, run_name)

    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    best_path = Path(cfg.checkpoint_dir) / "best.pt"
    best_val = float("inf")

    def step_loader(loader, train_mode: bool) -> tuple[float, float]:
        model.train(train_mode)
        total = 0.0
        rmse_total = 0.0
        n = 0
        for enc, target in loader:
            enc = enc.to(device)
            target = target.to(device)
            with torch.set_grad_enabled(train_mode):
                pred = model(enc, adj)
                loss = masked_mae(pred, target)
                if train_mode:
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    opt.step()
            total += loss.item() * enc.size(0)
            rmse_total += F.mse_loss(pred, target).item() * enc.size(0)
            n += enc.size(0)
        return total / max(n, 1), float(np.sqrt(rmse_total / max(n, 1)))

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(cfg.__dict__)
        for ep in range(cfg.epochs):
            t0 = time.time()
            tr_mae, tr_rmse = step_loader(train_loader, True)
            val_mae, val_rmse = step_loader(val_loader, False)
            sched.step()
            row = {"train_mae": tr_mae, "train_rmse": tr_rmse,
                   "val_mae": val_mae, "val_rmse": val_rmse,
                   "lr": opt.param_groups[0]["lr"], "epoch_sec": time.time() - t0}
            mlflow.log_metrics(row, step=ep)
            if wb_run:
                wb_run.log(row, step=ep)
            log.info("ep=%03d train_mae=%.4f val_mae=%.4f val_rmse=%.4f",
                     ep, tr_mae, val_mae, val_rmse)
            if val_mae < best_val:
                best_val = val_mae
                torch.save(model.state_dict(), best_path)
                mlflow.log_artifact(str(best_path), artifact_path="checkpoints")

        model.load_state_dict(torch.load(best_path, map_location=device))
        test_mae, test_rmse = step_loader(test_loader, False)
        mlflow.log_metrics({"test_mae": test_mae, "test_rmse": test_rmse})
        mlflow.pytorch.log_model(model, artifact_path="model")

    if wb_run:
        wb_run.finish()

    return {"best_val_mae": float(best_val), "test_mae": float(test_mae),
            "test_rmse": float(test_rmse)}


def train_synthetic(cfg: TrainConfig | None = None) -> dict[str, float]:
    cfg = cfg or TrainConfig()
    values, distances = synthesize(n_steps=600, n_nodes=12)
    return train(values, distances, cfg)
