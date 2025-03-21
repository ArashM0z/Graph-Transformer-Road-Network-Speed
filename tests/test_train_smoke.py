import os
from pathlib import Path

os.environ["WANDB_MODE"] = "disabled"

from graph_tt.train import TrainConfig, train_synthetic


def test_train_smoke_runs_two_epochs_and_returns_finite_metrics(tmp_path: Path):
    cfg = TrainConfig(
        encoder_length=12, decoder_length=6, hidden=16, n_heads=2, n_layers=1,
        epochs=2, batch_size=8, device="cpu",
        checkpoint_dir=str(tmp_path / "ckpt"),
        mlflow_tracking_uri=f"file:{tmp_path / 'mlruns'}",
    )
    out = train_synthetic(cfg)
    for k, v in out.items():
        assert isinstance(v, float) and v == v  # not NaN
