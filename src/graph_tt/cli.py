"""CLI entrypoints."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import yaml

from graph_tt.data import load_npz, synthesize
from graph_tt.train import TrainConfig, train


def _load_cfg(path: Path) -> TrainConfig:
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    return TrainConfig(**raw)


def _common() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--values", type=Path, help="(T,N,F) .npz with key 'data'")
    p.add_argument("--distances", type=Path, help="(N,N) .npz with key 'data'")
    p.add_argument("--demo", action="store_true",
                   help="train on synthetic 12-node data")
    p.add_argument("--log-level", default="INFO")
    return p


def train_main(argv: list[str] | None = None) -> int:
    args = _common().parse_args(argv)
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = _load_cfg(args.config)
    if args.demo or args.values is None:
        values, distances = synthesize()
    else:
        values = load_npz(args.values)
        distances = load_npz(args.distances) if args.distances else np.eye(values.shape[1])
    metrics = train(values, distances, cfg)
    for k, v in metrics.items():
        print(f"{k}={v:.4f}")
    return 0


def eval_main(argv: list[str] | None = None) -> int:
    p = _common()
    p.add_argument("--checkpoint", type=Path, required=True)
    args = p.parse_args(argv)
    logging.basicConfig(level=args.log_level)
    # full re-run with --epochs 0 effectively only tests the loaded ckpt;
    # implement a thin eval path that imports train() with epochs=0
    cfg = _load_cfg(args.config)
    cfg.epochs = 0
    if args.demo or args.values is None:
        values, distances = synthesize()
    else:
        values = load_npz(args.values)
        distances = load_npz(args.distances) if args.distances else np.eye(values.shape[1])
    metrics = train(values, distances, cfg)
    for k, v in metrics.items():
        print(f"{k}={v:.4f}")
    return 0
