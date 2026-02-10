# Graph Transformer — Road-Network Traffic Forecasting

PyTorch implementation of a spatio-temporal graph transformer for short-
horizon road-network traffic speed forecasting (METR-LA / PEMS-BAY style).
Each transformer block alternates a node-level **spatial attention**
(masked by a thresholded Gaussian-kernel adjacency) with a per-node
**temporal attention** (causal) and a position-wise feed-forward.

References:
- Dwivedi & Bresson (2021), *A Generalization of Transformer Networks to Graphs*.
- Zheng et al. (2020), *GMAN: A Graph Multi-Attention Network for Traffic Prediction*.
- Li et al. (2018), *Diffusion Convolutional Recurrent Neural Network* (DCRNN, adjacency construction).

## What's in the box

- Spatial attention masked by `build_adjacency(distances)` (Gaussian
  kernel with auto-tuned σ and configurable threshold).
- Causal temporal attention over the time axis.
- AdamW + cosine LR schedule, gradient clipping.
- MLflow + Weights & Biases dual logging.
- Synthetic 12-node fixture so the full training loop runs on a CPU in
  under 30 seconds via `make smoke`.

## Quickstart

```bash
pip install -e ".[dev]"
make smoke                           # ~30s CPU
gtt-train --config configs/default.yaml --values data/values.npz \
          --distances data/dist.npz
```

The `values.npz` must contain a `data` key of shape `(T, N, F)` and the
distance matrix is `(N, N)`. Use the synthesiser in
[src/graph_tt/data.py](src/graph_tt/data.py) as a schema reference.

## Layout

```
src/graph_tt/
├── model.py    # Spatial + temporal multi-head attention + FFN blocks
├── data.py     # Sliding-window dataset + Gaussian-kernel adjacency
├── train.py    # AdamW + cosine + MLflow + W&B
├── cli.py      # gtt-train / gtt-eval entrypoints
configs/        # default.yaml + smoke.yaml
tests/          # model shapes, gradients, adjacency, end-to-end smoke
```
