"""Spatio-Temporal Graph Transformer for traffic forecasting.

Inspired by GMAN (Zheng 2020) and the trajectory of GraphTransformer
variants (Dwivedi & Bresson 2021, Yun 2019). Models each sensor node with a
learnable spatial-attention layer over its road-network neighbours, and a
temporal Transformer over the time axis.
"""

from graph_tt.model import GraphTransformerTraffic
from graph_tt.data import TrafficWindowDataset, build_adjacency

__all__ = ["GraphTransformerTraffic", "TrafficWindowDataset", "build_adjacency"]
__version__ = "0.3.0"
