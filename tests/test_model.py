import numpy as np
import pytest
import torch

from graph_tt.data import build_adjacency, synthesize
from graph_tt.model import GraphTransformerTraffic


def _model(n_nodes=12, input_dim=1, enc=12, dec=12):
    return GraphTransformerTraffic(
        n_nodes=n_nodes, input_dim=input_dim, hidden=32, n_heads=4,
        n_layers=2, encoder_length=enc, decoder_length=dec,
    )


def test_forward_returns_correct_shape():
    m = _model()
    x = torch.randn(3, 12, 12, 1)
    adj = torch.ones(12, 12, dtype=torch.bool)
    y = m(x, adj)
    assert y.shape == (3, 12, 12)


def test_gradient_flows_through_attention_layers():
    m = _model()
    x = torch.randn(2, 12, 12, 1, requires_grad=True)
    adj = torch.ones(12, 12, dtype=torch.bool)
    y = m(x, adj)
    y.pow(2).mean().backward()
    n_grads = sum(int(p.grad is not None and p.grad.abs().sum() > 0)
                  for p in m.parameters() if p.requires_grad)
    assert n_grads > 0


def test_invalid_encoder_length_raises():
    m = _model(enc=12, dec=12)
    x = torch.randn(2, 8, 12, 1)
    adj = torch.ones(12, 12, dtype=torch.bool)
    with pytest.raises(ValueError):
        m(x, adj)


def test_hidden_must_divide_into_heads():
    with pytest.raises(ValueError):
        GraphTransformerTraffic(n_nodes=12, input_dim=1, hidden=15, n_heads=4)


def test_adjacency_from_distances_is_symmetric_and_bool():
    _, distances = synthesize(n_nodes=8)
    adj = build_adjacency(distances)
    assert adj.dtype == torch.bool
    assert torch.equal(adj, adj.T)
    assert adj.diag().all().item()
