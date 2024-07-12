import numpy as np
import pytest
import torch

from graph_tt.data import TrafficWindowDataset, build_adjacency, synthesize


def test_synthesize_shapes():
    v, d = synthesize(n_steps=200, n_nodes=8)
    assert v.shape == (200, 8, 1)
    assert d.shape == (8, 8)


def test_window_dataset_length_correct_for_stride_one():
    v, _ = synthesize(n_steps=100, n_nodes=4)
    ds = TrafficWindowDataset(v, encoder_length=12, decoder_length=12)
    assert len(ds) == 100 - 12 - 12 + 1


def test_window_dataset_returns_tensors_of_expected_shape():
    v, _ = synthesize(n_steps=100, n_nodes=4)
    ds = TrafficWindowDataset(v, encoder_length=12, decoder_length=12)
    enc, dec = ds[0]
    assert enc.shape == (12, 4, 1)
    assert dec.shape == (12, 4)


def test_window_dataset_normalizes_to_near_unit_variance():
    v, _ = synthesize(n_steps=300, n_nodes=8, seed=1)
    ds = TrafficWindowDataset(v, encoder_length=12, decoder_length=12)
    flat = ds.values.reshape(-1, ds.values.shape[-1])
    assert abs(flat.std(axis=0).mean() - 1.0) < 0.2


def test_too_short_series_raises():
    with pytest.raises(ValueError):
        TrafficWindowDataset(np.zeros((10, 4, 1), dtype=np.float32),
                             encoder_length=12, decoder_length=12)
