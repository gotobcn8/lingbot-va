from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
import torch


def _load_alignment_module():
    module_path = (
        Path(__file__).resolve().parents[1] / "wan_va" / "modules" / "alignment.py"
    )
    spec = spec_from_file_location("alignment_under_test", module_path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


alignment = _load_alignment_module()


def test_motion_incremental_alignment_returns_zero_for_identical_inputs():
    batch_size = 2
    frames = 4
    tokens = 3
    hidden_dim = 5

    x = torch.randn(batch_size, frames * tokens, hidden_dim)
    loss = alignment.motion_incremental_alignment(x, x.clone(), Tokens=tokens)

    assert loss.shape == torch.Size([])
    assert torch.isclose(loss, torch.tensor(0.0, dtype=loss.dtype), atol=1e-6)


@pytest.mark.parametrize("pool", ["mean", "max"])
def test_motion_incremental_alignment_runs_with_valid_inputs(pool):
    # Construct frame-wise motion so temporal deltas are non-zero and stable.
    frame_features = torch.tensor(
        [[[0.0, 1.0], [1.0, 3.0], [3.0, 6.0], [6.0, 10.0]]], dtype=torch.float32
    )
    a = frame_features.repeat_interleave(2, dim=1)
    b = a.clone()

    loss = alignment.motion_incremental_alignment(a, b, Tokens=2, pool=pool)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() <= 1e-6


def test_motion_incremental_alignment_returns_zero_when_only_one_frame():
    x = torch.randn(2, 4, 6)

    loss = alignment.motion_incremental_alignment(x, x.clone(), Tokens=4)

    assert loss.shape == torch.Size([])
    assert loss.item() == 0.0


def test_motion_incremental_alignment_raises_for_invalid_sequence_length():
    x = torch.randn(1, 5, 4)

    with pytest.raises(ValueError, match="not divisible by Tokens"):
        alignment.motion_incremental_alignment(x, x.clone(), Tokens=2)


def test_motion_incremental_alignment_raises_for_unknown_pool():
    x = torch.randn(1, 6, 4)

    with pytest.raises(ValueError, match="Unknown pool type"):
        alignment.motion_incremental_alignment(x, x.clone(), Tokens=3, pool="sum")


def test_unified_trace_align_returns_components_for_multiple_batches():
    batch_size = 2
    frames = 4
    tokens = 3
    hidden_dim = 2

    base_tokens = torch.tensor(
        [[[1.0, 2.0], [2.0, 1.0], [1.0, -1.0]]], dtype=torch.float32
    )
    scales = torch.arange(1, frames + 1, dtype=torch.float32).view(1, frames, 1, 1)
    x = (scales * base_tokens[:, None]).repeat(batch_size, 1, 1, 1)
    x = x.reshape(batch_size, frames * tokens, hidden_dim)

    future_loss, motion_loss = alignment.UnifiedTraceAlign(
        x,
        x.clone(),
        Tokens=tokens,
    )

    assert future_loss.shape == torch.Size([])
    assert motion_loss.shape == torch.Size([])
    assert torch.isfinite(future_loss)
    assert torch.isfinite(motion_loss)
    assert future_loss.item() <= 1e-6
    assert motion_loss.item() <= 1e-6


def test_unified_trace_align_handles_one_frame():
    x = torch.randn(2, 3, 4)

    future_loss, motion_loss = alignment.UnifiedTraceAlign(x, x.clone(), Tokens=3)

    assert future_loss.shape == torch.Size([])
    assert motion_loss.shape == torch.Size([])
    assert future_loss.item() == 0.0
    assert motion_loss.item() == 0.0


def test_unified_trace_align_raises_for_invalid_sequence_length():
    x = torch.randn(1, 5, 4)

    with pytest.raises(ValueError, match="not divisible by Tokens"):
        alignment.UnifiedTraceAlign(x, x.clone(), Tokens=2)
