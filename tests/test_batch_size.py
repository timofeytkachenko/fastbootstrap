"""Tests for `_compute_optimal_batch_size` and its helpers.

Covers:
- README-aligned heuristic buckets (10K/100K/500K/1M+ samples).
- Edge cases for tiny workloads and minimal samples.
- Memory squeeze (low/high `available_memory_gb`).
- Batch-invariant peak-memory `ResourceWarning`.
- Load-balancing invariant (>= MIN_BATCHES_PER_WORKER chunks per worker).
- Input validation via `ValidationError`.
- joblib `effective_n_jobs` alignment in `_resolve_n_workers`.
"""

from __future__ import annotations

import math
import warnings

import pytest
from joblib import effective_n_jobs

from fastbootstrap.constants import (
    BATCH_SIZE_LARGE,
    BATCH_SIZE_MASSIVE,
    BATCH_SIZE_MEDIUM,
    BATCH_SIZE_SMALL,
    LOW_MEM_BATCH_CAP,
    MIN_BATCH_FLOOR,
    MIN_BATCHES_PER_WORKER,
    SAMPLE_COMPLEXITY_DIVISOR,
)
from fastbootstrap.core import (
    _compute_optimal_batch_size,
    _resolve_n_workers,
)
from fastbootstrap.exceptions import ValidationError


# ---------------------------------------------------------------------------
# Worker resolution
# ---------------------------------------------------------------------------


class TestResolveNWorkers:
    """Cover joblib-style ``n_jobs`` interpretation via ``effective_n_jobs``."""

    @pytest.mark.parametrize("n_jobs", [-1, -2, 1, 4, 64])
    def test_matches_joblib(self, n_jobs: int) -> None:
        """Resolution must mirror what ``Parallel`` would actually use."""
        assert _resolve_n_workers(n_jobs) == max(1, effective_n_jobs(n_jobs))

    def test_explicit_positive_is_not_capped(self) -> None:
        """loky allows oversubscription for explicit positive ``n_jobs``."""
        assert _resolve_n_workers(1) == 1
        assert _resolve_n_workers(64) == 64

    def test_zero_raises(self) -> None:
        with pytest.raises(ValidationError):
            _resolve_n_workers(0)


# ---------------------------------------------------------------------------
# README heuristic buckets
# ---------------------------------------------------------------------------


@pytest.fixture
def big_mem() -> float:
    """Headroom-rich memory budget that disables the memory-aware cap."""
    return 1024.0  # 1 TiB free RAM


@pytest.mark.parametrize(
    ("n_samples", "expected_base"),
    [
        (1_000, BATCH_SIZE_SMALL),
        (10_000, BATCH_SIZE_SMALL),  # inclusive right edge
        (10_001, BATCH_SIZE_MEDIUM),
        (50_000, BATCH_SIZE_MEDIUM),
        (100_000, BATCH_SIZE_MEDIUM),  # inclusive right edge
        (100_001, BATCH_SIZE_LARGE),
        (500_000, BATCH_SIZE_LARGE),  # inclusive right edge
        (500_001, BATCH_SIZE_MASSIVE),
        (5_000_000, BATCH_SIZE_MASSIVE),
    ],
)
def test_heuristic_buckets(n_samples: int, expected_base: int, big_mem: float) -> None:
    """Base heuristic value is returned when no other cap binds."""
    batch = _compute_optimal_batch_size(
        n_samples,
        sample_size=1_000,
        n_jobs=1,  # n_workers=1 makes load-balancing cap == N
        available_memory_gb=big_mem,
    )
    # With 1 worker, max_batch = N // (1 * 4) = N // 4 >> base.
    # With infinite memory, mem_cap is huge. So the base heuristic wins.
    assert batch == expected_base


# ---------------------------------------------------------------------------
# Load-balancing invariant (the bug-fix)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n_samples", "n_workers"),
    [
        (1_000_000, 8),
        (1_000_000, 16),
        (5_000_000, 8),
        (100_000, 4),
        (10_000, 2),
    ],
)
def test_load_balancing_invariant(
    n_samples: int, n_workers: int, big_mem: float
) -> None:
    """Total number of batches must be >= MIN_BATCHES_PER_WORKER * n_workers."""
    batch = _compute_optimal_batch_size(
        n_samples,
        sample_size=1_000,
        n_jobs=n_workers,
        available_memory_gb=big_mem,
    )
    total_batches = math.ceil(n_samples / batch)
    assert total_batches >= n_workers * MIN_BATCHES_PER_WORKER


def test_previous_inversion_fixed(big_mem: float) -> None:
    """Regression: with 1M samples and 8 workers the old code returned 31_250.

    After the fix the batch must stay close to the heuristic (<= 1000) when
    workload spread allows it -- well below the previously buggy ~31k value.
    """
    batch = _compute_optimal_batch_size(
        1_000_000,
        sample_size=1_000,
        n_jobs=8,
        available_memory_gb=big_mem,
    )
    assert batch <= BATCH_SIZE_MASSIVE
    assert batch >= MIN_BATCH_FLOOR


# ---------------------------------------------------------------------------
# Memory tiers and memory-aware cap
# ---------------------------------------------------------------------------


def test_low_memory_caps_to_low_mem_cap() -> None:
    """RAM below MEMORY_LOW_THRESHOLD forces batch <= LOW_MEM_BATCH_CAP."""
    batch = _compute_optimal_batch_size(
        100_000,
        sample_size=1_000,
        n_jobs=1,
        available_memory_gb=1.0,  # < MEMORY_LOW_THRESHOLD (4.0)
    )
    assert batch <= LOW_MEM_BATCH_CAP


def test_moderate_memory_caps_to_medium() -> None:
    """RAM in [MEMORY_LOW, MEMORY_MODERATE) caps batch at BATCH_SIZE_MEDIUM."""
    batch = _compute_optimal_batch_size(
        500_000,
        sample_size=1_000,
        n_jobs=1,
        available_memory_gb=6.0,  # in [4.0, 8.0)
    )
    assert batch <= BATCH_SIZE_MEDIUM


def test_tight_memory_emits_resource_warning() -> None:
    """Peak memory is batch-invariant: tight RAM warns instead of shrinking batch."""
    with pytest.warns(ResourceWarning, match="peak resampling memory"):
        batch = _compute_optimal_batch_size(
            1_000_000,
            sample_size=10_000_000,  # 160 MB transient per worker (data + indices)
            n_jobs=4,
            available_memory_gb=2.0,
        )
    # peak = 4 * 10M * 16 = 640 MB > 0.25 * 2 GiB = 512 MB -> warning.
    # Batch itself follows low-mem tier (64) halved for wide samples (32),
    # not crushed to the floor by a phantom per-batch memory cost.
    assert batch == LOW_MEM_BATCH_CAP // SAMPLE_COMPLEXITY_DIVISOR


def test_plentiful_memory_emits_no_warning(big_mem: float) -> None:
    """No ResourceWarning when the peak estimate fits the budget."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", ResourceWarning)
        batch = _compute_optimal_batch_size(
            100_000, sample_size=1_000, n_jobs=4, available_memory_gb=big_mem
        )
    assert batch >= MIN_BATCH_FLOOR


# ---------------------------------------------------------------------------
# Sample-complexity dampening
# ---------------------------------------------------------------------------


def test_large_sample_halves_base(big_mem: float) -> None:
    """sample_size > LARGE_SAMPLE_THRESHOLD halves the base heuristic."""
    base = _compute_optimal_batch_size(
        50_000, sample_size=1_000, n_jobs=1, available_memory_gb=big_mem
    )
    halved = _compute_optimal_batch_size(
        50_000, sample_size=200_000, n_jobs=1, available_memory_gb=big_mem
    )
    assert halved <= base // SAMPLE_COMPLEXITY_DIVISOR + 1  # +1 for floor


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs"),
    [
        {"number_of_bootstrap_samples": 0, "sample_size": 100, "n_jobs": -1},
        {"number_of_bootstrap_samples": -5, "sample_size": 100, "n_jobs": -1},
        {"number_of_bootstrap_samples": 100, "sample_size": 1, "n_jobs": -1},
        {"number_of_bootstrap_samples": 100, "sample_size": 100, "n_jobs": 0},
    ],
)
def test_validation_errors(kwargs: dict) -> None:
    with pytest.raises(ValidationError):
        _compute_optimal_batch_size(**kwargs)


def test_dtype_bytes_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        _compute_optimal_batch_size(1_000, sample_size=100, n_jobs=-1, dtype_bytes=0)


@pytest.mark.parametrize("bad_memory", [float("nan"), float("inf"), -1.0])
def test_invalid_available_memory_raises(bad_memory: float) -> None:
    """NaN, inf or negative memory must fail with ValidationError, not ValueError."""
    with pytest.raises(ValidationError):
        _compute_optimal_batch_size(
            1_000, sample_size=100, n_jobs=1, available_memory_gb=bad_memory
        )


# ---------------------------------------------------------------------------
# General invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_samples", [2, 16, 128, 1_024, 100_000, 5_000_000])
@pytest.mark.parametrize("n_jobs", [-1, -2, 1, 4])
def test_floor_invariant(n_samples: int, n_jobs: int) -> None:
    """Output is always >= MIN_BATCH_FLOOR regardless of inputs."""
    batch = _compute_optimal_batch_size(
        n_samples, sample_size=100, n_jobs=n_jobs, available_memory_gb=8.0
    )
    assert batch >= MIN_BATCH_FLOOR
    assert isinstance(batch, int)
