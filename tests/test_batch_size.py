"""Tests for `_compute_optimal_batch_size` and its helpers.

Covers:
- README-aligned heuristic buckets (10K/100K/500K/1M+ samples).
- Edge cases for tiny workloads and minimal samples.
- Memory squeeze (low/high `available_memory_gb`).
- Batch-invariant peak-memory `ResourceWarning`.
- Load-balancing invariant (>= MIN_BATCHES_PER_WORKER chunks per worker).
- Input validation via `ValidationError`.
- joblib `effective_n_jobs` alignment in `_resolve_n_workers`.
- `bootstrap_resampling` rejects bad `batch_size` / `n_jobs` with
  `ValidationError` (not a re-wrapped joblib `ValueError`).
- `ResourceWarning` is attributed to the caller's file, not to `core.py`.
"""

from __future__ import annotations

import math
import os
import types
import warnings

import numpy as np
import pytest
from joblib import effective_n_jobs, parallel_backend

import fastbootstrap.core as core
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
    bootstrap_resampling,
)
from fastbootstrap.exceptions import ValidationError
from fastbootstrap.methods import bootstrap, one_sample_bootstrap, two_sample_bootstrap


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

    @pytest.mark.parametrize("bad", [True, 2.5, "4", [1]])
    def test_non_integer_raises(self, bad: object) -> None:
        """Non-integers must be a ValidationError, not a raw joblib TypeError."""
        with pytest.raises(ValidationError, match="n_jobs"):
            _resolve_n_workers(bad)  # type: ignore[arg-type]

    def test_numpy_integer_accepted(self) -> None:
        """NumPy integer scalars are legitimate n_jobs values."""
        assert _resolve_n_workers(np.int64(3)) == 3

    def test_none_defers_to_joblib(self) -> None:
        """``None`` is joblib's "unset" sentinel: 1 by default, or the value of
        an enclosing ``parallel_backend`` context. It must not be rejected."""
        assert _resolve_n_workers(None) == effective_n_jobs(None) == 1
        with parallel_backend("loky", n_jobs=2):
            assert _resolve_n_workers(None) == 2


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
        (5_000, BATCH_SIZE_SMALL),  # N // 4 = 1250 > base, so base wins
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
    # With 1 worker, max_batch = N // (1 * 4) = N // 4 > base for N >= 5K.
    # With infinite memory, no RAM tier binds. So the base heuristic wins.
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


def test_bases_are_monotonic_and_match_edges() -> None:
    """Lookup tables must be sorted and have len(bases) == len(edges) + 1."""
    from fastbootstrap.constants import BATCH_SIZE_BASES, BATCH_SIZE_EDGES

    assert list(BATCH_SIZE_EDGES) == sorted(BATCH_SIZE_EDGES)
    assert list(BATCH_SIZE_BASES) == sorted(BATCH_SIZE_BASES)
    assert len(BATCH_SIZE_BASES) == len(BATCH_SIZE_EDGES) + 1


@pytest.mark.parametrize(
    ("n_samples", "n_jobs", "memory_gb", "expected"),
    [
        # 16 workers, ample RAM: cap = N // 64 binds at 10K (156), bases win above.
        (10_000, 16, 1024.0, 156),
        (100_000, 16, 1024.0, BATCH_SIZE_MEDIUM),
        (500_000, 16, 1024.0, BATCH_SIZE_LARGE),
        (1_000_000, 16, 1024.0, BATCH_SIZE_MASSIVE),
        # 4 workers: cap = N // 16 never binds at these scales.
        (10_000, 4, 1024.0, BATCH_SIZE_SMALL),
        (100_000, 4, 1024.0, BATCH_SIZE_MEDIUM),
        # RAM tiers pinned as *literals* so a retune must consciously touch
        # this test (the tier tests below only check the min() mechanism).
        (1_000_000, 16, 6.0, 512),  # moderate RAM: 1000 -> BATCH_SIZE_MEDIUM
        (1_000_000, 16, 1.0, 64),  # low RAM:      1000 -> LOW_MEM_BATCH_CAP
        # Tiny N with a single worker is cap-bound since the 1.8.6 retune:
        # base 256 > N // (1 * 4) = 250.
        (1_000, 1, 1024.0, 250),
    ],
)
def test_reference_picks(
    n_samples: int, n_jobs: int, memory_gb: float, expected: int
) -> None:
    """Pin the picks documented in the README benchmark tables."""
    batch = _compute_optimal_batch_size(
        n_samples, sample_size=2_000, n_jobs=n_jobs, available_memory_gb=memory_gb
    )
    assert batch == expected


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


# ---------------------------------------------------------------------------
# bootstrap_resampling: argument validation surfaces as ValidationError
# ---------------------------------------------------------------------------


def _constant_statistic(rng: np.random.Generator) -> float:
    """Trivial sample function for validation tests."""
    return 0.0


@pytest.mark.parametrize("bad", [0, -5, "foo", 1.5, True, [16]])
def test_invalid_batch_size_raises_validation_error(bad: object) -> None:
    """Bad batch_size must not be re-wrapped as a NumericalError by joblib."""
    with pytest.raises(ValidationError, match="batch_size"):
        bootstrap_resampling(_constant_statistic, 10, n_jobs=1, batch_size=bad)


@pytest.mark.parametrize("batch_size", [None, "auto", "smart", 7, np.int64(7)])
def test_valid_batch_size_values_accepted(batch_size: object) -> None:
    """None, 'auto', 'smart', positive ints and NumPy ints are all legal.

    NumPy integers were accepted by joblib before validation was added, so
    rejecting them would be a regression.
    """
    out = bootstrap_resampling(_constant_statistic, 10, n_jobs=1, batch_size=batch_size)
    assert out.shape == (10,)


@pytest.mark.parametrize("batch_size", [None, 7, "smart"])
def test_n_jobs_zero_raises_validation_error_in_all_modes(
    batch_size: object,
) -> None:
    """n_jobs=0 must fail identically whether or not smart mode is used."""
    with pytest.raises(ValidationError, match="n_jobs"):
        bootstrap_resampling(_constant_statistic, 10, n_jobs=0, batch_size=batch_size)


@pytest.mark.parametrize("batch_size", [None, 7, "smart"])
def test_n_jobs_none_is_accepted_end_to_end(batch_size: object) -> None:
    """Regression: n_jobs=None worked before validation was added (joblib
    treats it as "use the parallel_backend context, else 1") and must keep
    working in every batch_size mode."""
    out = bootstrap_resampling(
        _constant_statistic, 10, n_jobs=None, batch_size=batch_size
    )
    assert out.shape == (10,)


def test_n_jobs_none_accepted_by_public_api() -> None:
    """The public entry points advertise ``Optional[int]`` for n_jobs; make sure
    ``None`` is honoured there too, including inside a parallel_backend."""
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=50), rng.normal(0.2, size=50)
    with parallel_backend("loky", n_jobs=2):
        one = one_sample_bootstrap(x, number_of_bootstrap_samples=20, n_jobs=None)
        two = two_sample_bootstrap(x, y, number_of_bootstrap_samples=20, n_jobs=None)
        uni = bootstrap(x, y, number_of_bootstrap_samples=20, n_jobs=None)
    assert one["confidence_interval"].shape == (2,)
    assert 0.0 <= two["p_value"] <= 1.0
    assert 0.0 <= uni["p_value"] <= 1.0


# ---------------------------------------------------------------------------
# ResourceWarning attribution
# ---------------------------------------------------------------------------


def test_resource_warning_points_at_caller(monkeypatch: pytest.MonkeyPatch) -> None:
    """The peak-memory warning must be attributed to user code, not core.py."""
    # Pretend the host has ~1 MiB of free RAM so the check fires for n=100K.
    monkeypatch.setattr(
        core.psutil,
        "virtual_memory",
        lambda: types.SimpleNamespace(available=2**20),
    )
    sample = np.random.default_rng(0).normal(size=100_000)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        one_sample_bootstrap(
            sample, number_of_bootstrap_samples=20, n_jobs=1, batch_size="smart"
        )
    resource = [w for w in caught if issubclass(w.category, ResourceWarning)]
    assert len(resource) == 1
    assert resource[0].filename == __file__


def test_package_prefix_does_not_match_sibling_paths() -> None:
    """skip_file_prefixes is a startswith match: the prefix must end with a
    separator so that ``<pkg>_bench.py`` next to the package is not skipped."""
    assert core._PACKAGE_DIR.endswith(os.sep)
    pkg_root = core._PACKAGE_DIR.rstrip(os.sep)
    assert core.__file__.startswith(core._PACKAGE_DIR)
    assert not (pkg_root + "_bench.py").startswith(core._PACKAGE_DIR)
