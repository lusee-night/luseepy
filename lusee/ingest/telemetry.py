"""Public hook that delegates DCB / encoder telemetry decoding.

The byte layout, channel names, and engineering-unit formulas for the
LuSEE-Night DCB and encoder telemetry packets are restricted and live
in a separate private package, ``lusee_telemetry``. luseepy itself is
public and contains no telemetry-specific knowledge.

This module discovers ``lusee_telemetry`` at runtime:

1. If ``LUSEE_TELEMETRY_PATH`` is set, that directory is prepended to
   ``sys.path``.
2. ``import lusee_telemetry`` is then attempted.
3. On success, every public function below delegates to it.
4. On failure, every public function returns an empty / None result and
   the pipeline simply produces an HDF5 / FITS without ``/DCB_telemetry/``.

This mirrors the pattern used for ``uncrater`` / ``UNCRATER_PATH``.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy loader for the private decoder
# ---------------------------------------------------------------------------

_decoder: Optional[Any] = None
_resolved: bool = False


def _try_import_decoder():
    """Import lusee_telemetry; return the module or None.

    Caches the result; the env-var path injection only happens once.
    """
    global _decoder, _resolved
    if _resolved:
        return _decoder
    path = os.environ.get("LUSEE_TELEMETRY_PATH")
    if path and path not in sys.path:
        sys.path.insert(0, path)
    try:
        import lusee_telemetry  # type: ignore[import-not-found]
        _decoder = lusee_telemetry
        log.info("loaded telemetry decoder from %s",
                 getattr(lusee_telemetry, "__file__", "<unknown>"))
    except ImportError as exc:
        log.info("no telemetry decoder available (%s); /DCB_telemetry/ "
                 "will not be written", exc)
        _decoder = None
    _resolved = True
    return _decoder


def has_decoder() -> bool:
    """True iff a private telemetry decoder is loadable."""
    return _try_import_decoder() is not None


# ---------------------------------------------------------------------------
# Public proxy API (matches the contract in lusee_telemetry's README)
# ---------------------------------------------------------------------------

def telemetry_apids() -> Tuple[int, ...]:
    """APIDs that the decoder claims (e.g. ``(0x314, 0x325)``); ``()`` if absent."""
    dec = _try_import_decoder()
    if dec is None:
        return ()
    return tuple(dec.telemetry_apids())


def parse_b01_packets(
    logical_packets: Iterable[Any],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Decode b01 telemetry logical packets into ``(fpga_arrays, encoder_arrays)``.

    Returns ``({}, {})`` when no decoder is available.
    """
    dec = _try_import_decoder()
    if dec is None:
        return {}, {}
    return dec.parse_b01_packets(logical_packets)


def find_legacy_sidecar(session_dir) -> Optional[Path]:
    """Locate a legacy telemetry sidecar in ``session_dir``."""
    dec = _try_import_decoder()
    if dec is None:
        return None
    return dec.find_legacy_sidecar(session_dir)


def parse_legacy_sidecar(path) -> Dict[str, np.ndarray]:
    """Decode a legacy telemetry sidecar into per-field FPGA arrays."""
    dec = _try_import_decoder()
    if dec is None:
        return {}
    return dec.parse_legacy_sidecar(path)


def field_groups() -> Dict[str, Tuple[str, ...]]:
    """Channel-name groups for plot panel layout. Empty if no decoder."""
    dec = _try_import_decoder()
    if dec is None or not hasattr(dec, "field_groups"):
        return {}
    return dec.field_groups()


# ---------------------------------------------------------------------------
# Generic helpers that don't depend on the decoder
# ---------------------------------------------------------------------------

def slice_arrays_by_window(
    arrays: Dict[str, np.ndarray],
    *,
    window_lower_raw_seconds: Optional[float] = None,
    window_upper_raw_seconds: Optional[float] = None,
    time_key: str = "mission_seconds",
) -> Dict[str, np.ndarray]:
    """Filter every array in ``arrays`` to ``[lower, upper)`` over ``time_key``.

    ``None`` at either end means -inf / +inf. If ``arrays`` does not
    contain ``time_key``, the input is returned unchanged.
    """
    if not arrays or time_key not in arrays:
        return arrays
    times = arrays[time_key]
    if times.size == 0:
        return arrays
    mask = np.ones(times.size, dtype=bool)
    if window_lower_raw_seconds is not None:
        mask &= times >= window_lower_raw_seconds
    if window_upper_raw_seconds is not None:
        mask &= times < window_upper_raw_seconds
    if mask.all():
        return arrays
    return {k: v[mask] for k, v in arrays.items()}
