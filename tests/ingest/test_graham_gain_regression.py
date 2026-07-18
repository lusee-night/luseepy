"""Optional end-to-end regression on Graham's September 2025 sessions.

Run explicitly with::

    LUSEE_GRAHAM_H5_DIR=/path/to/h5_from_json_new \
      python -m pytest -q tests/ingest/test_graham_gain_regression.py

The fixture is intentionally external to the repository, so this test skips
cleanly in CI and on developer machines that do not have the sessions.
"""

from __future__ import annotations

import hashlib
import os
import warnings
from pathlib import Path

import numpy as np
import pytest

import lusee.GainModel as gm


graham_env = os.environ.get("LUSEE_GRAHAM_H5_DIR")
GRAHAM_DIR = Path(graham_env) if graham_env else None

pytestmark = pytest.mark.skipif(
    GRAHAM_DIR is None or not GRAHAM_DIR.is_dir(),
    reason="set LUSEE_GRAHAM_H5_DIR to the Graham HDF5 session directory to run",
)


# Statistics of the time-mean auto spectra in the inclusive 1--50 MHz band:
# min, 1%, median, 99%, 99.9%, max. These were independently generated with
# the frozen pre-refactor evaluator plus direct CSV family selection.
SUMMARY_GOLDEN = {
    ("quadratic", "quadratic"): np.array([
        2.5157836160073725,
        2.5633655418281560,
        3.6108100707389923,
        5.6763159640464350,
        6.7297901659819440,
        10.244074835950290,
    ]),
    ("quadratic", "linear"): np.array([
        2.5155339989850707,
        2.5644780705045678,
        3.6155916704553370,
        5.7084006845709780,
        6.7776705304261760,
        10.455947722961284,
    ]),
}


# session_science_000, time-mean auto channel 0, nearest bins to these MHz.
REPRESENTATIVE_FREQS_MHZ = np.array([1, 5, 10, 20, 30, 40, 50], dtype=float)
SESSION_000_GOLDEN = {
    ("quadratic", "quadratic"): np.array([
        2.5891213414, 2.5457473135, 2.6126874512, 2.8266543028,
        3.6758771647, 4.9478456516, 4.8656674126,
    ]),
    ("quadratic", "linear"): np.array([
        2.5885691210, 2.5449122700, 2.6126018523, 2.8262074847,
        3.6754635915, 4.9492721763, 4.8690510583,
    ]),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_v2(path: Path):
    from lusee.ingest import load

    with pytest.warns(RuntimeWarning, match="layout-v2 spectra were bit-slice restored"):
        # legacy files carry no time_scale constant; these sessions used
        # uncrater's Unix-epoch .time, which is UTC-flavored
        return load(path, assume_scale="utc")


def _representative_auto0(data, physical) -> np.ndarray:
    with warnings.catch_warnings():
        # The gain artifact deliberately does not extrapolate below its first
        # 0.1-MHz anchor, so those all-NaN bins are expected here.
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(np.asarray(physical)[:, 0, :], axis=0)
    indices = [int(np.argmin(np.abs(data.freq - f)))
               for f in REPRESENTATIVE_FREQS_MHZ]
    return mean[indices]


def _summary_for_models(files, pc1: str, pc2: str) -> np.ndarray:
    gm.set_models(pc1=pc1, pc2=pc2)
    values = []
    for path in files:
        data = _load_v2(path)
        assert data.spectra.units == "SDU"
        assert data.metadata["actual_bitslice"].shape == (data.Nspectra, 16)
        physical = np.asarray(data.to_physical())[:, :4, :]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean = np.nanmean(physical, axis=0)
        in_band = (data.freq >= 1.0) & (data.freq <= 50.0)
        values.append(mean[:, in_band].ravel())

    values = np.concatenate(values)
    return np.array([
        np.nanmin(values),
        np.nanquantile(values, 0.01),
        np.nanmedian(values),
        np.nanquantile(values, 0.99),
        np.nanquantile(values, 0.999),
        np.nanmax(values),
    ])


def test_graham_qq_ql_golden_stale_cache_and_nonpersistence():
    files = sorted(GRAHAM_DIR.glob("session_science_0*.h5"))
    # _018 is the documented empty/aborted session and is not convertible.
    files = [path for path in files if path.stem != "session_science_018"]
    assert len(files) == 18
    session_000 = GRAHAM_DIR / "session_science_000.h5"
    checksum_before = _sha256(session_000)
    models_before = gm.get_models()

    try:
        # Reuse both the IngestData object and its internally cached gain model.
        # A selector change must still affect the very next conversion.
        data_000 = _load_v2(session_000)
        gm.set_models(pc1="quadratic", pc2="quadratic")
        qq = data_000.to_physical()
        gm.set_models(pc1="quadratic", pc2="linear")
        ql = data_000.to_physical()
        np.testing.assert_allclose(
            _representative_auto0(data_000, qq),
            SESSION_000_GOLDEN[("quadratic", "quadratic")],
            rtol=2e-9,
            atol=2e-9,
        )
        np.testing.assert_allclose(
            _representative_auto0(data_000, ql),
            SESSION_000_GOLDEN[("quadratic", "linear")],
            rtol=2e-9,
            atol=2e-9,
        )
        assert not np.array_equal(np.asarray(qq), np.asarray(ql), equal_nan=True)

        for selection, expected in SUMMARY_GOLDEN.items():
            actual = _summary_for_models(files, *selection)
            np.testing.assert_allclose(actual, expected, rtol=2e-9, atol=2e-9)
    finally:
        gm.set_models(**models_before)

    # Loading and both requested conversions are read-only operations.
    assert _sha256(session_000) == checksum_before
