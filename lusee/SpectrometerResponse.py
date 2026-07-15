import os
from functools import lru_cache

import numpy as np
from scipy.interpolate import interp1d

# Column layout of data/spectrometer_bin_response_normed.txt (see its header row):
#   0: dbin, 1: fbin (frequency offset in Hz),
#   2: pwr, 3: notch4, 4: notch16, 5: notch64  -> the four notch responses,
#   6..69: zm0..zm63                            -> the 64 zoom-bin responses.
_DATA_FILE = os.path.join(
    os.path.dirname(__file__), "data/spectrometer_bin_response_normed.txt"
)
_N_COLUMNS = 70
_NOTCH_COLUMNS = {0: 2, 4: 3, 16: 4, 64: 5}
_ZOOM_COL_START = 6
_N_ZOOM_BINS = 64


@lru_cache(maxsize=1)
def _load_cache():
    """Load and cache the spectrometer response interpolators.

    The file is read and the cubic interpolators are built only on the first
    call; subsequent calls return the same cached dict for free (via lru_cache).

    Returns a dict with keys:
        "freq"          : the frequency-offset grid (Hz),
        "response"      : {notch value -> interpolator} for notch in [0, 4, 16, 64],
        "response_zoom" : list of 64 interpolators, one per zoom bin.
    """
    data = np.loadtxt(_DATA_FILE, skiprows=1)  # first row is a text column header
    assert data.shape[1] == _N_COLUMNS, (
        f"Expected {_N_COLUMNS} columns in the response file, got {data.shape[1]}"
    )
    freq = data[:, 1]

    def make_interp(col):
        return interp1d(
            freq, data[:, col], kind="cubic", fill_value=0.0, bounds_error=False
        )

    response = {notch: make_interp(col) for notch, col in _NOTCH_COLUMNS.items()}
    response_zoom = [
        make_interp(_ZOOM_COL_START + i) for i in range(_N_ZOOM_BINS)
    ]
    return {"freq": freq, "response": response, "response_zoom": response_zoom}


def spectrometer_response(delta_freq_Hz, notch=0):
    """
    Returns the spectrometer response for a given frequency offset in Hz.

    Parameters:
    delta_freq_Hz : float or array-like
        Frequency offset in Hz for which to compute the response.
    notch : int, optional
        Notch filter value (0, 4, 16, 64) to apply. Default is 0 (no notch).

    Returns:
    response : float or array-like
        The spectrometer response at the specified frequency offset(s) in units
        1/Hz. Normalized to integrate to 1.
    """
    responses = _load_cache()["response"]
    try:
        interp = responses[notch]
    except KeyError:
        raise ValueError(
            f"Invalid notch value {notch!r}. Must be one of {sorted(responses)}."
        )
    return interp(delta_freq_Hz)


def spectrometer_response_zoom(delta_freq_Hz, zoom_bin):
    """
    Returns the spectrometer response for a given frequency offset in Hz for a
    specific zoom bin.

    Parameters:
    delta_freq_Hz : float or array-like
        Frequency offset in Hz for which to compute the response.
    zoom_bin : int
        Zoom bin index to apply. Can go 0-63 or -32 to 31, with negative
        indices wrapping around (e.g. -32 == 32 being Nyquist).

    Returns:
    response : float or array-like
        The spectrometer response at the specified frequency offset(s) in units
        1/Hz. Normalized to integrate to 1.
    """
    if not (-32 <= zoom_bin < _N_ZOOM_BINS):
        raise ValueError(
            f"Invalid zoom_bin value {zoom_bin!r}. Must be in the range [-32, {_N_ZOOM_BINS - 1}]."
        )
    # Negative indices wrap into [0, 64); for non-negative bins this is a no-op.
    return _load_cache()["response_zoom"][zoom_bin % _N_ZOOM_BINS](delta_freq_Hz)
