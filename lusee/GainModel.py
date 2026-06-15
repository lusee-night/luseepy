"""Spectrometer gain model: convert raw counts to physical units (nV/sqrt(Hz)).

Real LuSEE-Night observations arrive in raw spectrometer counts.  A linear
PCA model turns a handful of telemetry values into a per-frequency gain
spectrum, and the raw counts are then converted to noise spectral density.

The small coefficient artifacts are vendored under ``lusee/data/gain``
(``{L,M,H}{0-3}_{mean,eigvecs,freqs}.npy`` plus ``alpha_refit.csv``), so
nothing here depends on the external gain_model repo.

Model (per gain setting: level L/M/H, channel 0-3)::

    gain(f) = mean(f) + PC1 * eigvec1(f) + PC2 * eigvec2(f)

PC1 and PC2 are predicted from telemetry by a quadratic regression (the
coefficients in ``alpha_refit.csv``).  The gain is defined at 16 anchor
frequencies and interpolated to the target grid with a cubic spline (no
extrapolation: frequencies outside the anchor range become NaN).

Conversion to nV/sqrt(Hz)::

    auto  (products 0-3) :  Xhat = sqrt(X / G)
    cross (products 4-15):  Xhat = sign(X) * sqrt(|X| / sqrt(G_a * G_b))

Invalid bins (non-finite or non-positive gain, negative auto power, or
out-of-range frequencies) come out as NaN.  Results are returned as
LabeledArrays tagged ``units="nV/sqrt(Hz)"``.
"""

import csv
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

from .LabeledArray import label, asarray, frame_of

# Frequency spacing of the spectrometer's 2048 output bins, in MHz.
CHANNEL_BIN_MHZ = 0.025

# Two spectral-density conventions used in luseepy:
#   - the gain-model / real-data side works in an amplitude spectral density
#     (ASD), nV/sqrt(Hz);
#   - the simulator side (Data["...V"], MonoSkyModels, Throughput.noise) works
#     in a power spectral density (PSD), V^2/Hz.
# They differ by a square and the nV<->V scale.  The canonical pairing below is
# the current choice (pending domain-expert confirmation of which to
# standardize on); asd_to_psd / psd_to_asd bridge the two.
NV_PER_SQRT_HZ = "nV/sqrt(Hz)"   # amplitude spectral density (ASD)
V2_PER_HZ = "V^2/Hz"             # power spectral density (PSD)
_NV_IN_V = 1e-9                  # 1 nV expressed in volts

GAIN_LEVELS = ("L", "M", "H")

# Product index -> (channel_i, channel_j) for the 12 cross-correlation
# products.  Each cross-correlation contributes two products (real, imag)
# that share the same channel pair.  Auto-correlations are products 0-3
# (product index == channel).
CROSS_PRODUCT_CHANNELS = {
    4: (0, 1), 5: (0, 1),
    6: (0, 2), 7: (0, 2),
    8: (0, 3), 9: (0, 3),
    10: (1, 2), 11: (1, 2),
    12: (1, 3), 13: (1, 3),
    14: (2, 3), 15: (2, 3),
}


def bin_frequencies(nbins):
    """Return the spectrometer bin centre frequencies in MHz for ``nbins`` bins."""
    return np.arange(nbins) * CHANNEL_BIN_MHZ


def counts_to_nv_auto(counts, gain):
    """Convert auto-correlation counts to nV/sqrt(Hz): ``sqrt(X / G)``.

    Bins where the gain is non-finite/non-positive or the power is negative
    are returned as NaN.  Returns a bare numpy array.
    """
    X, G = np.broadcast_arrays(np.asarray(counts, dtype=float),
                               np.asarray(gain, dtype=float))
    out = np.full(X.shape, np.nan)
    valid = np.isfinite(X) & np.isfinite(G) & (G > 0.0) & (X >= 0.0)
    out[valid] = np.sqrt(X[valid] / G[valid])
    return out


def counts_to_nv_cross(counts, gain_a, gain_b):
    """Convert cross-correlation counts to nV/sqrt(Hz).

    ``sign(X) * sqrt(|X| / sqrt(G_a * G_b))`` (the sign of the real/imag
    component is preserved).  Bins with non-finite/non-positive geometric
    gain are returned as NaN.  Returns a bare numpy array.
    """
    g_geom = np.sqrt(np.asarray(gain_a, dtype=float) * np.asarray(gain_b, dtype=float))
    X, g_geom = np.broadcast_arrays(np.asarray(counts, dtype=float), g_geom)
    out = np.full(X.shape, np.nan)
    valid = np.isfinite(X) & np.isfinite(g_geom) & (g_geom > 0.0)
    out[valid] = np.sign(X[valid]) * np.sqrt(np.abs(X[valid]) / g_geom[valid])
    return out


def asd_to_psd(x):
    """Convert an amplitude spectral density to a power spectral density.

    From nV/sqrt(Hz) (the gain-model convention) to V^2/Hz (the simulator
    convention)::

        PSD[V^2/Hz] = (ASD[nV/sqrt(Hz)] * 1e-9)^2

    The sign is preserved (so signed cross-correlation Re/Im components round
    trip through psd_to_asd).  Operates on real arrays -- a complex spectrum
    should be converted component by component; passing one raises ValueError.

    :param x: ASD values (array or LabeledArray); any units label is ignored,
        the input is assumed to be in nV/sqrt(Hz).
    :returns: LabeledArray in V^2/Hz (the input's frame label is preserved).
    """
    a = np.asarray(asarray(x))
    if np.iscomplexobj(a):
        raise ValueError("asd_to_psd operates on real arrays; convert the real "
                         "and imaginary components separately")
    a = a.astype(float)
    psd = np.sign(a) * (np.abs(a) * _NV_IN_V) ** 2
    return label(psd, units=V2_PER_HZ, frame=frame_of(x))


def psd_to_asd(x):
    """Convert a power spectral density to an amplitude spectral density.

    From V^2/Hz (the simulator convention) to nV/sqrt(Hz) (the gain-model
    convention)::

        ASD[nV/sqrt(Hz)] = sqrt(PSD[V^2/Hz]) / 1e-9

    The sign is preserved (inverse of asd_to_psd).  Operates on real arrays --
    a complex spectrum should be converted component by component; passing one
    raises ValueError.

    :param x: PSD values (array or LabeledArray); any units label is ignored,
        the input is assumed to be in V^2/Hz.
    :returns: LabeledArray in nV/sqrt(Hz) (the input's frame label is preserved).
    """
    p = np.asarray(asarray(x))
    if np.iscomplexobj(p):
        raise ValueError("psd_to_asd operates on real arrays; convert the real "
                         "and imaginary components separately")
    p = p.astype(float)
    asd = np.sign(p) * np.sqrt(np.abs(p)) / _NV_IN_V
    return label(asd, units=NV_PER_SQRT_HZ, frame=frame_of(x))


class SpectrometerGain:
    """Predict gain spectra and convert raw counts to physical units.

    :param model_dir: Directory holding the vendored gain artifacts.  Defaults
        to the ``LUSEE_GAIN_DIR`` environment variable if set, else the
        ``lusee/data/gain`` directory bundled with the package.
    """

    def __init__(self, model_dir=None):
        if model_dir is None:
            model_dir = os.environ.get("LUSEE_GAIN_DIR")
        if model_dir is None:
            model_dir = Path(__file__).resolve().parent / "data" / "gain"
        self.model_dir = Path(model_dir).expanduser().resolve()

        alpha_path = self.model_dir / "alpha_refit.csv"
        if not alpha_path.exists():
            raise FileNotFoundError(f"alpha_refit.csv not found at {alpha_path}")

        # _alphas[gain_key][pc] = list of (term, alpha).  Only the quadratic
        # model rows are used (these define which terms actually contribute).
        self._alphas = defaultdict(lambda: defaultdict(list))
        with open(alpha_path, newline="") as fh:
            for row in csv.DictReader(fh):
                if row["model"] != "quadratic":
                    continue
                self._alphas[row["gain_setting"]][row["component"]].append(
                    (row["term"], float(row["alpha_refit"]))
                )

        self._npy_cache = {}

    # -- gain prediction ----------------------------------------------------
    def predict_gain(self, level, channel, telemetry, freqs_mhz=None):
        """Predict the gain spectrum for one (level, channel).

        :param level: Gain level 'L', 'M', or 'H' (case-insensitive).
        :param channel: Channel index 0-3.
        :param telemetry: dict with THERM_FPGA, SPE_ADC0_T (channels 0/1) or
            SPE_ADC1_T (channels 2/3), SPE_1VAD8_V, VMON_1V2D, SPE_1VAD8_C.
        :param freqs_mhz: Target frequencies (MHz).  If None, returns the gain
            at the 16 anchor frequencies; otherwise cubic-spline interpolated
            (no extrapolation, so out-of-range bins are NaN).
        :returns: gain array (numpy).
        """
        key = self._gain_key(level, channel)
        mean_vec, eigvecs, anchor = self._load_npy(key)
        feats = self._features(channel, telemetry)
        pc1 = self._predict_pc(key, "PC1", feats)
        pc2 = self._predict_pc(key, "PC2", feats)
        gain = mean_vec + pc1 * eigvecs[:, 0] + pc2 * eigvecs[:, 1]
        if freqs_mhz is None:
            return gain
        spline = CubicSpline(anchor, gain, extrapolate=False)
        return np.asarray(spline(np.asarray(freqs_mhz, dtype=float)), dtype=float)

    def anchor_freqs(self, level, channel):
        """Return the 16 anchor frequencies (MHz) for one (level, channel)."""
        return self._load_npy(self._gain_key(level, channel))[2]

    # -- conversion ---------------------------------------------------------
    def convert_product(self, product, counts, telemetry, levels, freqs_mhz=None):
        """Convert one product's raw counts to nV/sqrt(Hz).

        :param product: Product index 0-15 (0-3 auto, 4-15 cross Re/Im pairs).
        :param counts: Raw spectrometer counts for this product (1D array).
        :param telemetry: Telemetry dict (see predict_gain).
        :param levels: Per-channel gain levels -- a length-4 sequence or a
            dict {channel: level} giving 'L'/'M'/'H' for each channel.
        :param freqs_mhz: Bin frequencies (MHz).  Defaults to the standard
            0.025 MHz spectrometer grid for ``len(counts)`` bins.
        :returns: LabeledArray with units "nV/sqrt(Hz)".
        """
        counts = np.asarray(counts, dtype=float)
        if freqs_mhz is None:
            freqs_mhz = bin_frequencies(counts.shape[-1])

        def gain_for(ch):
            return self.predict_gain(self._level_of(levels, ch), ch, telemetry, freqs_mhz)

        return label(self._convert_one(product, counts, gain_for), units=NV_PER_SQRT_HZ)

    def convert_row(self, spectra, telemetry, levels, freqs_mhz=None):
        """Convert a full set of products for one telemetry row to nV/sqrt(Hz).

        Each channel's gain spectrum is predicted once and reused across the
        autos/crosses that use it (4 gain predictions instead of one per
        product), which matters for bulk HDF5 conversion.

        :param spectra: Raw counts shaped (nproduct, nfreq); product order is
            the standard layout (0-3 auto, 4-15 cross Re/Im pairs).
        :param telemetry: Telemetry dict (see predict_gain).
        :param levels: Per-channel gain levels (length-4 sequence or dict).
        :param freqs_mhz: Bin frequencies (MHz); defaults to the 0.025 MHz grid.
        :returns: LabeledArray (nproduct, nfreq) with units "nV/sqrt(Hz)".
        """
        spectra = np.asarray(spectra, dtype=float)
        if freqs_mhz is None:
            freqs_mhz = bin_frequencies(spectra.shape[-1])

        cache = {}

        def gain_for(ch):
            if ch not in cache:
                cache[ch] = self.predict_gain(self._level_of(levels, ch), ch,
                                              telemetry, freqs_mhz)
            return cache[ch]

        out = np.stack([self._convert_one(p, spectra[p], gain_for)
                        for p in range(spectra.shape[0])])
        return label(out, units=NV_PER_SQRT_HZ)

    @staticmethod
    def _convert_one(product, counts, gain_for):
        """Convert one product's counts to nV/sqrt(Hz) (bare array).

        ``gain_for(channel)`` returns that channel's gain spectrum; the caller
        controls whether it is recomputed or cached.
        """
        if product <= 3:
            return counts_to_nv_auto(counts, gain_for(product))
        i, j = CROSS_PRODUCT_CHANNELS[product]
        return counts_to_nv_cross(counts, gain_for(i), gain_for(j))

    # -- internals ----------------------------------------------------------
    @staticmethod
    def _gain_key(level, channel):
        lvl = str(level).strip().upper()
        if lvl not in GAIN_LEVELS:
            raise ValueError(f"level must be one of {GAIN_LEVELS}; got {level!r}")
        ch = int(channel)
        if ch not in (0, 1, 2, 3):
            raise ValueError(f"channel must be 0-3; got {channel!r}")
        return f"{lvl}{ch}"

    @staticmethod
    def _level_of(levels, channel):
        if isinstance(levels, dict):
            return levels[channel]
        return levels[channel]

    def _load_npy(self, key):
        if key not in self._npy_cache:
            d = self.model_dir
            self._npy_cache[key] = (
                np.load(d / f"{key}_mean.npy"),
                np.load(d / f"{key}_eigvecs.npy"),
                np.load(d / f"{key}_freqs.npy"),
            )
        return self._npy_cache[key]

    @staticmethod
    def _features(channel, telemetry):
        adc_key = "SPE_ADC0_T" if int(channel) <= 1 else "SPE_ADC1_T"
        required = ["THERM_FPGA", adc_key, "SPE_1VAD8_V", "VMON_1V2D", "SPE_1VAD8_C"]
        missing = [k for k in required if k not in telemetry or telemetry[k] is None]
        if missing:
            raise ValueError(f"missing telemetry keys for channel {channel}: {missing}")
        T = float(telemetry["THERM_FPGA"])
        Tc = float(telemetry[adc_key])
        return {
            "1": 1.0,
            "THERM_FPGA": T,
            adc_key: Tc,
            "SPE_1VAD8_V": float(telemetry["SPE_1VAD8_V"]),
            "VMON_1V2D": float(telemetry["VMON_1V2D"]),
            "SPE_1VAD8_C": float(telemetry["SPE_1VAD8_C"]),
            "THERM_FPGA*THERM_FPGA": T * T,
            f"{adc_key}*{adc_key}": Tc * Tc,
            f"THERM_FPGA*{adc_key}": T * Tc,
        }

    def _predict_pc(self, key, pc, feats):
        return sum(alpha * feats.get(term, 0.0) for term, alpha in self._alphas[key][pc])
