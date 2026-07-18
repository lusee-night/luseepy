"""Spectrometer gain model: convert spectrometer SDU to physical units.

Real LuSEE-Night observations arrive in raw spectrometer counts.  A PCA
reconstruction model turns a handful of telemetry values into a per-frequency
gain spectrum, and the raw counts are then converted to noise spectral density.

The small coefficient artifacts are vendored under ``lusee/data/gain``
(``{L,M,H}{0-3}_{mean,eigvecs,freqs}.npy`` plus ``alpha_refit.csv``), so
nothing here depends on the external gain_model repo.

Model (per gain setting: level L/M/H, channel 0-3)::

    gain(f) = mean(f) + PC1 * eigvec1(f) + PC2 * eigvec2(f)

PC1 and PC2 are predicted from telemetry by independently selectable linear
or quadratic regressions (the coefficients in ``alpha_refit.csv``).  The
process-wide defaults can be changed with :func:`set_models`; each prediction
or conversion snapshots the selection once, so an in-flight conversion cannot
mix model families.  The gain is defined at 16 anchor frequencies and
interpolated to the target grid with a cubic spline (no extrapolation:
frequencies outside the anchor range become NaN).

Conversion to nV/sqrt(Hz)::

    auto  (products 0-3) :  Xhat = sqrt(X / G)
    cross (products 4-15):  Xhat = sign(X) * sqrt(|X| / sqrt(G_a * G_b))

Invalid bins (non-finite or non-positive gain, negative auto power, or
out-of-range frequencies) come out as NaN.  Results are returned as
LabeledArrays tagged ``units="nV/sqrt(Hz)"``.
"""

import csv
import os
from collections import OrderedDict
from pathlib import Path
from threading import RLock

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
NV2_PER_HZ = "nV^2/Hz"           # input-referred power in native gain units
V2_PER_HZ = "V^2/Hz"             # power spectral density (PSD)
_NV_IN_V = 1e-9                  # 1 nV expressed in volts

GAIN_LEVELS = ("L", "M", "H")
MODEL_FAMILIES = ("linear", "quadratic")

# Process-wide selection used by new and existing SpectrometerGain instances.
# A tuple replacement is atomic, and the lock gives the same snapshot semantics
# on Python implementations without relying on CPython's GIL.
_MODEL_LOCK = RLock()
_PC_MODELS = ("quadratic", "quadratic")


def _normalize_model(model, component):
    if not isinstance(model, str):
        raise ValueError(
            f"{component} model must be one of {MODEL_FAMILIES}; got {model!r}"
        )
    normalized = model.strip().lower()
    if normalized not in MODEL_FAMILIES:
        raise ValueError(
            f"{component} model must be one of {MODEL_FAMILIES}; got {model!r}"
        )
    return normalized


def set_models(*, pc1, pc2):
    """Select the regression family used for PC1 and PC2 process-wide.

    Both arguments are required and may be ``"linear"`` or ``"quadratic"``
    (case-insensitive).  The pair is replaced atomically.  Existing
    :class:`SpectrometerGain` objects observe it on their next public
    prediction/conversion call because their caches contain only
    model-independent spectral bases.

    Returns a fresh dict in the same form as :func:`get_models`.
    """
    selection = (
        _normalize_model(pc1, "pc1"),
        _normalize_model(pc2, "pc2"),
    )
    global _PC_MODELS
    with _MODEL_LOCK:
        _PC_MODELS = selection
    return {"pc1": selection[0], "pc2": selection[1]}


def get_models():
    """Return the current process-wide PC regression selection."""
    pc1, pc2 = _snapshot_models()
    return {"pc1": pc1, "pc2": pc2}


def _snapshot_models():
    """Take one immutable model-selection snapshot for an operation."""
    with _MODEL_LOCK:
        return _PC_MODELS

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
    component is preserved).  Bins where either channel gain is non-finite or
    non-positive are returned as NaN.  Returns a bare numpy array.
    """
    X, Ga, Gb = np.broadcast_arrays(
        np.asarray(counts, dtype=float),
        np.asarray(gain_a, dtype=float),
        np.asarray(gain_b, dtype=float),
    )
    out = np.full(X.shape, np.nan)
    valid = (
        np.isfinite(X) & np.isfinite(Ga) & np.isfinite(Gb)
        & (Ga > 0.0) & (Gb > 0.0)
    )
    geometric_gain = np.sqrt(Ga[valid] * Gb[valid])
    out[valid] = (
        np.sign(X[valid]) * np.sqrt(np.abs(X[valid]) / geometric_gain)
    )
    return out


def counts_to_nv2_auto(counts, gain):
    """Convert auto-correlation SDU to input-referred ``nV^2/Hz``.

    This is the physically linear PSD conversion ``X / G``.  Negative auto
    power and bins with non-finite/non-positive gain are returned as NaN.
    Returns a bare numpy array.
    """
    X, G = np.broadcast_arrays(np.asarray(counts, dtype=float),
                               np.asarray(gain, dtype=float))
    out = np.full(X.shape, np.nan)
    valid = np.isfinite(X) & np.isfinite(G) & (G > 0.0) & (X >= 0.0)
    out[valid] = X[valid] / G[valid]
    return out


def counts_to_nv2_cross(counts, gain_a, gain_b):
    """Convert a signed cross component to input-referred ``nV^2/Hz``.

    The conversion is linear in the measured real or imaginary component::

        X / sqrt(G_a * G_b)

    Both channel gains must be finite and strictly positive.  Returns a bare
    numpy array; real and imaginary cross products remain separate products.
    """
    X, Ga, Gb = np.broadcast_arrays(
        np.asarray(counts, dtype=float),
        np.asarray(gain_a, dtype=float),
        np.asarray(gain_b, dtype=float),
    )
    out = np.full(X.shape, np.nan)
    valid = (
        np.isfinite(X) & np.isfinite(Ga) & np.isfinite(Gb)
        & (Ga > 0.0) & (Gb > 0.0)
    )
    out[valid] = X[valid] / np.sqrt(Ga[valid] * Gb[valid])
    return out


def asd_to_psd(x):
    """Convert an amplitude spectral density to a power spectral density.

    From nV/sqrt(Hz) (the gain-model convention) to V^2/Hz (the simulator
    convention)::

        PSD[V^2/Hz] = sign(ASD) * (|ASD[nV/sqrt(Hz)]| * 1e-9)^2

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

        ASD[nV/sqrt(Hz)] = sign(PSD) * sqrt(|PSD[V^2/Hz]|) / 1e-9

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

        # Dense tensor layout:
        #   family x level x channel x PC x telemetry-feature.
        # Screened-out terms remain zero.  A fit labelled "quadratic" can
        # legitimately contain only first-order terms; the label selects a
        # fitted coefficient family, not a forced evaluation degree.
        self._coefficients = np.zeros((2, 3, 4, 2, 9), dtype=float)
        seen = set()
        with open(alpha_path, newline="") as fh:
            for row in csv.DictReader(fh):
                family = row["model"].strip().lower()
                if family not in MODEL_FAMILIES:
                    raise ValueError(
                        f"unknown model family {row['model']!r} in {alpha_path}"
                    )
                key = self._gain_key_from_csv(row["gain_setting"], alpha_path)
                component = row["component"].strip().upper()
                if component not in ("PC1", "PC2"):
                    raise ValueError(
                        f"unknown component {row['component']!r} in {alpha_path}"
                    )
                level_i = GAIN_LEVELS.index(key[0])
                channel = int(key[1])
                pc_i = int(component[-1]) - 1
                feature_i = self._feature_term_index(channel, row["term"])
                slot = (family, key, component, feature_i)
                if slot in seen:
                    raise ValueError(
                        f"duplicate coefficient for {family}/{key}/{component}/"
                        f"{row['term']} in {alpha_path}"
                    )
                seen.add(slot)
                self._coefficients[
                    MODEL_FAMILIES.index(family), level_i, channel, pc_i, feature_i
                ] = float(row["alpha_refit"])

        missing = [
            (family, f"{level}{channel}", component)
            for family in MODEL_FAMILIES
            for level in GAIN_LEVELS
            for channel in range(4)
            for component in ("PC1", "PC2")
            if not any(
                slot[0] == family
                and slot[1] == f"{level}{channel}"
                and slot[2] == component
                for slot in seen
            )
        ]
        if missing:
            raise ValueError(
                f"alpha_refit.csv is missing coefficient groups: {missing}"
            )
        self._coefficients.setflags(write=False)

        self._npy_cache = {}
        # Interpolation is linear in its ordinate values.  Cache the
        # interpolated [mean, eigvec1, eigvec2] basis, which is independent of
        # telemetry and the selected regression families.
        self._basis_cache = OrderedDict()
        self._cache_lock = RLock()
        self._basis_cache_limit = 48

    @staticmethod
    def _gain_key_from_csv(value, alpha_path):
        key = str(value).strip().upper()
        if len(key) != 2 or key[0] not in GAIN_LEVELS or key[1] not in "0123":
            raise ValueError(f"invalid gain_setting {value!r} in {alpha_path}")
        return key

    @staticmethod
    def _feature_term_index(channel, term):
        adc_key = "SPE_ADC0_T" if int(channel) <= 1 else "SPE_ADC1_T"
        terms = (
            "1",
            "THERM_FPGA",
            adc_key,
            "SPE_1VAD8_V",
            "VMON_1V2D",
            "SPE_1VAD8_C",
            "THERM_FPGA*THERM_FPGA",
            f"{adc_key}*{adc_key}",
            f"THERM_FPGA*{adc_key}",
        )
        try:
            return terms.index(str(term).strip())
        except ValueError as exc:
            raise ValueError(
                f"unknown gain-model term {term!r} for channel {channel}"
            ) from exc

    # -- gain prediction ----------------------------------------------------
    def predict_gain(self, level, channel, telemetry, freqs_mhz=None, *,
                     _models=None):
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
        models = _snapshot_models() if _models is None else _models
        key = self._gain_key(level, channel)
        feats = self._features(channel, telemetry)
        pc1 = self._predict_pc(key, "PC1", feats, models)
        pc2 = self._predict_pc(key, "PC2", feats, models)
        basis = self._spectral_basis(key, freqs_mhz)
        return basis[0] + pc1 * basis[1] + pc2 * basis[2]

    def predict_gain_batch(self, levels, telemetry, freqs_mhz=None, *,
                           _models=None, _nrow=None):
        """Predict all four channel gains for one or more telemetry rows.

        ``levels`` may be one per-channel setting (a length-4 sequence/dict,
        a single letter, or a four-letter string), or an ``(N, 4)`` array for
        row-varying settings. Entries may be L/M/H or ``None``; unmodelled
        ``None`` channels are returned as NaN. ``telemetry`` maps each field
        to either a scalar or a length-N one-dimensional array.

        Returns a bare ndarray shaped ``(N, 4, Nfreq)``.  The process-wide PC
        model selection is snapshotted exactly once for the whole batch.
        """
        models = _snapshot_models() if _models is None else _models
        nrow = self._infer_batch_rows(levels, telemetry, expected=_nrow)
        level_rows = self._level_rows(levels, nrow)
        if freqs_mhz is None:
            nfreq = self._load_npy("L0")[2].size
        else:
            freqs_mhz = self._validate_freqs(freqs_mhz)
            nfreq = freqs_mhz.size
        out = np.full((nrow, 4, nfreq), np.nan, dtype=float)

        for channel in range(4):
            features = self._feature_matrix(channel, telemetry, nrow)
            for level in GAIN_LEVELS:
                row_mask = level_rows[:, channel] == level
                if not np.any(row_mask):
                    continue
                key = f"{level}{channel}"
                pc1 = self._predict_pc(key, "PC1", features[row_mask], models)
                pc2 = self._predict_pc(key, "PC2", features[row_mask], models)
                basis = self._spectral_basis(key, freqs_mhz)
                out[row_mask, channel, :] = (
                    basis[0]
                    + pc1[:, None] * basis[1]
                    + pc2[:, None] * basis[2]
                )
        return out

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
        :returns: LabeledArray with units "nV/sqrt(Hz)" (the input's frame
            label, if any, is preserved).
        """
        models = _snapshot_models()
        frame = frame_of(counts)
        counts = np.asarray(counts, dtype=float)
        if counts.ndim == 0:
            raise ValueError("counts must have a frequency axis")
        self._validate_product(product)
        if freqs_mhz is None:
            freqs_mhz = bin_frequencies(counts.shape[-1])
        else:
            freqs_mhz = self._validate_freqs(freqs_mhz, counts.shape[-1])

        def gain_for(ch):
            return self.predict_gain(
                self._level_of(levels, ch), ch, telemetry, freqs_mhz,
                _models=models,
            )

        return label(self._convert_one(product, counts, gain_for),
                     units=NV_PER_SQRT_HZ, frame=frame)

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
        :returns: LabeledArray (nproduct, nfreq) with units "nV/sqrt(Hz)" (the
            input's frame label, if any, is preserved).
        """
        models = _snapshot_models()
        frame = frame_of(spectra)
        spectra = np.asarray(spectra, dtype=float)
        if spectra.ndim != 2:
            raise ValueError(
                f"spectra must have shape (nproduct, nfreq); got {spectra.shape}"
            )
        if not 1 <= spectra.shape[0] <= 16:
            raise ValueError("spectra must contain between 1 and 16 products")
        if freqs_mhz is None:
            freqs_mhz = bin_frequencies(spectra.shape[-1])
        else:
            freqs_mhz = self._validate_freqs(freqs_mhz, spectra.shape[-1])

        cache = {}

        def gain_for(ch):
            if ch not in cache:
                cache[ch] = self.predict_gain(self._level_of(levels, ch), ch,
                                              telemetry, freqs_mhz,
                                              _models=models)
            return cache[ch]

        out = np.stack([self._convert_one(p, spectra[p], gain_for)
                        for p in range(spectra.shape[0])])
        return label(out, units=NV_PER_SQRT_HZ, frame=frame)

    def convert_batch(self, spectra, telemetry, levels, freqs_mhz=None, *,
                      chunk_size=None):
        """Vectorized signed-ASD conversion for a batch of spectra.

        ``spectra`` must have shape ``(N, Nproduct, Nfreq)``.  Telemetry
        fields may be length-N arrays or broadcastable scalars, and ``levels``
        accepts the forms documented by :meth:`predict_gain_batch`.  The four
        channel gains are evaluated in groups by level, then reused for all 16
        products.  Returns a :class:`LabeledArray` in ``nV/sqrt(Hz)``.

        The signed square-root convention for crosses is retained for legacy
        compatibility.  New physical analysis should generally prefer
        :meth:`convert_batch_psd`, which is linear in cross components.
        """
        frame = frame_of(spectra)
        spectra, freqs = self._prepare_batch_spectra(spectra, freqs_mhz)
        models = _snapshot_models()
        out = self._convert_prepared_batch(
            spectra, telemetry, levels, freqs, models,
            psd=False, chunk_size=chunk_size,
        )
        return label(out, units=NV_PER_SQRT_HZ, frame=frame)

    def convert_batch_psd(self, spectra, telemetry, levels, freqs_mhz=None,
                          *, units=V2_PER_HZ, chunk_size=None):
        """Vectorized, physically linear PSD conversion for all products.

        Autos use ``X/G`` and each signed cross real/imaginary component uses
        ``X/sqrt(G_a G_b)``. ``units`` may be ``"nV^2/Hz"`` (the gain model's
        native input-referred units) or ``"V^2/Hz"`` (default SI scaling).
        Returns a :class:`LabeledArray` with the requested units.
        """
        if units not in (NV2_PER_HZ, V2_PER_HZ):
            raise ValueError(f"units must be {NV2_PER_HZ!r} or {V2_PER_HZ!r}")
        frame = frame_of(spectra)
        spectra, freqs = self._prepare_batch_spectra(spectra, freqs_mhz)
        models = _snapshot_models()
        out = self._convert_prepared_batch(
            spectra, telemetry, levels, freqs, models,
            psd=True, chunk_size=chunk_size,
        )
        if units == V2_PER_HZ:
            out *= _NV_IN_V ** 2
        return label(out, units=units, frame=frame)

    def _convert_prepared_batch(self, spectra, telemetry, levels, freqs,
                                models, *, psd, chunk_size):
        nrow = spectra.shape[0]
        level_rows = self._level_rows(levels, nrow)
        telemetry_rows = self._telemetry_rows(telemetry, nrow)
        if chunk_size is None:
            chunk_size = max(nrow, 1)
        elif isinstance(chunk_size, (bool, np.bool_)) or int(chunk_size) != chunk_size \
                or int(chunk_size) <= 0:
            raise ValueError("chunk_size must be a positive integer or None")
        else:
            chunk_size = int(chunk_size)

        out = np.full(spectra.shape, np.nan, dtype=float)
        for start in range(0, nrow, chunk_size):
            stop = min(start + chunk_size, nrow)
            row_slice = slice(start, stop)
            telemetry_chunk = {
                key: values[row_slice] for key, values in telemetry_rows.items()
            }
            gains = self.predict_gain_batch(
                level_rows[row_slice], telemetry_chunk, freqs,
                _models=models, _nrow=stop - start,
            )
            out[row_slice] = self._convert_batch_values(
                spectra[row_slice], gains, psd=psd
            )
        return out

    @staticmethod
    def _convert_batch_values(spectra, gains, *, psd):
        out = np.full(spectra.shape, np.nan, dtype=float)
        auto_func = counts_to_nv2_auto if psd else counts_to_nv_auto
        cross_func = counts_to_nv2_cross if psd else counts_to_nv_cross
        for product in range(spectra.shape[1]):
            if product <= 3:
                out[:, product, :] = auto_func(
                    spectra[:, product, :], gains[:, product, :]
                )
            else:
                channel_a, channel_b = CROSS_PRODUCT_CHANNELS[product]
                out[:, product, :] = cross_func(
                    spectra[:, product, :],
                    gains[:, channel_a, :],
                    gains[:, channel_b, :],
                )
        return out

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
        # works for both dict {channel: level} and length-4 sequences
        return levels[channel]

    def _load_npy(self, key):
        with self._cache_lock:
            cached = self._npy_cache.get(key)
            if cached is not None:
                return cached

            d = self.model_dir
            mean = np.load(d / f"{key}_mean.npy", allow_pickle=False)
            eigvecs = np.load(d / f"{key}_eigvecs.npy", allow_pickle=False)
            anchor = np.load(d / f"{key}_freqs.npy", allow_pickle=False)
            if mean.ndim != 1 or anchor.shape != mean.shape:
                raise ValueError(f"invalid mean/frequency artifacts for {key}")
            if eigvecs.ndim != 2 or eigvecs.shape[0] != mean.size \
                    or eigvecs.shape[1] < 2:
                raise ValueError(f"invalid eigenvector artifact for {key}")
            if not np.all(np.diff(anchor) > 0.0):
                raise ValueError(f"anchor frequencies are not increasing for {key}")
            for value in (mean, eigvecs, anchor):
                value.setflags(write=False)
            cached = (mean, eigvecs, anchor)
            self._npy_cache[key] = cached
            return cached

    def _spectral_basis(self, key, freqs_mhz):
        mean, eigvecs, anchor = self._load_npy(key)
        if freqs_mhz is None:
            return np.vstack((mean, eigvecs[:, 0], eigvecs[:, 1]))

        freqs = self._validate_freqs(freqs_mhz)
        token = (freqs.shape, freqs.tobytes())
        cache_key = (key, token)
        with self._cache_lock:
            cached = self._basis_cache.get(cache_key)
            if cached is not None:
                self._basis_cache.move_to_end(cache_key)
                return cached

        anchor_basis = np.column_stack((mean, eigvecs[:, 0], eigvecs[:, 1]))
        basis = np.asarray(
            CubicSpline(anchor, anchor_basis, axis=0, extrapolate=False)(freqs),
            dtype=float,
        ).T
        basis.setflags(write=False)
        with self._cache_lock:
            self._basis_cache[cache_key] = basis
            self._basis_cache.move_to_end(cache_key)
            while len(self._basis_cache) > self._basis_cache_limit:
                self._basis_cache.popitem(last=False)
        return basis

    @staticmethod
    def _validate_freqs(freqs_mhz, expected_size=None):
        freqs = np.asarray(freqs_mhz, dtype=float)
        if freqs.ndim != 1:
            raise ValueError(
                f"freqs_mhz must be one-dimensional; got shape {freqs.shape}"
            )
        if expected_size is not None and freqs.size != expected_size:
            raise ValueError(
                f"frequency grid has {freqs.size} bins; expected {expected_size}"
            )
        return freqs

    @classmethod
    def _prepare_batch_spectra(cls, spectra, freqs_mhz):
        # Keep the persisted float32 SDU buffer as a view.  Conversion helpers
        # promote one product/chunk at a time, avoiding a second full-size
        # float64 input cube alongside the necessarily float64 output.
        spectra = np.asarray(spectra)
        if spectra.ndim != 3:
            raise ValueError(
                "spectra must have shape (nrow, nproduct, nfreq); "
                f"got {spectra.shape}"
            )
        if np.iscomplexobj(spectra) or not np.issubdtype(spectra.dtype, np.number):
            raise ValueError("spectra must contain real numeric SDU values")
        if not 1 <= spectra.shape[1] <= 16:
            raise ValueError("spectra must contain between 1 and 16 products")
        if freqs_mhz is None:
            freqs = bin_frequencies(spectra.shape[2])
        else:
            freqs = cls._validate_freqs(freqs_mhz, spectra.shape[2])
        return spectra, freqs

    @staticmethod
    def _validate_product(product):
        try:
            normalized = int(product)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"product must be an integer 0-15; got {product!r}") from exc
        if normalized != product or normalized not in range(16):
            raise ValueError(f"product must be an integer 0-15; got {product!r}")
        return normalized

    @staticmethod
    def _normalize_level_value(level):
        if level is None:
            return None
        if isinstance(level, (float, np.floating)) and np.isnan(level):
            return None
        if isinstance(level, (bytes, np.bytes_)):
            level = level.decode("ascii", "ignore")
        if isinstance(level, str):
            normalized = level.strip().upper()
            if normalized in GAIN_LEVELS:
                return normalized
        raise ValueError(
            f"level entries must be one of {GAIN_LEVELS} or None; got {level!r}"
        )

    @classmethod
    def _level_rows(cls, levels, nrow):
        if isinstance(levels, dict):
            try:
                constant = [levels[channel] for channel in range(4)]
            except KeyError as exc:
                raise ValueError("levels dict must contain channels 0-3") from exc
            rows = np.tile(np.asarray(constant, dtype=object), (nrow, 1))
        elif levels is None:
            rows = np.full((nrow, 4), None, dtype=object)
        elif isinstance(levels, (str, bytes, np.str_, np.bytes_)):
            if isinstance(levels, (bytes, np.bytes_)):
                levels = levels.decode("ascii", "ignore")
            text = str(levels).strip()
            if len(text) == 1:
                constant = [text] * 4
            elif len(text) == 4:
                constant = list(text)
            else:
                raise ValueError(
                    f"string levels must contain one or four letters; got {levels!r}"
                )
            rows = np.tile(np.asarray(constant, dtype=object), (nrow, 1))
        else:
            values = np.asarray(levels, dtype=object)
            if values.ndim == 1 and values.shape == (4,):
                rows = np.tile(values, (nrow, 1))
            elif values.ndim == 2 and values.shape == (nrow, 4):
                rows = values.copy()
            else:
                raise ValueError(
                    "levels must be a constant length-4 specification or have "
                    f"shape ({nrow}, 4); got {values.shape}"
                )
        for index in np.ndindex(rows.shape):
            rows[index] = cls._normalize_level_value(rows[index])
        return rows

    @staticmethod
    def _infer_batch_rows(levels, telemetry, expected=None):
        if not hasattr(telemetry, "items"):
            raise ValueError("telemetry must be a mapping")
        sizes = []
        telemetry_keys = (
            "THERM_FPGA", "SPE_ADC0_T", "SPE_ADC1_T", "SPE_1VAD8_V",
            "VMON_1V2D", "SPE_1VAD8_C",
        )
        for key in telemetry_keys:
            value = telemetry.get(key)
            if value is None:
                continue
            arr = np.asarray(value)
            if arr.ndim == 0:
                continue
            if arr.ndim != 1:
                raise ValueError(
                    "batch telemetry values must be scalars or one-dimensional; "
                    f"got shape {arr.shape}"
                )
            sizes.append(arr.size)

        if levels is not None and not isinstance(
                levels, (dict, str, bytes, np.str_, np.bytes_)):
            level_values = np.asarray(levels, dtype=object)
            if level_values.ndim == 2:
                sizes.append(level_values.shape[0])
            elif level_values.ndim > 2:
                raise ValueError(f"invalid levels shape {level_values.shape}")
        if expected is not None:
            sizes.append(int(expected))
        if not sizes:
            return 1
        if any(size != sizes[0] for size in sizes[1:]):
            raise ValueError(f"batch row dimensions disagree: {sizes}")
        return sizes[0]

    @staticmethod
    def _telemetry_vector(telemetry, key, nrow):
        if key not in telemetry or telemetry[key] is None:
            raise ValueError(f"missing telemetry key {key!r}")
        value = np.asarray(telemetry[key], dtype=float)
        if value.ndim == 0:
            return np.full(nrow, float(value), dtype=float)
        if value.ndim != 1 or value.size != nrow:
            raise ValueError(
                f"telemetry {key!r} must be scalar or length {nrow}; "
                f"got shape {value.shape}"
            )
        return value

    @classmethod
    def _telemetry_rows(cls, telemetry, nrow):
        required = (
            "THERM_FPGA", "SPE_ADC0_T", "SPE_ADC1_T", "SPE_1VAD8_V",
            "VMON_1V2D", "SPE_1VAD8_C",
        )
        return {
            key: cls._telemetry_vector(telemetry, key, nrow) for key in required
        }

    @classmethod
    def _feature_matrix(cls, channel, telemetry, nrow):
        adc_key = "SPE_ADC0_T" if int(channel) <= 1 else "SPE_ADC1_T"
        therm = cls._telemetry_vector(telemetry, "THERM_FPGA", nrow)
        adc = cls._telemetry_vector(telemetry, adc_key, nrow)
        v8 = cls._telemetry_vector(telemetry, "SPE_1VAD8_V", nrow)
        v12 = cls._telemetry_vector(telemetry, "VMON_1V2D", nrow)
        i8 = cls._telemetry_vector(telemetry, "SPE_1VAD8_C", nrow)
        features = np.empty((nrow, 9), dtype=float)
        features[:, 0] = 1.0
        features[:, 1] = therm
        features[:, 2] = adc
        features[:, 3] = v8
        features[:, 4] = v12
        features[:, 5] = i8
        features[:, 6] = therm * therm
        features[:, 7] = adc * adc
        features[:, 8] = therm * adc
        return features

    @classmethod
    def _features(cls, channel, telemetry):
        return cls._feature_matrix(channel, telemetry, 1)[0]

    def _predict_pc(self, key, pc, features, models):
        pc_i = {"PC1": 0, "PC2": 1}[pc]
        family_i = MODEL_FAMILIES.index(models[pc_i])
        level_i = GAIN_LEVELS.index(key[0])
        channel = int(key[1])
        coefficients = self._coefficients[
            family_i, level_i, channel, pc_i, :
        ]
        # The explicit contraction remains vectorized but evaluates each row
        # identically regardless of batch/chunk shape (BLAS matvec kernels can
        # otherwise differ by a few ulps as group sizes change).
        return np.einsum(
            "...i,i->...", np.asarray(features, dtype=float), coefficients,
            optimize=False,
        )
