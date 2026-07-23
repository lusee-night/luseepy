"""Full-Stokes four-port covariance simulators."""

import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import astropy.units as u
from astropy.time import Time
import fitsio
import jax
import jax.numpy as jnp
import numpy as np

from .Covariance import (
    assemble_open_covariance,
    blackbody_normalization,
    load_covariance,
    pack_covariance,
)
from .frequencies import FrequencyMap
from .GainModel import V2_PER_HZ
from .InstrumentResponse import InstrumentResponse
from .LabeledArray import FRAME_TOPO, LabeledArray


def _as_time_array(times):
    if isinstance(times, Time):
        result = times
    else:
        result = Time(times)
    if result.isscalar:
        result = Time([result])
    if len(result) == 0:
        raise ValueError("times must contain at least one timestamp.")
    return result


def _frame_name(frame):
    value = str(frame).lower()
    aliases = {
        "mcmf": "mcmf",
        "galactic": "galactic",
        "equatorial": "fk5",
        "icrs": "fk5",
        "topo": "topo",
        "instrument-topocentric": "topo",
    }
    if value not in aliases:
        raise ValueError(f"Unsupported sky frame {frame!r}.")
    return aliases[value]


def _interpolate_frequency_axis(array, frequency_map, axis):
    moved = jnp.moveaxis(jnp.asarray(array), axis, 0)
    interpolated = frequency_map.from_unique(moved)
    return jnp.moveaxis(interpolated, 0, axis)


def _gridded_frequency_axis(value):
    if hasattr(value, "get_alm_at_freq") or hasattr(
        value, "polarized_alm_at_freq"
    ):
        return None
    frequencies = getattr(value, "freq", None)
    if frequencies is None:
        frequencies = getattr(value, "freqs", None)
    if frequencies is None:
        return None
    frequencies = np.asarray(frequencies, dtype=np.float64).reshape(-1)
    if frequencies.size == 0:
        return None
    return frequencies


def default_target_frequencies(beam, sky_model, receiver):
    """Return response channels inside every gridded input interval."""
    target = np.asarray(beam.freq, dtype=np.float64).reshape(-1)
    gridded_inputs = (
        ("sky", _gridded_frequency_axis(sky_model)),
        ("receiver", getattr(receiver, "freq", None)),
    )
    removed = {}
    for name, source_frequency in gridded_inputs:
        if source_frequency is None:
            continue
        source_frequency = np.asarray(
            source_frequency, dtype=np.float64
        ).reshape(-1)
        keep = (
            (target >= source_frequency[0])
            & (target <= source_frequency[-1])
        )
        if np.any(~keep):
            removed[name] = target[~keep].copy()
        target = target[keep]
    if target.size == 0:
        raise ValueError(
            "Response, sky, and receiver have no common frequency channels."
        )
    return target, removed


def _package_version(distribution):
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "unknown"


def _i_only_sky_alms(sky_model, target_freqs, lmax):
    import s2fft

    if hasattr(sky_model, "get_alm_at_freq"):
        packed = jnp.asarray(sky_model.get_alm_at_freq(target_freqs))
    elif getattr(sky_model, "freq", None) is None:
        packed = jnp.asarray(
            sky_model.get_alm(
                np.zeros(len(target_freqs), dtype=np.int32),
                freq=target_freqs,
            )
        )
    else:
        frequency_map = FrequencyMap.build(target_freqs, sky_model.freq)
        native = jnp.asarray(
            sky_model.get_alm(frequency_map.source_indices)
        )
        packed = frequency_map.from_unique(native)
    L = lmax + 1
    scalar = jax.vmap(
        lambda values: s2fft.sampling.reindex.flm_hp_to_2d_fast(values, L)
    )(packed)
    zeros = jnp.zeros_like(scalar)
    return jnp.stack((scalar, zeros, zeros, zeros), axis=1)


def prepare_polarized_sky_alms(sky_model, target_freqs, lmax):
    """Prepare target-aligned Croissant harmonic-dual sky coefficients."""
    import croissant as cro

    if hasattr(sky_model, "polarized_alm_at_freq"):
        result = jnp.asarray(
            sky_model.polarized_alm_at_freq(target_freqs, lmax=lmax)
        )
    elif hasattr(cro, "PolarizedSky") and isinstance(
        sky_model, cro.PolarizedSky
    ):
        source_freqs = np.asarray(sky_model.freqs, dtype=np.float64)
        frequency_map = FrequencyMap.build(target_freqs, source_freqs)
        source_indices = frequency_map.source_indices
        selected = cro.PolarizedSky(
            sky_model.data[source_indices],
            source_freqs[source_indices],
            sampling=sky_model.sampling,
            coord=sky_model.coord,
            convention=sky_model.convention,
            stokes=sky_model.stokes,
            units=sky_model.units,
            frame=sky_model.frame,
            tangent_basis=sky_model.tangent_basis,
            niter=sky_model._niter,
        )
        if lmax > selected.lmax:
            raise ValueError(
                f"Requested lmax={lmax} exceeds sky limit {selected.lmax}."
            )
        native = selected.compute_alm(lmax=lmax)
        result = frequency_map.from_unique(native)
    else:
        result = _i_only_sky_alms(sky_model, target_freqs, lmax)
    expected = (len(target_freqs), 4, lmax + 1, 2 * lmax + 1)
    if result.shape != expected:
        raise ValueError(
            f"Polarized sky alms must have shape {expected}; got "
            f"{result.shape}."
        )
    return result


def _rotate_sky_to_topo(sky_alms, sky_frame, obs, time, lmax):
    import croissant as cro
    from lunarsky import LunarTopo

    source = _frame_name(sky_frame)
    if source == "topo":
        return sky_alms
    topo = LunarTopo(obstime=time, location=obs.loc)
    rotation, dl_array = cro.rotations.generate_euler_dl(
        lmax,
        source,
        topo,
    )
    return cro.rotations.rotate_alm(
        sky_alms,
        rotation,
        dl_array=dl_array,
    )


class FullStokesSimulatorBase:
    """Shared response preparation and covariance postprocessing."""

    engine = "base"

    def __init__(
        self,
        obs,
        beam,
        sky_model,
        receiver,
        T_moon=250.0,
        products="all",
        freq=None,
        lmax=128,
    ):
        if not isinstance(beam, InstrumentResponse):
            raise TypeError("beam must be one four-port InstrumentResponse.")
        self.obs = obs
        self.beam = beam
        self.sky_model = sky_model
        self.receiver = receiver
        self.T_moon = T_moon
        self.products = products
        if freq is None:
            target_freq, removed = default_target_frequencies(
                beam, sky_model, receiver
            )
        else:
            target_freq = freq
            removed = {}
        self.freq = np.asarray(target_freq, dtype=np.float64).reshape(-1)
        self.default_frequency_removals = removed
        if self.freq.size == 0:
            raise ValueError("Target frequency array must be nonempty.")
        self.Nfreq = self.freq.size
        self.lmax = int(lmax)
        self.result = None
        self.result_times = None
        self.result_units = V2_PER_HZ
        self.result_frame = FRAME_TOPO
        self.product_labels = None
        self.covariance = None
        self.ZA_target = None
        self.ZL_target = None
        self.M_target = None
        self.Rsky_target = None
        self.Rmoon_target = None
        self.blackbody_normalization = None
        self.result_beam = None
        self.result_receiver = None
        self.result_sky = None

    @property
    def result_labeled(self):
        if self.result is None:
            return None
        return LabeledArray(
            self.result,
            units=self.result_units,
            frame=self.result_frame,
        )

    def prepare_pair_alms(self, beam):
        pair_alms, frequency_map = beam.pair_stokes_alms(
            self.lmax,
            self.freq,
        )
        ZA = frequency_map.from_native(beam.ZA)
        monopole = pair_alms[:, :, 0, 0, self.lmax]
        pair_rsky = 0.25 * monopole * jnp.sqrt(4 * jnp.pi)
        from .InstrumentResponse import assemble_pair_matrix

        Rsky = assemble_pair_matrix(jnp.swapaxes(pair_rsky, 0, 1))
        dissipative = 0.5 * (
            ZA + jnp.swapaxes(ZA.conjugate(), -1, -2)
        )
        Rmoon = dissipative - Rsky
        return pair_alms, ZA, Rsky, Rmoon

    def _convolve(self, pair_alms, sky_alms, sky_frame, times):
        raise NotImplementedError

    def simulate(
        self,
        times=None,
        *,
        sky=None,
        beam=None,
        receiver=None,
    ):
        """Simulate exact supplied timestamps and return a bare array."""
        times = _as_time_array(self.obs.times if times is None else times)
        effective_sky = self.sky_model if sky is None else sky
        effective_beam = self.beam if beam is None else beam
        effective_receiver = self.receiver if receiver is None else receiver
        if not isinstance(effective_beam, InstrumentResponse):
            raise TypeError("beam override must be an InstrumentResponse.")

        pair_alms, ZA, Rsky, Rmoon = self.prepare_pair_alms(effective_beam)
        sky_alms = prepare_polarized_sky_alms(
            effective_sky,
            self.freq,
            self.lmax,
        )
        sky_frame = getattr(
            effective_sky,
            "frame",
            getattr(effective_sky, "coord", None),
        )
        pair_integrals = self._convolve(
            pair_alms,
            sky_alms,
            sky_frame,
            times,
        )
        open_covariance = assemble_open_covariance(
            pair_integrals,
            Rmoon,
            self.T_moon,
        )
        ZL = effective_receiver.Z(self.freq)
        covariance, M = load_covariance(open_covariance, ZA, ZL)
        packed, labels = pack_covariance(covariance, self.products)

        self.result = packed
        self.result_times = times.copy()
        self.product_labels = labels
        self.covariance = covariance
        self.ZA_target = ZA
        self.ZL_target = ZL
        self.M_target = M
        self.Rsky_target = Rsky
        self.Rmoon_target = Rmoon
        self.blackbody_normalization = blackbody_normalization(ZA, M)
        self.result_beam = effective_beam
        self.result_receiver = effective_receiver
        self.result_sky = effective_sky
        return self.result

    def temperature_equivalent(self):
        """Return the covariance divided by blackbody normalization."""
        if self.covariance is None:
            raise RuntimeError("simulate() must be called first.")
        equivalent = self.covariance / self.blackbody_normalization[None]
        packed, _ = pack_covariance(equivalent, self.product_labels)
        return LabeledArray(packed, units="K", frame=FRAME_TOPO)

    def write_fits(self, filename):
        """Persist target-aligned covariance, matrices, units, and exact times."""
        if self.result is None or self.result_times is None:
            raise RuntimeError("simulate() must be called before write_fits().")
        filename = str(Path(filename))
        response = self.result_beam
        receiver = self.result_receiver
        sky = self.result_sky
        header = {
            "VERSION": 3,
            "ENGINE": self.engine,
            "LUNAR_DAY": str(self.obs.time_range),
            "LUN_LAT_DEG": float(self.obs.lun_lat_deg),
            "LUN_LONG_DEG": float(self.obs.lun_long_deg),
            "LUN_HEIGHT_M": float(self.obs.lun_height_m),
            "DELTAT_SEC": float(self.obs.deltaT_sec),
            "RESPONSE": str(response.filename or "in-memory"),
            "RESPHASH": response.content_hash,
            "RESPVAL": bool(response.validated),
            "FREQINT": "linear-native-alm",
            "TIMESYS": self.result_times.scale.upper(),
            "TIMEUNIT": "d",
            "CLOCKSRC": "astropy/lunarsky",
            "SCALEASM": False,
            "RECMODEL": type(receiver).__name__,
            "RECCHANS": ",".join(
                getattr(receiver, "channel_map", ())
            ),
            "RECSRC": str(getattr(receiver, "source", None) or "analytic"),
            "SKYMODEL": type(sky).__name__,
            "SKYFRAME": str(
                getattr(
                    sky,
                    "frame",
                    getattr(sky, "coord", "unknown"),
                )
            ),
            "SKYSRC": str(
                getattr(
                    sky,
                    "filename",
                    getattr(sky, "source", "in-memory"),
                )
            ),
            "LUSEEVER": _package_version("lusee"),
            "CROVER": _package_version("croissant-sim"),
            "S2FFTVER": _package_version("s2fft"),
        }
        fits = fitsio.FITS(filename, "rw", clobber=True)
        fits.write(
            np.asarray(self.result),
            extname="data",
            header={**header, "BUNIT": V2_PER_HZ},
        )
        fits.write(
            np.asarray(self.freq, dtype=np.float64),
            extname="freq",
            header={"BUNIT": "MHz"},
        )
        fits.write(
            np.asarray(self.result_times.mjd, dtype=np.float64),
            extname="time",
            header={
                "BUNIT": "d",
                "TIMESYS": self.result_times.scale.upper(),
                "TIMEUNIT": "d",
                "CLOCKSRC": "astropy/lunarsky",
                "SCALEASM": False,
            },
        )
        product_table = np.zeros(
            len(self.product_labels),
            dtype=[("label", "S8")],
        )
        product_table["label"] = np.asarray(self.product_labels, dtype="S8")
        fits.write_table(
            product_table,
            extname="products",
            header={"BUNIT": "1"},
        )

        def write_complex(name, values, units):
            values = np.asarray(values)
            fits.write(
                values.real,
                extname=f"{name}_real",
                header={"BUNIT": units},
            )
            fits.write(
                values.imag,
                extname=f"{name}_imag",
                header={"BUNIT": units},
            )

        write_complex("ZA", self.ZA_target, "Ohm")
        write_complex("ZL", self.ZL_target, "Ohm")
        write_complex("M", self.M_target, "1")
        write_complex("Rsky", self.Rsky_target, "Ohm")
        write_complex("Rmoon", self.Rmoon_target, "Ohm")
        write_complex(
            "blackbody_normalization",
            self.blackbody_normalization,
            "V^2/(Hz K)",
        )
        params = getattr(receiver, "params", {})
        payload = json.dumps(
            {
                key: np.asarray(value).tolist()
                for key, value in params.items()
            },
            sort_keys=True,
        ).encode("utf-8")
        fits.write(
            np.frombuffer(payload, dtype=np.uint8),
            extname="receiver_params",
            header={"BUNIT": "1"},
        )
        fits.close()
        return filename


class FullStokesCroSimulator(FullStokesSimulatorBase):
    """Diagonal-in-m Croissant time kernel for four-port covariance."""

    engine = "croissant"

    def _convolve(self, pair_alms, sky_alms, sky_frame, times):
        import croissant as cro

        source = _frame_name(sky_frame)
        elapsed = np.asarray(
            (times.tdb - times[0].tdb).to_value(u.s),
            dtype=np.float64,
        )
        self.result_epoch_tdb_jd = float(times[0].tdb.jd)
        self.elapsed_tdb_seconds = jnp.asarray(elapsed)
        if source == "topo":
            beam_work = pair_alms
            sky_work = sky_alms
            phases = jnp.ones(
                (len(times), 2 * self.lmax + 1),
                dtype=pair_alms.dtype,
            )
        else:
            from lunarsky import LunarTopo

            from .spice_utils import ensure_lunarsky_moon_frame

            ensure_lunarsky_moon_frame()
            et = cro.rotations.jd_to_et(times[0].tdb.jd)
            topo = LunarTopo(obstime=times[0], location=self.obs.loc)
            beam_rotation, beam_dl = cro.rotations.generate_euler_dl(
                self.lmax,
                topo,
                "mepa",
                et=et,
            )
            beam_work = cro.rotations.rotate_alm(
                pair_alms,
                beam_rotation,
                dl_array=beam_dl,
            )
            if source == "galactic":
                sky_work = cro.rotations.gal2mepa(sky_alms, et=et)
            else:
                sky_rotation, sky_dl = cro.rotations.generate_euler_dl(
                    self.lmax,
                    source,
                    "mepa",
                    et=et,
                )
                sky_work = cro.rotations.rotate_alm(
                    sky_alms,
                    sky_rotation,
                    dl_array=sky_dl,
                )
            phases = cro.simulator.rot_alm_z(
                self.lmax,
                times=elapsed,
            )
        return jnp.swapaxes(
            cro.polarized_convolve(
                beam_work,
                sky_work,
                phases,
            ),
            1,
            2,
        )


class FullStokesTopoJaxSimulator(FullStokesSimulatorBase):
    """Independent per-time Wigner rotation and contraction."""

    engine = "topo"

    def _convolve(self, pair_alms, sky_alms, sky_frame, times):
        values = []
        for time in times:
            sky_topo = _rotate_sky_to_topo(
                sky_alms,
                sky_frame,
                self.obs,
                time,
                self.lmax,
            )
            pair_value = jnp.einsum(
                "fclm,pfclm->pf",
                sky_topo.conjugate(),
                pair_alms,
            )
            values.append(jnp.swapaxes(pair_value, 0, 1))
        self.result_epoch_tdb_jd = float(times[0].tdb.jd)
        self.elapsed_tdb_seconds = jnp.asarray(
            (times.tdb - times[0].tdb).to_value(u.s)
        )
        return jnp.stack(values, axis=0)


CovarianceCroSimulator = FullStokesCroSimulator
CovarianceTopoJaxSimulator = FullStokesTopoJaxSimulator
