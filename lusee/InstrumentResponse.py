"""Four-port open-circuit instrument response and frequency-aware transforms."""

import copy
import os
from pathlib import Path

import fitsio
import jax
import jax.numpy as jnp
import numpy as np
from scipy.constants import c, physical_constants

from .frequencies import FrequencyMap
from .ReceiverImpedance import loading_matrix
from .ResponsePhysics import (
    compute_sky_moon_resistance,
    response_payload_hash,
    validate_response_matrices,
)


VACUUM_IMPEDANCE_OHM = physical_constants[
    "characteristic impedance of vacuum"
][0]
PORT_PAIRS = tuple((a, b) for a in range(4) for b in range(a, 4))
DEFAULT_TRANSFORM_MEMORY_BUDGET_BYTES = 512 * 2**20
RESPONSE_UNITS = {
    "freq": "MHz",
    "theta": "deg",
    "phi": "deg",
    "H_theta_real": "m",
    "H_theta_imag": "m",
    "H_phi_real": "m",
    "H_phi_imag": "m",
    "ZA_real": "Ohm",
    "ZA_imag": "Ohm",
    "Rsky_real": "Ohm",
    "Rsky_imag": "Ohm",
    "Rmoon_real": "Ohm",
    "Rmoon_imag": "Ohm",
}
REQUIRED_PROVENANCE = (
    "SOURCE",
    "SOURCE_ROOT",
    "INPUT_KIND",
    "FIELD_KIND",
    "AMP_CONV",
    "TIMECONV",
    "ZA_SOURCE",
    "GIT_SHA",
    "COORDSYS",
    "THETADEF",
    "PHIDEF",
    "OMEGADEF",
    "POLBASIS",
    "PHASEREF",
    "PORTS",
)
CANONICAL_CONVENTIONS = {
    "TIMECONV": {"e+jwt"},
    "COORDSYS": {"instrument-topocentric"},
    "THETADEF": {"colatitude-from-+z"},
    "PHIDEF": {"right-handed-about-+z"},
    "OMEGADEF": {"source-arrival-direction"},
    "POLBASIS": {"e_theta,e_phi"},
    "PORTS": {"0123"},
    "INPUT_KIND": {"bare", "embedded"},
    "FIELD_KIND": {"re", "r_e", "effective-length", "effective_length"},
    "AMP_CONV": {"rms", "peak"},
}


def _transform_memory_budget():
    raw = os.environ.get(
        "LUSEE_RESPONSE_TRANSFORM_MAX_BYTES",
        str(DEFAULT_TRANSFORM_MEMORY_BUDGET_BYTES),
    )
    try:
        result = int(raw)
    except ValueError as error:
        raise ValueError(
            "LUSEE_RESPONSE_TRANSFORM_MAX_BYTES must be an integer."
        ) from error
    if result <= 0:
        raise ValueError(
            "LUSEE_RESPONSE_TRANSFORM_MAX_BYTES must be positive."
        )
    return result


def _hdu_names(fits):
    return {hdu.get_extname().lower() for hdu in fits}


def _read_unit_checked(fits, name, expected_unit):
    names = _hdu_names(fits)
    if name.lower() not in names:
        raise ValueError(f"Response FITS is missing HDU {name!r}.")
    hdu = fits[name]
    header = dict(hdu.read_header())
    actual = str(header.get("BUNIT", "")).strip()
    if actual != expected_unit:
        raise ValueError(
            f"HDU {name!r} has BUNIT={actual!r}; expected {expected_unit!r}."
        )
    return hdu.read()


def _read_complex(fits, name, units):
    real = _read_unit_checked(fits, f"{name}_real", units)
    imag = _read_unit_checked(fits, f"{name}_imag", units)
    if real.shape != imag.shape:
        raise ValueError(f"Real/imaginary {name} HDUs have different shapes.")
    return real + 1j * imag


def _validate_provenance(header):
    unknown_values = {"", "unknown", "unspecified", "none"}
    missing = [
        key
        for key in REQUIRED_PROVENANCE
        if key not in header
        or str(header[key]).strip().lower() in unknown_values
    ]
    if missing:
        raise ValueError(
            "Validated response is missing explicit provenance for: "
            + ", ".join(missing)
        )
    for key, allowed in CANONICAL_CONVENTIONS.items():
        value = str(header[key]).strip().lower()
        if value not in allowed:
            raise ValueError(
                f"Validated response has unsupported {key}={header[key]!r}; "
                f"expected one of {sorted(allowed)}."
            )


@jax.tree_util.register_pytree_node_class
class InstrumentResponse:
    """One coupled four-port response loaded from instrument FITS v3."""

    is_four_port_response = True

    def __init__(self, filename, *, require_validated=True):
        self.filename = str(Path(filename))
        header = dict(fitsio.read_header(self.filename))
        version = float(header.get("VERSION", -1))
        if version != 3.0:
            raise ValueError(
                f"InstrumentResponse requires FITS VERSION=3; got {version}."
            )
        validated = bool(header.get("VALIDATED", False))
        if require_validated and not validated:
            raise ValueError(
                "Response is marked VALIDATED=False; pass "
                "require_validated=False only for development data."
            )
        if validated:
            _validate_provenance(header)
        fits = fitsio.FITS(self.filename, "r")
        self.freq = np.asarray(
            _read_unit_checked(fits, "freq", "MHz"), dtype=np.float64
        )
        self.theta_deg = np.asarray(
            _read_unit_checked(fits, "theta", "deg"), dtype=np.float64
        )
        self.phi_deg = np.asarray(
            _read_unit_checked(fits, "phi", "deg"), dtype=np.float64
        )
        H_theta = _read_complex(fits, "H_theta", "m")
        H_phi = _read_complex(fits, "H_phi", "m")
        ZA = _read_complex(fits, "ZA", "Ohm")
        Rsky = _read_complex(fits, "Rsky", "Ohm")
        Rmoon = _read_complex(fits, "Rmoon", "Ohm")
        names = _hdu_names(fits)
        Vsource = (
            _read_complex(fits, "Vsource", "V")
            if "vsource_real" in names
            else None
        )
        Zref = (
            _read_unit_checked(fits, "Zref", "Ohm")
            if "zref" in names
            else None
        )
        fits.close()
        if not np.all(np.isfinite(H_theta)):
            raise ValueError("H_theta contains non-finite values.")
        if not np.all(np.isfinite(H_phi)):
            raise ValueError("H_phi contains non-finite values.")
        self.H_theta = jnp.asarray(H_theta)
        self.H_phi = jnp.asarray(H_phi)
        self.ZA = jnp.asarray(ZA)
        self.Rsky_native = jnp.asarray(Rsky)
        self.Rmoon_native = jnp.asarray(Rmoon)
        self.header = header
        self.validated = validated
        self.id = header.get("PORTS", "0123")
        self.frame = str(header.get("COORDSYS", "instrument-topocentric"))
        self.tangent_basis = str(header.get("POLBASIS", "e_theta,e_phi"))
        self.nports = 4
        self.pairs = PORT_PAIRS
        self.theta = jnp.asarray(np.radians(self.theta_deg))
        self.phi = jnp.asarray(np.radians(self.phi_deg))
        self.Nfreq = self.freq.size
        self.Ntheta = self.theta_deg.size
        self.Nphi = self.phi_deg.size
        self.freq_min = float(self.freq[0])
        self.freq_max = float(self.freq[-1])
        self._validate()
        if validated:
            field_rsky, _ = compute_sky_moon_resistance(
                self.freq,
                self.theta_deg,
                self.phi_deg,
                H_theta,
                H_phi,
                ZA,
            )
            validate_response_matrices(
                ZA,
                Rsky,
                Rmoon,
                field_rsky=field_rsky,
            )
        computed_hash = response_payload_hash(
            freq=self.freq,
            theta_deg=self.theta_deg,
            phi_deg=self.phi_deg,
            H_theta=H_theta,
            H_phi=H_phi,
            ZA=ZA,
            Rsky=Rsky,
            Rmoon=Rmoon,
            Vsource=Vsource,
            Zref=Zref,
            metadata=header,
        )
        if "CONTENT" in header and str(header["CONTENT"]) != computed_hash:
            raise ValueError(
                "Response CONTENT hash does not match the persisted payload."
            )
        self.content_hash = computed_hash
        self._alm_cache = {}
        self._transform_count = 0
        self.transform_memory_budget_bytes = _transform_memory_budget()

    @classmethod
    def from_arrays(
        cls,
        freq_mhz,
        theta_deg,
        phi_deg,
        H_theta,
        H_phi,
        ZA,
        Rsky,
        Rmoon,
        *,
        validated=None,
        metadata=None,
    ):
        """Construct a response in memory for analytic fixtures and tests."""
        obj = cls.__new__(cls)
        obj.filename = None
        obj.freq = np.asarray(freq_mhz, dtype=np.float64)
        obj.theta_deg = np.asarray(theta_deg, dtype=np.float64)
        obj.phi_deg = np.asarray(phi_deg, dtype=np.float64)
        obj.H_theta = jnp.asarray(H_theta)
        obj.H_phi = jnp.asarray(H_phi)
        obj.ZA = jnp.asarray(ZA)
        obj.Rsky_native = jnp.asarray(Rsky)
        obj.Rmoon_native = jnp.asarray(Rmoon)
        obj.header = {
            str(key).upper(): value
            for key, value in dict(metadata or {}).items()
        }
        if validated is None:
            validated = bool(obj.header.get("VALIDATED", False))
        obj.header.setdefault("VERSION", 3)
        obj.header["VALIDATED"] = bool(validated)
        obj.validated = bool(validated)
        if obj.validated:
            _validate_provenance(obj.header)
        obj.id = str(obj.header.get("PORTS", "0123"))
        obj.frame = str(
            obj.header.get("COORDSYS", "instrument-topocentric")
        )
        obj.tangent_basis = str(
            obj.header.get("POLBASIS", "e_theta,e_phi")
        )
        obj.nports = 4
        obj.pairs = PORT_PAIRS
        obj.theta = jnp.asarray(np.radians(obj.theta_deg))
        obj.phi = jnp.asarray(np.radians(obj.phi_deg))
        obj.Nfreq = obj.freq.size
        obj.Ntheta = obj.theta_deg.size
        obj.Nphi = obj.phi_deg.size
        obj.freq_min = float(obj.freq[0])
        obj.freq_max = float(obj.freq[-1])
        obj._validate()
        if obj.validated:
            try:
                field_rsky, _ = compute_sky_moon_resistance(
                    obj.freq,
                    obj.theta_deg,
                    obj.phi_deg,
                    obj.H_theta,
                    obj.H_phi,
                    obj.ZA,
                )
            except jax.errors.TracerArrayConversionError as error:
                raise ValueError(
                    "Traced in-memory response overrides must pass "
                    "validated=False; VALIDATED=True requires host-side "
                    "field/matrix validation."
                ) from error
            validate_response_matrices(
                obj.ZA,
                obj.Rsky_native,
                obj.Rmoon_native,
                field_rsky=field_rsky,
            )
        try:
            obj.content_hash = response_payload_hash(
                freq=obj.freq,
                theta_deg=obj.theta_deg,
                phi_deg=obj.phi_deg,
                H_theta=obj.H_theta,
                H_phi=obj.H_phi,
                ZA=obj.ZA,
                Rsky=obj.Rsky_native,
                Rmoon=obj.Rmoon_native,
                metadata=obj.header,
            )
        except jax.errors.TracerArrayConversionError:
            # Traced overrides have no stable byte representation. Their
            # per-instance harmonic cache remains isolated.
            obj.content_hash = "differentiable-in-memory"
        obj._alm_cache = {}
        obj._transform_count = 0
        obj.transform_memory_budget_bytes = _transform_memory_budget()
        return obj

    @property
    def Rsky(self):
        return self.Rsky_native

    @property
    def Rmoon(self):
        return self.Rmoon_native

    def _validate(self):
        if self.frame.strip().lower() not in {
            "instrument-topocentric",
            "topo",
        }:
            raise ValueError(
                "Instrument response frame must be "
                "'instrument-topocentric' (alias 'topo')."
            )
        if self.tangent_basis.strip().lower() not in {
            "e_theta,e_phi",
            "theta-phi",
        }:
            raise ValueError(
                "Instrument response tangent basis must be "
                "'e_theta,e_phi' (alias 'theta-phi')."
            )
        if self.freq.ndim != 1 or self.freq.size == 0:
            raise ValueError("Response frequency grid must be one-dimensional.")
        if not np.all(np.isfinite(self.freq)):
            raise ValueError("Response frequency grid contains non-finite values.")
        if self.freq.size > 1 and not np.all(np.diff(self.freq) > 0):
            raise ValueError("Response frequency grid must be strictly increasing.")
        expected = (4, self.Nfreq, self.Ntheta, self.Nphi)
        if self.H_theta.shape != expected or self.H_phi.shape != expected:
            raise ValueError(f"Response fields must have shape {expected}.")
        matrix_shape = (self.Nfreq, 4, 4)
        for name in ("ZA", "Rsky_native", "Rmoon_native"):
            if getattr(self, name).shape != matrix_shape:
                raise ValueError(f"{name} must have shape {matrix_shape}.")
        if not np.isclose(self.phi_deg[0], 0.0):
            raise ValueError("Response phi grid must start at zero degrees.")
        if not np.isclose(self.theta_deg[0], 0.0):
            raise ValueError("Response theta grid must start at zero degrees.")
        if self.validated:
            if self.theta_deg.size < 2 or self.phi_deg.size < 3:
                raise ValueError(
                    "Validated response grids need at least two theta samples "
                    "and three stored phi samples including the wrap."
                )
            if not np.isclose(self.theta_deg[-1], 90.0):
                raise ValueError(
                    "Validated response theta grid must end at 90 degrees."
                )
            if not np.allclose(
                np.diff(self.theta_deg),
                np.diff(self.theta_deg)[0],
            ):
                raise ValueError(
                    "Validated response theta grid must be uniform."
                )
            if not np.isclose(self.phi_deg[-1], 360.0):
                raise ValueError(
                    "Validated response phi grid must retain the 0/360 wrap."
                )
            if not np.allclose(
                np.diff(self.phi_deg),
                np.diff(self.phi_deg)[0],
            ):
                raise ValueError(
                    "Validated response phi grid must be uniform."
                )
            expected_phi_count = 4 * (self.theta_deg.size - 1) + 1
            if self.phi_deg.size != expected_phi_count:
                raise ValueError(
                    "Validated response angular grid must satisfy "
                    "Nphi-1 == 4*(Ntheta-1) for full-sphere MWSS padding."
                )

    def tree_flatten(self):
        children = (
            self.H_theta,
            self.H_phi,
            self.ZA,
            self.Rsky_native,
            self.Rmoon_native,
        )
        aux = (
            tuple(self.freq.tolist()),
            tuple(self.theta_deg.tolist()),
            tuple(self.phi_deg.tolist()),
            self.filename,
            self.validated,
            self.id,
            self.frame,
            self.tangent_basis,
            tuple(
                sorted(
                    (str(key), str(value))
                    for key, value in self.header.items()
                )
            ),
            self.content_hash,
            self.transform_memory_budget_bytes,
        )
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        (
            freq,
            theta_deg,
            phi_deg,
            filename,
            validated,
            id_value,
            frame,
            tangent_basis,
            header_items,
            content_hash,
            transform_memory_budget_bytes,
        ) = aux
        H_theta, H_phi, ZA, Rsky, Rmoon = children
        obj = cls.__new__(cls)
        obj.filename = filename
        obj.freq = np.asarray(freq, dtype=np.float64)
        obj.theta_deg = np.asarray(theta_deg, dtype=np.float64)
        obj.phi_deg = np.asarray(phi_deg, dtype=np.float64)
        obj.H_theta = H_theta
        obj.H_phi = H_phi
        obj.ZA = ZA
        obj.Rsky_native = Rsky
        obj.Rmoon_native = Rmoon
        obj.validated = validated
        obj.id = id_value
        obj.frame = frame
        obj.tangent_basis = tangent_basis
        obj.header = dict(header_items)
        obj.content_hash = content_hash
        obj.transform_memory_budget_bytes = transform_memory_budget_bytes
        obj.nports = 4
        obj.pairs = PORT_PAIRS
        obj.theta = jnp.asarray(np.radians(obj.theta_deg))
        obj.phi = jnp.asarray(np.radians(obj.phi_deg))
        obj.Nfreq = obj.freq.size
        obj.Ntheta = obj.theta_deg.size
        obj.Nphi = obj.phi_deg.size
        obj.freq_min = float(obj.freq[0])
        obj.freq_max = float(obj.freq[-1])
        obj._alm_cache = {}
        obj._transform_count = 0
        return obj

    def pair_stokes_maps(self, a, b, freq_ndx=None):
        """Return complex bare I, Q, U, V response maps for one port pair."""
        if not (0 <= a < 4 and 0 <= b < 4):
            raise ValueError("Port indices must lie in [0, 3].")
        indices = (
            jnp.arange(self.Nfreq)
            if freq_ndx is None
            else jnp.atleast_1d(jnp.asarray(freq_ndx, dtype=jnp.int32))
        )
        at = self.H_theta[a, indices]
        ap = self.H_phi[a, indices]
        bt = self.H_theta[b, indices]
        bp = self.H_phi[b, indices]
        response_i = at * bt.conjugate() + ap * bp.conjugate()
        response_q = at * bt.conjugate() - ap * bp.conjugate()
        response_u = at * bp.conjugate() + ap * bt.conjugate()
        response_v = 1j * (
            ap * bt.conjugate() - at * bp.conjugate()
        )
        return jnp.stack(
            (response_i, response_q, response_u, response_v),
            axis=1,
        )

    def all_pair_stokes_maps(self, freq_ndx=None):
        """Return maps with layout pair, frequency, IQUV, theta, phi."""
        return self.pair_stokes_maps_for_pairs(
            self.pairs,
            freq_ndx=freq_ndx,
        )

    def pair_stokes_maps_for_pairs(self, pairs, freq_ndx=None):
        """Return maps for an ordered subset of unique port pairs."""
        pairs = tuple(tuple(pair) for pair in pairs)
        return jnp.stack(
            [
                self.pair_stokes_maps(a, b, freq_ndx=freq_ndx)
                for a, b in pairs
            ],
            axis=0,
        )

    def _transform_chunk_shape(self, lmax):
        """Choose pair/frequency chunks under the transform memory budget."""
        full_theta = 2 * (self.Ntheta - 1) + 1
        unique_phi = self.Nphi - 1
        map_elements = 4 * full_theta * unique_phi
        alm_elements = 4 * (int(lmax) + 1) * (2 * int(lmax) + 1)
        itemsize = np.dtype(self.H_theta.dtype).itemsize
        # Account for the input, output, spin workspaces, and compiler
        # temporaries. This is intentionally conservative.
        estimated_per_pair_frequency = (
            map_elements + alm_elements
        ) * itemsize * 6
        pair_chunk = min(
            len(self.pairs),
            max(
                1,
                self.transform_memory_budget_bytes
                // max(estimated_per_pair_frequency, 1),
            ),
        )
        frequency_chunk = min(
            self.Nfreq,
            max(
                1,
                self.transform_memory_budget_bytes
                // max(estimated_per_pair_frequency * pair_chunk, 1),
            ),
        )
        return int(pair_chunk), int(frequency_chunk)

    def _full_sphere_maps(self, maps):
        phi_has_wrap = np.isclose(
            self.phi_deg[-1] - self.phi_deg[0], 360.0
        )
        if not phi_has_wrap:
            raise ValueError(
                "Instrument response requires a periodic phi wraparound bin."
            )
        maps = maps[..., :-1]
        if np.isclose(self.theta_deg[-1], 180.0):
            return maps
        if self.theta_deg.size < 2:
            raise ValueError("Response theta grid is too short.")
        step = float(self.theta_deg[1] - self.theta_deg[0])
        if not np.allclose(np.diff(self.theta_deg), step):
            raise ValueError("Response theta grid must be uniform.")
        full_count = int(round(180.0 / step)) + 1
        full = jnp.zeros(
            maps.shape[:-2] + (full_count, maps.shape[-1]),
            dtype=maps.dtype,
        )
        return full.at[..., : self.Ntheta, :].set(maps)

    def pair_stokes_alms_native(self, lmax, source_indices):
        """Transform only unique native source indices through Croissant."""
        import croissant as cro

        if not hasattr(cro, "PairStokesBeam"):
            raise ImportError(
                "Full-Stokes response transforms require the companion "
                "Croissant polarization branch."
            )
        source_indices = np.asarray(source_indices, dtype=np.int32).reshape(-1)
        if source_indices.size == 0:
            raise ValueError("source_indices must be nonempty.")
        missing = [
            int(index)
            for index in source_indices
            if (int(lmax), int(index)) not in self._alm_cache
        ]
        if missing:
            pair_chunk, frequency_chunk = self._transform_chunk_shape(lmax)
            transformed = {}
            for frequency_start in range(0, len(missing), frequency_chunk):
                frequency_indices = missing[
                    frequency_start : frequency_start + frequency_chunk
                ]
                for pair_start in range(0, len(self.pairs), pair_chunk):
                    selected_pairs = self.pairs[
                        pair_start : pair_start + pair_chunk
                    ]
                    maps = self._full_sphere_maps(
                        self.pair_stokes_maps_for_pairs(
                            selected_pairs,
                            np.asarray(frequency_indices, dtype=np.int32),
                        )
                    )
                    pair_beam = cro.PairStokesBeam(
                        maps,
                        self.freq[frequency_indices],
                        selected_pairs,
                        sampling="mwss",
                        convention="IAU",
                        units="m^2",
                        frequency_units="MHz",
                        frame="topo",
                        tangent_basis=self.tangent_basis,
                        baseline_direction="a<=b",
                        visibility_definition="<v_a v_b*>",
                        horizon=jnp.asarray(
                            np.linspace(0.0, 180.0, maps.shape[-2])[:, None]
                            <= 90.0
                        ),
                    )
                    if lmax > pair_beam.lmax:
                        raise ValueError(
                            f"Requested lmax={lmax} exceeds response limit "
                            f"{pair_beam.lmax}."
                        )
                    alms = pair_beam.compute_alm(lmax=int(lmax))
                    for frequency_position, native_index in enumerate(
                        frequency_indices
                    ):
                        for pair_position in range(len(selected_pairs)):
                            transformed[
                                (native_index, pair_start + pair_position)
                            ] = alms[pair_position, frequency_position]
            for native_index in missing:
                self._alm_cache[(int(lmax), native_index)] = jnp.stack(
                    [
                        transformed[(native_index, pair_index)]
                        for pair_index in range(len(self.pairs))
                    ],
                    axis=0,
                )
            self._transform_count += len(missing)
        return jnp.stack(
            [
                self._alm_cache[(int(lmax), int(index))]
                for index in source_indices
            ],
            axis=1,
        )

    def pair_stokes_alms(self, lmax, target_freqs):
        """Return target-aligned physical W alms and their FrequencyMap."""
        frequency_map = FrequencyMap.build(target_freqs, self.freq)
        native = self.pair_stokes_alms_native(
            lmax,
            frequency_map.source_indices,
        )
        native_frequency = self.freq[frequency_map.source_indices]
        wavelength = c / (native_frequency * 1e6)
        scale = jnp.asarray(
            VACUUM_IMPEDANCE_OHM / wavelength**2
        )
        scaled = native * scale[None, :, None, None, None]
        target_major = frequency_map.from_unique(
            jnp.swapaxes(scaled, 0, 1)
        )
        return jnp.swapaxes(target_major, 0, 1), frequency_map

    def target_matrices(self, target_freqs):
        """Interpolate ZA on targets and derive Rsky from the same W monopole."""
        alms, frequency_map = self.pair_stokes_alms(0, target_freqs)
        monopole = alms[:, :, 0, 0, 0]
        pair_rsky = 0.25 * monopole * jnp.sqrt(4 * jnp.pi)
        Rsky = assemble_pair_matrix(
            jnp.swapaxes(pair_rsky, 0, 1),
            self.pairs,
        )
        ZA = frequency_map.from_native(self.ZA)
        dissipative = 0.5 * (
            ZA + jnp.swapaxes(ZA.conjugate(), -1, -2)
        )
        Rmoon = dissipative - Rsky
        return ZA, Rsky, Rmoon, frequency_map

    def rotate(self, degrees):
        """Rotate the complete instrument by an integer number of phi bins."""
        if self.Nphi < 2:
            raise ValueError("Response phi grid is too short to rotate.")
        step = float(self.phi_deg[1] - self.phi_deg[0])
        bins = int(round(float(degrees) / step))
        if not np.isclose(bins * step, degrees):
            raise ValueError(
                f"Rotation must be a multiple of the phi step {step} deg."
            )
        result = copy.copy(self)

        def rotate_field(field):
            unique = field[..., :-1]
            rotated = jnp.roll(unique, shift=-bins, axis=-1)
            return jnp.concatenate((rotated, rotated[..., :1]), axis=-1)

        result.H_theta = rotate_field(self.H_theta)
        result.H_phi = rotate_field(self.H_phi)
        result._alm_cache = {}
        result._transform_count = 0
        result.content_hash = response_payload_hash(
            freq=result.freq,
            theta_deg=result.theta_deg,
            phi_deg=result.phi_deg,
            H_theta=result.H_theta,
            H_phi=result.H_phi,
            ZA=result.ZA,
            Rsky=result.Rsky_native,
            Rmoon=result.Rmoon_native,
            metadata=result.header,
        )
        return result

    @property
    def native_transform_count(self):
        """Number of native frequency endpoints transformed by this instance."""
        return self._transform_count

    def loaded_response(self, ZL, freq_ndx=None):
        """Return loaded grid response ``M H`` for selected native channels."""
        indices = (
            jnp.arange(self.Nfreq)
            if freq_ndx is None
            else jnp.atleast_1d(jnp.asarray(freq_ndx, dtype=jnp.int32))
        )
        ZL = jnp.asarray(ZL)
        if ZL.shape[0] == self.Nfreq:
            ZL = ZL[indices]
        M = loading_matrix(self.ZA[indices], ZL)
        H = jnp.stack(
            (self.H_theta[:, indices], self.H_phi[:, indices]),
            axis=2,
        )
        return jnp.einsum("fab,bftxy->aftxy", M, H)

    def loaded_response_at(
        self,
        ZL,
        theta_rad,
        phi_rad,
        freq_mhz,
    ):
        """Bilinearly sample the loaded voltage response for diagnostics."""
        frequency_map = FrequencyMap.build(freq_mhz, self.freq)
        Ht = frequency_map.from_native(jnp.swapaxes(self.H_theta, 0, 1))
        Hp = frequency_map.from_native(jnp.swapaxes(self.H_phi, 0, 1))
        ZA = frequency_map.from_native(self.ZA)
        M = loading_matrix(ZA, ZL)

        theta_value = float(theta_rad)
        phi_value = float(phi_rad) % (2 * np.pi)
        theta_grid = np.radians(self.theta_deg)
        phi_grid = np.radians(self.phi_deg[:-1])
        if theta_value < theta_grid[0] or theta_value > theta_grid[-1]:
            raise ValueError("theta is outside the stored response region.")
        ti = np.searchsorted(theta_grid, theta_value, side="right")
        thi = min(max(ti, 1), theta_grid.size - 1)
        tlo = thi - 1
        ta = (theta_value - theta_grid[tlo]) / (
            theta_grid[thi] - theta_grid[tlo]
        )
        pi = np.searchsorted(phi_grid, phi_value, side="right")
        plo = (pi - 1) % phi_grid.size
        phi_index = pi % phi_grid.size
        phi_lo = phi_grid[plo]
        phi_hi = phi_grid[phi_index]
        if phi_index == 0:
            phi_hi += 2 * np.pi
        phi_eval = phi_value if phi_value >= phi_lo else phi_value + 2 * np.pi
        pa = (phi_eval - phi_lo) / (phi_hi - phi_lo)

        def angular_sample(values):
            low = (1 - pa) * values[..., tlo, plo] + pa * values[..., tlo, phi_index]
            high = (1 - pa) * values[..., thi, plo] + pa * values[..., thi, phi_index]
            return (1 - ta) * low + ta * high

        H = jnp.stack((angular_sample(Ht), angular_sample(Hp)), axis=-1)
        return jnp.einsum("fab,fbc->fac", M, H)

    def pair_stokes_at(
        self,
        theta_rad,
        phi_rad,
        target_freqs,
    ):
        """Sample the frequency-interpolated physical W pair kernel."""
        frequency_map = FrequencyMap.build(target_freqs, self.freq)
        native = self.all_pair_stokes_maps()
        wavelength = c / (self.freq * 1e6)
        scaled = native * jnp.asarray(
            VACUUM_IMPEDANCE_OHM / wavelength**2
        )[None, :, None, None, None]
        target = _interpolate_axis_one(scaled, frequency_map)
        return _sample_periodic_maps(
            target,
            self.theta_deg,
            self.phi_deg,
            theta_rad,
            phi_rad,
        )

    def sky_coupling_check(self, tolerance=1e-10):
        """Return eigenvalue diagnostics for native Moon resistance."""
        hermitian_error = jnp.max(
            jnp.abs(
                self.Rmoon_native
                - jnp.swapaxes(self.Rmoon_native.conjugate(), -1, -2)
            ),
            axis=(-2, -1),
        )
        eigenvalues = jnp.linalg.eigvalsh(self.Rmoon_native)
        return {
            "hermitian_error": hermitian_error,
            "eigenvalues": eigenvalues,
            "physical": jnp.all(eigenvalues >= -tolerance),
        }


FourPortBeam = InstrumentResponse


def assemble_pair_matrix(pair_values, pairs=PORT_PAIRS):
    """Assemble Hermitian matrices from values for unique ``a <= b`` pairs."""
    values = jnp.asarray(pair_values)
    if values.shape[-1] != len(pairs):
        raise ValueError(
            f"Last axis must contain {len(pairs)} pair values."
        )
    result = jnp.zeros(values.shape[:-1] + (4, 4), dtype=values.dtype)
    for index, (a, b) in enumerate(pairs):
        result = result.at[..., a, b].set(values[..., index])
        if a != b:
            result = result.at[..., b, a].set(values[..., index].conjugate())
    return result


def _interpolate_axis_one(values, frequency_map):
    target_major = frequency_map.from_unique(
        jnp.swapaxes(
            values[:, frequency_map.source_indices],
            0,
            1,
        )
    )
    return jnp.swapaxes(target_major, 0, 1)


def _sample_periodic_maps(
    values,
    theta_deg,
    phi_deg,
    theta_rad,
    phi_rad,
):
    """Bilinearly sample arrays whose final axes are theta and wrapped phi."""
    theta_value = float(theta_rad)
    phi_value = float(phi_rad) % (2 * np.pi)
    theta_grid = np.radians(np.asarray(theta_deg, dtype=np.float64))
    phi_grid = np.radians(np.asarray(phi_deg[:-1], dtype=np.float64))
    if theta_value < theta_grid[0] or theta_value > theta_grid[-1]:
        raise ValueError("theta is outside the stored response region.")
    ti = np.searchsorted(theta_grid, theta_value, side="right")
    thi = min(max(ti, 1), theta_grid.size - 1)
    tlo = thi - 1
    ta = (theta_value - theta_grid[tlo]) / (
        theta_grid[thi] - theta_grid[tlo]
    )
    pi = np.searchsorted(phi_grid, phi_value, side="right")
    plo = (pi - 1) % phi_grid.size
    phi_index = pi % phi_grid.size
    phi_lo = phi_grid[plo]
    phi_hi = phi_grid[phi_index]
    if phi_index == 0:
        phi_hi += 2 * np.pi
    phi_eval = phi_value if phi_value >= phi_lo else phi_value + 2 * np.pi
    pa = (phi_eval - phi_lo) / (phi_hi - phi_lo)
    low = (
        (1 - pa) * values[..., tlo, plo]
        + pa * values[..., tlo, phi_index]
    )
    high = (
        (1 - pa) * values[..., thi, plo]
        + pa * values[..., thi, phi_index]
    )
    return (1 - ta) * low + ta * high
