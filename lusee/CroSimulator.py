"""
CroSimulator — JAX-differentiable radio telescope simulator.

The constructor precomputes observation-fixed quantities (MEPA rotation
matrices, time-evolution phases).  ``simulate()`` takes the differentiable
inputs — beam ALMs, sky ALMs, ground temperature, ground power — as
explicit arguments, so ``jax.jit`` and ``jax.grad`` work naturally::

    sim = CroSimulator(obs, beams, sky, Tground=0, ...)
    wf  = sim.simulate(sim.beam_alms, sim.sky_mepa,
                       sim.Tground, sim.ground_power)

    # gradient w.r.t. beam ALMs
    grad_fn = jax.grad(lambda b: jnp.sum(
        sim.simulate(b, sim.sky_mepa, sim.Tground, sim.ground_power) ** 2))
    g = grad_fn(sim.beam_alms)
"""

from functools import partial
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import os
import sys

import croissant as cro
import s2fft
from lunarsky import LunarTopo

from .BeamCouplings import BeamCouplings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_plot_dir():
    """Default save directory for diagnostic plots."""
    return os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "simulation", "output", "figures")
    )


def _prepare_beams(beams, combinations, lmax, freq_ndx_beam,
                   cross_power=None, extra_opts=None):
    """Prepare beam ALM products for all antenna combinations.

    Returns a list of (i, j, beamreal, beamimag, groundPowerReal,
    groundPowerImag) tuples.
    """
    if cross_power is None:
        cross_power = BeamCouplings()
    if extra_opts is None:
        extra_opts = {}
    efbeams = []
    combinations = [(int(i), int(j)) for i, j in combinations]
    for i, j in combinations:
        bi, bj = beams[i], beams[j]
        print(f"  intializing beam combination {bi.id} x {bj.id} ...")
        norm = np.sqrt(bi.gain_conv[freq_ndx_beam] * bj.gain_conv[freq_ndx_beam])
        beamreal, beamimag = bi.get_healpix_alm(
            lmax, freq_ndx=freq_ndx_beam, other=bj,
            return_I_stokes_only=True, return_complex_components=True,
        )
        beamreal = beamreal * norm[:, None]
        if beamimag is not None:
            beamimag = beamimag * norm[:, None]
        if i == j:
            groundPowerReal = np.array([1 - np.real(br[0]) / np.sqrt(4 * np.pi)
                                        for br in beamreal])
            beamimag = None
            groundPowerImag = 0.0
        else:
            cp = cross_power.Ex_coupling(bi, bj, freq_ndx_beam)
            print(f"    cross power is {cp[0]} ... {cp[-1]} ")
            groundPowerReal = np.array([c - np.real(br[0]) / np.sqrt(4 * np.pi)
                                        for br, c in zip(beamreal, cp)])
            groundPowerImag = np.array([0 - np.real(bi_[0]) / np.sqrt(4 * np.pi)
                                        for bi_ in beamimag])
        if "dump_beams" in extra_opts:
            np.save(bi.id + bj.id, beamreal)
        efbeams.append((i, j, beamreal, beamimag, groundPowerReal, groundPowerImag))
    return efbeams


def _hp_to_2d(alms, sim_L):
    """Convert a stack of healpy-packed ALMs to s2fft 2D layout."""
    return jnp.stack([
        s2fft.sampling.reindex.flm_hp_to_2d_fast(jnp.asarray(a), sim_L)
        for a in alms
    ])


def _plot_sky_beam_healpix(sky_alm, beam_alm, nside, lmax,
                           save_dir="output/figures",
                           save_filename="sky_beam_healpix_cro.png",
                           title_prefix=""):
    """Plot sky and beam as healpix mollweide maps (side-effect, not JAX)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import healpy as hp

    outpath = os.path.join(save_dir, save_filename)
    os.makedirs(save_dir, exist_ok=True)
    sky_map = hp.alm2map(np.asarray(sky_alm, dtype=np.complex128), nside, lmax=lmax)
    beam_map = hp.alm2map(np.asarray(beam_alm, dtype=np.complex128), nside, lmax=lmax)
    plt.figure(figsize=(12, 5))
    hp.mollview(sky_map, title=(title_prefix + " Sky").strip(), sub=(1, 2, 1))
    hp.mollview(beam_map, title=(title_prefix + " Beam").strip(), sub=(1, 2, 2))
    plt.savefig(outpath, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  plot_sky_and_beam: saved {outpath}")


# ---------------------------------------------------------------------------
# CroSimulator
# ---------------------------------------------------------------------------

class CroSimulator:
    """JAX-differentiable Croissant simulator.

    The constructor precomputes observation-fixed quantities and stores
    default values for the differentiable inputs as attributes.
    ``simulate()`` takes those differentiable inputs as explicit arguments,
    making ``jax.jit`` / ``jax.grad`` straightforward.

    Precomputed (fixed per observation, stored on ``self``):
        ``eul_topo``, ``dl_topo``, ``phases`` — MEPA rotation & time phases.

    Default differentiable arrays (also stored on ``self`` for convenience):
        ``beam_alms``, ``sky_mepa``, ``Tground``, ``ground_power``.

    :param obs: Observation (time range, location).
    :param beams: Instrument beams (lusee.Beam / BeamGauss).
    :param sky_model: Sky model (must have ``frame == "galactic"``).
    :param Tground: Ground temperature [K].
    :param combinations: Beam combination indices, e.g. ``[(0,0),(1,1),(0,2)]``.
    :param freq: Frequencies in MHz (default: from beams).
    :param lmax: Maximum spherical harmonic degree.
    :param cross_power: BeamCouplings for cross terms.
    :param extra_opts: Dict for plotting / dump_beams options.
    """

    def __init__(self, obs, beams, sky_model, Tground=200.0,
                 combinations=((0, 0), (1, 1), (0, 2), (1, 3), (1, 2)),
                 freq=None, lmax=128, cross_power=None, extra_opts=None):
        if extra_opts is None:
            extra_opts = {}

        # ── frequency index setup ──
        if freq is None:
            freq = beams[0].freq
        freq = np.asarray(freq)
        freq_ndx_beam = []
        freq_ndx_sky = []
        for f in freq:
            try:
                ndx = list(beams[0].freq).index(f)
            except ValueError:
                print(f"Error: Frequency {f} does not exist in beams.")
                sys.exit(1)
            freq_ndx_beam.append(ndx)
            try:
                ndx = list(sky_model.freq).index(f)
            except ValueError:
                print(f"Error: Frequency {f} does not exist in sky model.")
                sys.exit(1)
            freq_ndx_sky.append(ndx)

        self.lmax = lmax
        self.freq = freq
        self._freq_ndx_beam = freq_ndx_beam

        # prepare beams (numpy) to stack into JAX arrays
        combinations = [(int(i), int(j)) for i, j in combinations]
        self._combinations = combinations
        efbeams = _prepare_beams(
            beams, combinations, lmax, freq_ndx_beam,
            cross_power=cross_power, extra_opts=extra_opts,
        )

        sim_L = lmax + 1
        all_beam_2d = []
        all_ground = []
        channel_layout = []
        for ci, cj, beamreal, beamimag, gpr, gpi in efbeams:
            all_beam_2d.append(_hp_to_2d(beamreal, sim_L))
            all_ground.append(np.asarray(gpr))
            channel_layout.append(f"{ci}{cj}R")
            if ci != cj:
                all_beam_2d.append(_hp_to_2d(beamimag, sim_L))
                all_ground.append(np.asarray(gpi))
                channel_layout.append(f"{ci}{cj}I")

        self.beam_alms = jnp.stack(all_beam_2d)       # (Nch, Nf, L, 2L-1)
        self.ground_power = jnp.stack(all_ground)      # (Nch, Nf)
        self.Tground = jnp.asarray(float(Tground))
        self.channel_layout = tuple(channel_layout)

        # ── MEPA setup (fixed per observation) ──
        if sky_model.frame != "galactic":
            raise NotImplementedError(
                f"CroSimulator requires galactic sky frame, got {sky_model.frame}"
            )
        times = obs.times
        topo = LunarTopo(obstime=times[0], location=obs.loc)
        eul_topo, dl_topo = cro.rotations.generate_euler_dl(lmax, topo, "mepa")
        self.eul_topo = tuple(float(a) for a in eul_topo)
        self.dl_topo = jnp.asarray(dl_topo)

        sky_gal = sky_model.get_alm(freq_ndx_sky, freq)
        sky_2d = jnp.stack([
            s2fft.sampling.reindex.flm_hp_to_2d_fast(jnp.asarray(s), sim_L)
            for s in sky_gal
        ])
        et = cro.rotations.jd_to_et(times[0].jd)
        self.sky_mepa = jnp.asarray(cro.rotations.gal2mepa(sky_2d, et=et))

        delta_t_sec = np.arange(len(times), dtype=float) * obs.deltaT_sec
        self.phases = jnp.asarray(cro.simulator.rot_alm_z(lmax, times=delta_t_sec))
        self._nside = getattr(sky_model, "Nside", 64)

        # ── I/O metadata ──
        self._obs_meta = {
            "version":      0.1,
            "lunar_day":    obs.time_range,
            "lun_lat_deg":  obs.lun_lat_deg,
            "lun_long_deg": obs.lun_long_deg,
            "lun_height_m": obs.lun_height_m,
            "deltaT_sec":   obs.deltaT_sec,
        }
        self._beams_ZReIm = tuple(
            (b.ZRe[freq_ndx_beam], b.ZIm[freq_ndx_beam]) for b in beams
        )

    # -----------------------------------------------------------------
    # Pure-JAX simulation
    # -----------------------------------------------------------------

    def simulate(self, beam_alms=None, sky_mepa=None,
                 Tground=None, ground_power=None):
        """Run the beam-sky convolution.

        All arguments are optional — defaults come from the values computed
        in ``__init__``.  Pass explicit JAX arrays to differentiate through
        them with ``jax.grad``.

        **Pure JAX** when called with explicit arguments (no Python
        side-effects, compatible with ``jax.jit`` / ``jax.grad``).

        :param beam_alms: ``(Nchannels, Nfreq, L, 2L-1)`` beam ALMs in
            topo-frame s2fft 2D layout.
        :param sky_mepa: ``(Nfreq, L, 2L-1)`` sky ALMs in MEPA frame.
            The galactic→MEPA rotation is a fixed linear transform applied
            once in ``__init__``; sampling/optimizing in MEPA avoids
            repeating that rotation every forward pass and the gradient
            landscape is identical (just rotated).
        :param Tground: Scalar ground temperature [K].
        :param ground_power: ``(Nchannels, Nfreq)`` ground power fractions.
        :returns: Waterfall array, shape ``(Ntimes, Nchannels, Nfreq)``.
        """
        if beam_alms is None:
            beam_alms = self.beam_alms
        if sky_mepa is None:
            sky_mepa = self.sky_mepa
        if Tground is None:
            Tground = self.Tground
        if ground_power is None:
            ground_power = self.ground_power

        sim_L = self.lmax + 1
        topo2mepa = partial(
            s2fft.utils.rotation.rotate_flms,
            L=sim_L, rotation=self.eul_topo, dl_array=self.dl_topo,
        )
        # Rotate all beam channels: vmap over channels, then over freqs.
        beam_mepa = jax.vmap(jax.vmap(topo2mepa))(beam_alms)
        # Convolve all channels with sky at once.
        vis = cro.multipair.multi_convolve(beam_mepa, sky_mepa, self.phases)
        # vis: (Nchannels, Ntimes, Nfreq)
        T = vis.real / (4.0 * jnp.pi) + Tground * ground_power[:, None, :]
        return jnp.moveaxis(T, 0, 1)   # (Ntimes, Nchannels, Nfreq)

    # -----------------------------------------------------------------
    # I/O
    # -----------------------------------------------------------------

    def write_fits(self, out_file, result=None):
        """Write simulation result to FITS.

        :param out_file: Output file path.
        :param result: Waterfall array; if *None*, calls ``simulate()``.
        """
        import fitsio
        if result is None:
            result = self.simulate()
        fits = fitsio.FITS(out_file, 'rw', clobber=True)
        fits.write(np.asarray(result), header=self._obs_meta, extname='data')
        fits.write(np.asarray(self.freq), extname='freq')
        fits.write(np.array(self._combinations), extname='combinations')
        for i, (zre, zim) in enumerate(self._beams_ZReIm):
            fits.write(np.asarray(zre), extname=f'ZRe_{i}')
            fits.write(np.asarray(zim), extname=f'ZIm_{i}')

    # -----------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------

    def plot_sky_beam(self, freq_idx=0, save_dir=None, save_filename=None):
        """Plot sky and beam at one frequency as healpix mollweide maps.

        :param freq_idx: Frequency index to plot.
        :param save_dir: Output directory (default: simulation/output/figures).
        :param save_filename: Output filename.
        """
        sim_L = self.lmax + 1
        nf = len(self.freq)
        if nf == 0:
            raise ValueError("Cannot plot: no frequencies")
        if not (0 <= freq_idx < nf):
            freq_idx = max(0, min(freq_idx, nf - 1))
            warnings.warn(
                f"freq_idx={freq_idx} out of bounds for len(freq)={nf}.",
                UserWarning, stacklevel=2,
            )
        topo2mepa = partial(
            s2fft.utils.rotation.rotate_flms,
            L=sim_L, rotation=self.eul_topo, dl_array=self.dl_topo,
        )
        beam0_mepa = jax.vmap(topo2mepa)(self.beam_alms[0])
        sky_packed = s2fft.sampling.reindex.flm_2d_to_hp_fast(
            np.asarray(self.sky_mepa[freq_idx]), sim_L)
        beam_packed = s2fft.sampling.reindex.flm_2d_to_hp_fast(
            np.asarray(beam0_mepa[freq_idx]), sim_L)
        _plot_sky_beam_healpix(
            sky_packed, beam_packed, self._nside, self.lmax,
            save_dir=save_dir or _default_plot_dir(),
            save_filename=save_filename or "sky_beam_healpix_cro.png",
            title_prefix=f"Croissant at {self.freq[freq_idx]} MHz ",
        )
