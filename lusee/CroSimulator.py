from functools import partial
from importlib.resources import files
import warnings

from .Observation import Observation
from .Beam import Beam
from .BeamCouplings import BeamCouplings
from .SimulatorBase import SimulatorBase, default_plot_sky_beam_dir, get_topo_z_rotation_angles
from .spice_utils import ensure_lunarsky_moon_frame
import numpy as np
import fitsio
import sys
import os
import jax.numpy as jnp
import croissant as cro
from croissant.multipair import multi_convolve
import jax
from lunarsky import LunarTopo
import spiceypy as spice
import s2fft

"""
CroSimulator: same inputs as DefaultSimulator (beam, sky, obs, etc.) but uses
the Croissant engine for the actual simulation (MEPA frame, rot_alm_z phases,
croissant.multipair.multi_convolve over beam combinations). Freq, time range,
and antenna location come from the observation object (config). Croissant
handles single polarization / single dipole per beam; effective beams are
still one combination each, batched at the convolution step.
"""


_SPICE_LUNAR_KERNELS_LOADED = False


def _ensure_spice_lunar_frames():
    """Load Lunarsky's bundled lunar frame definitions for Croissant.

    Lunarsky 1.x evaluates its own coordinate transforms and no longer loads
    these SPICE kernels as an import side effect.  Croissant still uses SPICE
    directly for its ``MOON_ME`` to ``J2000`` transform, so register the
    packaged frame and orientation kernels before asking it for rotations.
    """
    global _SPICE_LUNAR_KERNELS_LOADED
    if _SPICE_LUNAR_KERNELS_LOADED:
        return

    kernel_root = files("lunarsky.data")
    for relative_path in (
        "fk/satellites/moon_080317.tf",
        "fk/satellites/moon_assoc_me.tf",
        "pck/moon_pa_de421_1900-2050.bpc",
    ):
        spice.furnsh(str(kernel_root.joinpath(relative_path)))
    _SPICE_LUNAR_KERNELS_LOADED = True


class CroSimulator(SimulatorBase):
    """
    Croissant simulator: same inputs as DefaultSimulator (obs, beams, sky_model,
    combinations, freq, lmax) 

    - Freq, time grid, and antenna location are taken from the observation
      object (set from config). [luseepy.observation class]
    - Lunar topo frame is built from obs class (obstime=first time, location=obs.loc).
    - Beam and sky are transformed to MEPA; time evolution uses
      croissant.simulator.rot_alm_z (moon sidereal rotation).
    - Croissant handles single polarization / single dipole per beam; real-part
      beams for all combinations are convolved in one multipair batch; cross
      imaginary beams are batched separately when present.
    - Sky gal→MEPA once (epoch-aware), beam topo(t0)→MEPA once, rot_alm_z(dt) for time evolution, then convolve.
    - Output layout is (N_times, N_combos, N_freq).

    :param obs: Observation (time range, deltaT_sec, lun_lat_deg, lun_long_deg)
    :param beams: Instrument beams [luseepy.beam class]
    :param sky_model: Sky model [luseepy.skymodels class]
    :param combinations: Beam combination indices [(0,0),(1,1),(0,2),(1,3),(1,2)]
    :param lmax: Maximum l
    :param Tground: Ground temperature [K]
    :param freq: Frequencies in MHz (from config / obs)
    :param cross_power: BeamCouplings for cross terms [luseepy.BeamCouplings class]
    :param extra_opts: optional dict, e.g. cache_transform, dump_beams (see DefaultSimulator),
        freq_idx_plot (int): index of frequency at which to plot sky and beam.
    """

    def __init__ (self, obs, beams, sky_model, Tground = 200.0,
                  combinations = [(0,0),(1,1),(0,2),(1,3),(1,2)], freq = None,
                  lmax = 128, cross_power = None,
                  extra_opts = {}):
        super().__init__(obs, beams, sky_model, Tground, combinations, freq)
        ensure_lunarsky_moon_frame()
        self.lmax = lmax
        self.extra_opts = extra_opts
        self.cross_power = cross_power if (cross_power is not None) else BeamCouplings()
        self._prepare_beams_for_simulation(beams, combinations)
        self._mepa_plan_cache = {}

    def _prepare_beams_for_simulation(self, beams, combinations):
        if all(getattr(beam, "is_jax_pytree_beam", False) for beam in beams):
            self._prepare_beams_jax(beams, combinations)
            return
        self.prepare_beams(beams, combinations)

    def _prepare_beams_jax(self, beams, combinations):
        self.beams = beams
        self.efbeams = []
        self.combinations = [(int(i), int(j)) for i, j in combinations]
        beam_idx = jnp.asarray(np.array(self.freq_ndx_beam, dtype=np.int32))

        for i, j in self.combinations:
            bi, bj = beams[i], beams[j]
            print (f"  intializing beam combination {bi.id} x {bj.id} ...")
            norm = jnp.sqrt(
                jnp.asarray(bi.gain_conv)[beam_idx]
                * jnp.asarray(bj.gain_conv)[beam_idx]
            )
            beamreal, beamimag = bi.get_healpix_alm(
                self.lmax,
                freq_ndx=self.freq_ndx_beam,
                other=bj,
                return_I_stokes_only=True,
                return_complex_components=True,
            )
            beamreal = jnp.asarray(beamreal) * norm[:, None]
            if beamimag is not None:
                beamimag = jnp.asarray(beamimag) * norm[:, None]

            if i==j:
                groundPowerReal = 1.0 - jnp.real(beamreal[:,0]) / jnp.sqrt(4*jnp.pi)
                beamimag = None
                groundPowerImag = 0.0
            else:
                cross_power = jnp.asarray(self.cross_power.Ex_coupling(bi,bj,self.freq_ndx_beam))
                print (f"    cross power is {cross_power[0]} ... {cross_power[-1]} ")
                groundPowerReal = cross_power - jnp.real(beamreal[:,0]) / jnp.sqrt(4*jnp.pi)
                groundPowerImag = -jnp.real(beamimag[:,0]) / jnp.sqrt(4*jnp.pi)
            if "dump_beams" in getattr(self, "extra_opts", {}):
                np.save(bi.id+bj.id, np.asarray(beamreal))
            self.efbeams.append((i,j,beamreal, beamimag, groundPowerReal,
                                 groundPowerImag))

            
                                
    def simulate(self, times=None, *, sky=None, beam=None):
        """
        Simulate using Croissant.

        :param times: List of times; if None, use self.obs.times (from config).
        :param sky: Optional pytree sky object with ``.get_alm(ndx, freq)``
            and ``.frame == "galactic"``.  When provided, the sky's
            ``get_alm()`` is called inside the traced computation so
            ``jax.grad`` flows through the pytree leaves.
        :param beam: Optional pytree beam object with an ``.efbeams``
            property returning a list of
            ``(ci, cj, beamreal, beamimag, groundPowerReal, groundPowerImag)``
            tuples.  Arrays in the tuples are JAX-traced so ``jax.grad``
            flows through the pytree leaves.
        :returns: Waterfall (N_times, N_combos_with_imag, N_freq).
        """
        if times is None:
            times = self.obs.times
        ntimes = len(times)
        delta_t = float(self.obs.deltaT_sec)

        sky_model = sky if sky is not None else self.sky_model
        if sky_model.frame != "galactic":
            raise NotImplementedError(
                f"CroSimulator requires galactic sky frame, got {sky_model.frame}"
            )
        efbeams = beam.efbeams if beam is not None else self.efbeams
        use_batched = (
            beam is None
            and self.extra_opts.get("batched_mepa", True)
            and not self.extra_opts.get("plot_sky_and_beam", False)
        )
        if use_batched:
            self.result = self._simulate_croissant_mepa_batched(
                times, sky_model=sky_model
            )
        else:
            self.result = self._simulate_croissant_mepa_loop(
                times, ntimes, delta_t, sky_model=sky_model, efbeams=efbeams
            )
        return self.result

    def _mepa_plan_key(self, times):
        return (
            tuple(float(t.jd) for t in times),
            int(self.lmax),
            tuple(float(f) for f in np.asarray(self.freq)),
            int(len(self.efbeams)),
        )

    def _beam_efbeams_to_channel_arrays(self, efbeams):
        beam_channels = []
        ground_channels = []
        for ci, cj, beamreal, beamimag, groundPowerReal, groundPowerImag in efbeams:
            beam_channels.append(jnp.asarray(beamreal))
            gp = jnp.asarray(groundPowerReal)
            if gp.ndim == 0:
                gp = jnp.full((self.Nfreq,), gp)
            ground_channels.append(gp)
            if ci != cj:
                beam_channels.append(jnp.asarray(beamimag))
                gp = jnp.asarray(groundPowerImag)
                if gp.ndim == 0:
                    gp = jnp.full((self.Nfreq,), gp)
                ground_channels.append(gp)
        return jnp.stack(beam_channels), jnp.stack(ground_channels)

    def _get_mepa_plan(self, times):
        key = self._mepa_plan_key(times)
        if key in self._mepa_plan_cache:
            return self._mepa_plan_cache[key]

        sim_L = self.lmax + 1
        topo = LunarTopo(obstime=times[0], location=self.obs.loc)
        eul_topo, dl_topo = crojax.rotations.generate_euler_dl(
            self.lmax, topo, "mepa"
        )
        topo2mepa = partial(
            s2fft.utils.rotation.rotate_flms,
            L=sim_L,
            rotation=eul_topo,
            dl_array=dl_topo,
        )

        et = cro.rotations.jd_to_et(times[0].jd)
        eul_gal, dl_gal = crojax.rotations.generate_euler_dl(
            self.lmax, "galactic", "mepa", et=et
        )
        delta_t_sec = np.arange(len(times), dtype=float) * self.obs.deltaT_sec
        phases = cro.simulator.rot_alm_z(self.lmax, times=delta_t_sec)

        beam_hp, ground_power = self._beam_efbeams_to_channel_arrays(self.efbeams)

        def hp_to_2d(flm_hp):
            return s2fft.sampling.reindex.flm_hp_to_2d_fast(flm_hp, sim_L)

        beam_2d = jax.vmap(jax.vmap(hp_to_2d))(beam_hp)
        beam_mepa = jax.vmap(jax.vmap(topo2mepa))(beam_2d)

        plan = {
            "sim_L": sim_L,
            "phases": jnp.asarray(phases),
            "beam_mepa": jnp.asarray(beam_mepa),
            "ground_power": jnp.asarray(ground_power),
            "gal_eul": eul_gal,
            "gal_dl": dl_gal,
        }
        self._mepa_plan_cache[key] = plan
        return plan

    def _simulate_croissant_mepa_batched(self, times, sky_model=None):
        """Batched MEPA pipeline for fixed beams.

        The observation-dependent phases, frame rotations, and beam MEPA alms
        are cached once. Each forward call only builds the sky alms and performs
        a single channel-batched convolution.
        """
        if sky_model is None:
            sky_model = self.sky_model
        plan = self._get_mepa_plan(times)
        sim_L = plan["sim_L"]

        def hp_to_2d(flm_hp):
            return s2fft.sampling.reindex.flm_hp_to_2d_fast(
                jnp.asarray(flm_hp), sim_L
            )

        sky_gal = sky_model.get_alm(self.freq_ndx_sky)
        sky_2d = jax.vmap(hp_to_2d)(sky_gal)
        gal2mepa = partial(
            s2fft.utils.rotation.rotate_flms,
            L=sim_L,
            rotation=plan["gal_eul"],
            dl_array=plan["gal_dl"],
        )
        sky_mepa = jax.vmap(gal2mepa)(sky_2d)

        vis = jnp.einsum(
            "flm,tm,cflm->tcf",
            jnp.conj(sky_mepa),
            plan["phases"],
            plan["beam_mepa"],
            optimize=True,
        )
        return (
            vis.real / (4.0 * jnp.pi)
            + self.Tground * plan["ground_power"][None, :, :]
        )

    def _simulate_croissant_mepa(self, times, ntimes, delta_t,
                                    sky_model=None, efbeams=None):
        return self._simulate_croissant_mepa_loop(
            times, ntimes, delta_t, sky_model=sky_model, efbeams=efbeams
        )

    def _simulate_croissant_mepa_loop(self, times, ntimes, delta_t,
                                      sky_model=None, efbeams=None):
        """MEPA pipeline: sky gal→MEPA once (epoch-aware), beam topo(t0)→MEPA once,
        rot_alm_z(dt) for time evolution, then convolve."""
        _ensure_spice_lunar_frames()
        if sky_model is None:
            sky_model = self.sky_model
        if efbeams is None:
            efbeams = self.efbeams
        topo = LunarTopo(obstime=times[0], location=self.obs.loc)
        sim_L = self.lmax + 1
        eul_topo, dl_topo = cro.rotations.generate_euler_dl(
            self.lmax, topo, "mepa"
        )
        topo2mepa = partial(
            s2fft.utils.rotation.rotate_flms,
            L=sim_L,
            rotation=eul_topo,
            dl_array=dl_topo,
        )
        sky_gal = sky_model.get_alm(self.freq_ndx_sky)
        sky_2d = jnp.stack([
            s2fft.sampling.reindex.flm_hp_to_2d_fast(jnp.asarray(s_), sim_L)
            for s_ in sky_gal
        ])
        # Use the observation's TDB epoch when defining the MEPA reference frame.
        # Using the generic .jd value here introduces a ~69 s UTC/TDB offset, which
        # is large enough to create geometry-dependent Croissant drift on short windows.
        et = cro.rotations.jd_to_et(times[0].tdb.jd)
        sky_mepa = cro.rotations.gal2mepa(sky_2d, et=et)
        delta_t_sec = np.arange(len(times), dtype=float) * self.obs.deltaT_sec
        phases = cro.simulator.rot_alm_z(self.lmax, times=delta_t_sec)
        norm_factor = 4.0 * np.pi
        # croissant>=5.1.x handles LunarTopo (NEU) to ENU convention
        # internally in rotations.get_rot_mat/generate_euler_dl.
        # Do not apply an additional manual m-dependent phase here.
        # multipair.multi_convolve is raw convolve vmap — no extra normalization.
        # We keep dividing by 4π only (not multipair.compute_visibilities).
        beam_mepas = []
        ground_reals = []
        beamimag_mepas = []
        ground_imags = []
        for ci, cj, beamreal, beamimag, groundPowerReal, groundPowerImag in efbeams:
            beam_2d = jnp.stack([
                s2fft.sampling.reindex.flm_hp_to_2d_fast(
                    jnp.asarray(br_), sim_L
                )
                for br_ in beamreal
            ])
            beam_mepa = jax.vmap(topo2mepa)(beam_2d)
            beam_mepas.append(beam_mepa)
            ground_reals.append(jnp.asarray(groundPowerReal))
            if ci != cj:
                beamimag_2d = jnp.stack([
                    s2fft.sampling.reindex.flm_hp_to_2d_fast(
                        jnp.asarray(bi_), sim_L
                    )
                    for bi_ in beamimag
                ])
                beamimag_mepa = jax.vmap(topo2mepa)(beamimag_2d)
                beamimag_mepas.append(beamimag_mepa)
                ground_imags.append(jnp.asarray(groundPowerImag))

        if self.extra_opts.get("plot_sky_and_beam") and beam_mepas:
            nf = len(self.freq)
            freq_idx_plot = int(self.extra_opts.get("freq_idx_plot", 0))
            if nf == 0:
                raise ValueError("Cannot plot: no frequencies in self.freq")
            if not (0 <= freq_idx_plot < nf):
                clamped = max(0, min(freq_idx_plot, nf - 1))
                warnings.warn(
                    f"freq_idx_plot={freq_idx_plot} is out of bounds for "
                    f"len(freq)={nf}; using {clamped}.",
                    UserWarning,
                    stacklevel=2,
                )
                freq_idx_plot = clamped
            nside = getattr(self.sky_model, "Nside", 64)
            sky_packed = s2fft.sampling.reindex.flm_2d_to_hp_fast(
                np.asarray(sky_mepa[freq_idx_plot]), self.lmax + 1
            )
            beam_packed = s2fft.sampling.reindex.flm_2d_to_hp_fast(
                np.asarray(beam_mepas[0][freq_idx_plot]), self.lmax + 1
            )
            self._plot_sky_beam_healpix(
                sky_packed,
                beam_packed,
                nside,
                self.lmax,
                save_dir=self.extra_opts.get("plot_dir", default_plot_sky_beam_dir()),
                save_filename=self.extra_opts.get(
                    "plot_filename", "sky_beam_healpix_cro.png"
                ),
                title_prefix=f"Croissant at {self.freq[freq_idx_plot]} MHz ",
            )

        beam_stack = jnp.stack(beam_mepas, axis=0)
        vis_all = multi_convolve(beam_stack, sky_mepa, phases)
        ground_real_stack = jnp.stack(ground_reals, axis=0)
        T_all = (
            vis_all.real / norm_factor
            + self.Tground * ground_real_stack[:, jnp.newaxis, :]
        )

        if beamimag_mepas:
            imag_stack = jnp.stack(beamimag_mepas, axis=0)
            vis_imag_all = multi_convolve(imag_stack, sky_mepa, phases)
            ground_imag_stack = jnp.stack(ground_imags, axis=0)
            Timag_all = (
                vis_imag_all.real / norm_factor
                + self.Tground * ground_imag_stack[:, jnp.newaxis, :]
            )
        else:
            Timag_all = None

        combo_results = []
        imag_cursor = 0
        for k, (ci, cj, _, _, _, _) in enumerate(efbeams):
            T = T_all[k]
            if ci == cj:
                combo_results.append((T, None))
            else:
                combo_results.append((T, Timag_all[imag_cursor]))
                imag_cursor += 1
        wfall = []
        for ti in range(ntimes):
            res = []
            for T, Timag in combo_results:
                res.append(T[ti])
                if Timag is not None:
                    res.append(Timag[ti])
            wfall.append(res)
        return jnp.asarray(wfall)
