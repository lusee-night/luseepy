from functools import partial
import warnings

from .Observation import Observation
from .Beam import Beam
from .BeamCouplings import BeamCouplings
from .SimulatorBase import SimulatorBase, default_plot_sky_beam_dir, get_topo_z_rotation_angles
import numpy as np
import fitsio
import sys
import os
import jax.numpy as jnp
import croissant as cro
import croissant.jax as crojax
import jax
from lunarsky import LunarTopo
import s2fft

"""
CroSimulator: same inputs as DefaultSimulator (beam, sky, obs, etc.) but uses
the Croissant engine for the actual simulation (MEPA frame, rot_alm_z phases,
crojax.simulator.convolve). Freq, time range, and antenna location come from
the observation object (config). Croissant currently supports single
polarization / single dipole per beam; one beam combination at a time
"""


class CroSimulator(SimulatorBase):
    """
    Croissant simulator: same inputs as DefaultSimulator (obs, beams, sky_model,
    combinations, freq, lmax) 

    - Freq, time grid, and antenna location are taken from the observation
      object (set from config). [luseepy.observation class]
    - Lunar topo frame is built from obs class (obstime=first time, location=obs.loc).
    - Beam and sky are transformed to MEPA; time evolution uses
      crojax.simulator.rot_alm_z (moon sidereal rotation).
    - Croissant handles single polarization / single dipole per beam; each
      combination is convolved separately to match DefaultSimulator output shape.
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
        self.lmax = lmax
        self.extra_opts = extra_opts
        self.cross_power = cross_power if (cross_power is not None) else BeamCouplings()
        self._prepare_beams_for_simulation(beams, combinations)

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

            
                                
    def simulate(self, times=None):
        """
        Simulate using Croissant.
        freq, time grid, and antenna location from
        observation; beam and sky transformed to MEPA; rot_alm_z phases;
        crojax.simulator.convolve for sky and beam convolution.
        :param times: List of times; if None, use self.obs.times (from config).
        :returns: Waterfall (N_times, N_combos_with_imag, N_freq), same as DefaultSimulator.
        """
        if times is None:
            times = self.obs.times
        ntimes = len(times)
        delta_t = float(self.obs.deltaT_sec)

        if self.sky_model.frame != "galactic":
            raise NotImplementedError(
                f"CroSimulator requires galactic sky frame, got {self.sky_model.frame}"
            )
        self.result = self._simulate_croissant_mepa(times, ntimes, delta_t)
        return self.result

    def _simulate_croissant_mepa(self, times, ntimes, delta_t):
        """MEPA pipeline: sky gal→MEPA once (epoch-aware), beam topo(t0)→MEPA once,
        rot_alm_z(dt) for time evolution, then convolve."""
        topo = LunarTopo(obstime=times[0], location=self.obs.loc)
        sim_L = self.lmax + 1
        eul_topo, dl_topo = crojax.rotations.generate_euler_dl(
            self.lmax, topo, "mepa"
        )
        topo2mepa = partial(
            s2fft.utils.rotation.rotate_flms,
            L=sim_L,
            rotation=eul_topo,
            dl_array=dl_topo,
        )
        sky_gal = self.sky_model.get_alm(self.freq_ndx_sky, self.freq)
        sky_2d = jnp.stack([
            s2fft.sampling.reindex.flm_hp_to_2d_fast(jnp.asarray(s_), sim_L)
            for s_ in sky_gal
        ])
        et = cro.rotations.jd_to_et(times[0].jd)
        sky_mepa = cro.rotations.gal2mepa(sky_2d, et=et)
        delta_t_sec = np.arange(len(times), dtype=float) * self.obs.deltaT_sec
        phases = cro.simulator.rot_alm_z(self.lmax, times=delta_t_sec)
        norm_factor = 4.0 * np.pi
        # croissant>=5.1.x handles LunarTopo (NEU) to ENU convention
        # internally in rotations.get_rot_mat/generate_euler_dl.
        # Do not apply an additional manual m-dependent phase here.
        combo_results = []
        plot_done = False
        for ci, cj, beamreal, beamimag, groundPowerReal, groundPowerImag in self.efbeams:
            beam_2d = jnp.stack([
                s2fft.sampling.reindex.flm_hp_to_2d_fast(
                    jnp.asarray(br_), sim_L
                )
                for br_ in beamreal
            ])
            beam_mepa = jax.vmap(topo2mepa)(beam_2d)
            if self.extra_opts.get("plot_sky_and_beam") and not plot_done:
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
                sky_packed = s2fft.sampling.reindex.flm_2d_to_hp_fast(np.asarray(sky_mepa[freq_idx_plot]), self.lmax+1)
                beam_packed = s2fft.sampling.reindex.flm_2d_to_hp_fast(np.asarray(beam_mepa[freq_idx_plot]), self.lmax+1)
                self._plot_sky_beam_healpix(
                    sky_packed, beam_packed, nside, self.lmax,
                    save_dir=self.extra_opts.get("plot_dir", default_plot_sky_beam_dir()),
                    save_filename=self.extra_opts.get("plot_filename", "sky_beam_healpix_cro.png"),
                    title_prefix=f"Croissant at {self.freq[freq_idx_plot]} MHz ",
                )
                plot_done = True
            vis = crojax.simulator.convolve(beam_mepa, sky_mepa, phases)
            T = vis.real / norm_factor + self.Tground * groundPowerReal
            combo_results.append((T, None))
            if ci != cj:
                beamimag_2d = jnp.stack([
                    s2fft.sampling.reindex.flm_hp_to_2d_fast(
                        jnp.asarray(bi_), sim_L
                    )
                    for bi_ in beamimag
                ])
                beamimag_mepa = jax.vmap(topo2mepa)(beamimag_2d)
                vis_imag = crojax.simulator.convolve(beamimag_mepa, sky_mepa, phases)
                Timag = vis_imag.real / norm_factor + self.Tground * groundPowerImag
                combo_results[-1] = (T, Timag)
        wfall = []
        for ti in range(ntimes):
            res = []
            for T, Timag in combo_results:
                res.append(T[ti])
                if Timag is not None:
                    res.append(Timag[ti])
            wfall.append(res)
        return jnp.asarray(wfall)
