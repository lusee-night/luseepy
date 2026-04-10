from .Observation import Observation
from .Beam import Beam
from .BeamCouplings import BeamCouplings
from .SimulatorBase import SimulatorBase, default_plot_sky_beam_dir, rot2eul
import numpy as np
import jax
import jax.numpy as jnp
import healpy as hp
import fitsio
import sys
import pickle
import os
import warnings
from scipy.spatial.transform import Rotation as ScipyRotation
import time
from s2fft.recursions.risbo_jax import compute_full


class JaxSimulator(SimulatorBase):
    """
    JAX-backed simulator in luseepy.

    :param obs: Observation parameters, from lusee.observation class
    :type obs: class
    :param beams: Instrument beams, from lusee.beam class
    :type beams: class
    :param sky_model: Simulated model of the sky, from lusee.skymodels
    :type sky_model: class
    :param combinations: Indices for beam combinations/cross correlations to simulate
    :type combinations: list[tuple]
    :param lmax: Maximum l value of beams
    :type lmax: int
    :param Tground: Temperature of lunar ground
    :type Tground: float
    :param freq: Frequencies at which instrument observes sky in MHz. If empty, taken from lusee.beam class.
    :type freq: list[float]
    :param cross_power: Beam coupling model for cross-power terms. If empty, uses :class:`lusee.BeamCouplings`.
    :type cross_power: BeamCouplings
    :param extra_opts: Extra options for simulation. Supports "dump_beams" (saves instrument beams to file),
        "cache_transform" (loads/saves beam transformations from file),
        "force_recompute_cache_transform" (ignores any existing cached transform file),
        "time_batch_size" (int): mini-batch size for time-axis JAX mapping in the
        rotating sky path. If omitted, JaxSimulator chooses a chunk size that keeps
        the vmapped rotation kernel on GPU while avoiding the full-time memory blowup
        at large lmax. "freq_idx_plot" (int): index of frequency at which to plot
        sky and beam.
    :type extra_opts: dict
    """

    def __init__ (self, obs, beams, sky_model, Tground = 200.0,
                  combinations = [(0,0),(1,1),(0,2),(1,3),(1,2)], freq = None,
                  lmax = 128, cross_power = None,
                  extra_opts = {}):
        t_init0 = time.perf_counter()
        super().__init__(obs, beams, sky_model, Tground, combinations, freq)
        self.lmax = lmax
        self.extra_opts = extra_opts
        self.cross_power = cross_power if (cross_power is not None) else BeamCouplings()
        self._timing_enabled = bool(self.extra_opts.get("profile_timing", False))
        self._debug_enabled = bool(self.extra_opts.get("debug_jax", False))
        self._debug_hlo_chars = int(self.extra_opts.get("debug_hlo_chars", 2500))
        t0 = time.perf_counter()
        self._setup_jax_rotate_ops()
        self._log_timing("__init__._setup_jax_rotate_ops", t0)
        t0 = time.perf_counter()
        self._prepare_beams_for_simulation(beams, combinations)
        self._block_ready(self.efbeams)
        self._log_timing("__init__._prepare_beams_for_simulation", t0)
        t0 = time.perf_counter()
        self._prepare_output_tensors()
        self._block_ready((self._output_beams, self._output_ground))
        self._log_timing("__init__._prepare_output_tensors", t0)
        t0 = time.perf_counter()
        self._setup_simulation_kernels()
        self._log_timing("__init__._setup_simulation_kernels", t0)
        if self._debug_enabled:
            self._debug_print(
                f"init summary: frame={self.sky_model.frame} lmax={self.lmax} "
                f"nfreq={len(self.freq)} nfreq_sky={len(self.freq_ndx_sky)} "
                f"nfreq_beam={len(self.freq_ndx_beam)} ncomb={len(self.combinations)}"
            )
            self._debug_array_summary("freq", self.freq)
            self._debug_array_summary("freq_ndx_sky", self.freq_ndx_sky)
            self._debug_array_summary("freq_ndx_beam", self.freq_ndx_beam)
            self._debug_array_summary("_output_beams", self._output_beams)
            self._debug_array_summary("_output_ground", self._output_ground)
        self._log_timing("__init__.total", t_init0)

    def _log_timing(self, label, t0):
        if self._timing_enabled:
            print(f"JaxSimulator timing {label}: {time.perf_counter() - t0:.3f} s", flush=True)

    def _debug_print(self, message):
        if self._debug_enabled:
            print(f"JaxSimulator debug: {message}", flush=True)

    def _debug_array_summary(self, label, value, max_items=6):
        if not self._debug_enabled:
            return
        try:
            arr = np.asarray(value)
        except Exception as exc:
            self._debug_print(f"{label}: unable to materialize summary ({exc!r})")
            return

        flat = arr.reshape(-1) if arr.shape else arr.reshape(1)
        sample = flat[:max_items].tolist()
        summary = f"{label}: shape={arr.shape} dtype={arr.dtype} size={arr.size}"
        if np.issubdtype(arr.dtype, np.number) or np.issubdtype(arr.dtype, np.complexfloating):
            sample_arr = flat[: min(max_items, flat.size)]
            finite_count = int(np.isfinite(sample_arr).sum()) if sample_arr.size else 0
            summary += f" finite_in_sample={finite_count}/{sample_arr.size}"
        summary += f" sample={sample}"
        self._debug_print(summary)

    def _debug_lowered_text(self, label, fn, *args):
        if not self._debug_enabled:
            return
        try:
            lowered = fn.lower(*args)
            text = lowered.as_text() if hasattr(lowered, "as_text") else None
            self._debug_print(f"{label}: lowered_type={type(lowered).__name__}")
            if text is not None:
                snippet = text[: self._debug_hlo_chars]
                self._debug_print(
                    f"{label}: lowered_text_first_{len(snippet)}_chars=\n{snippet}"
                )
        except Exception as exc:
            self._debug_print(f"{label}: lowering failed ({exc!r})")

    def _block_ready(self, value):
        if self._timing_enabled:
            jax.block_until_ready(value)
        return value

    def _time_batch_size(self, ntimes):
        batch_size = self.extra_opts.get("time_batch_size")
        if batch_size is None:
            return self._auto_time_batch_size(ntimes)
        batch_size = int(batch_size)
        if batch_size < 1:
            raise ValueError("extra_opts['time_batch_size'] must be >= 1 or None")
        return min(batch_size, ntimes)

    def _auto_time_batch_size(self, ntimes):
        return min(300, ntimes)

    def _prepare_beams_for_simulation(self, beams, combinations):
        if all(getattr(beam, "is_jax_pytree_beam", False) for beam in beams):
            self._prepare_beams_jax(beams, combinations)
            return
        SimulatorBase.prepare_beams(self, beams, combinations)
        self._convert_efbeams_to_jax()

    def _prepare_beams_jax(self, beams, combinations):
        self.beams = beams
        self.efbeams = []
        self.combinations = [(int(i), int(j)) for i, j in combinations]
        beam_idx = jnp.asarray(np.array(self.freq_ndx_beam, dtype=np.int32))

        for i, j in self.combinations:
            bi, bj = beams[i], beams[j]
            print(f"  intializing beam combination {bi.id} x {bj.id} ...")
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

            if i == j:
                ground_power_real = 1.0 - jnp.real(beamreal[:, 0]) / jnp.sqrt(4.0 * jnp.pi)
                ground_power_imag = jnp.zeros_like(ground_power_real)
                beamimag = None
            else:
                cross_power = jnp.asarray(self.cross_power.Ex_coupling(bi, bj, self.freq_ndx_beam))
                print(f"    cross power is {cross_power[0]} ... {cross_power[-1]} ")
                ground_power_real = cross_power - jnp.real(beamreal[:, 0]) / jnp.sqrt(4.0 * jnp.pi)
                ground_power_imag = -jnp.real(beamimag[:, 0]) / jnp.sqrt(4.0 * jnp.pi)

            if "dump_beams" in self.extra_opts:
                np.save(bi.id + bj.id, np.asarray(beamreal))

            self.efbeams.append(
                (i, j, beamreal, beamimag, ground_power_real, ground_power_imag)
            )

    def _setup_jax_rotate_ops(self):
        self._rotate_sky_packed_batch_jax = None
        L = self.lmax + 1
        M = 2 * L - 1
        alm_size = hp.sphtfunc.Alm.getsize(self.lmax)
        m_offset = L - 1
        self._debug_print(
            f"rotate op setup: L={L} M={M} alm_size={alm_size} m_offset={m_offset}"
        )

        packed_idx = []
        ell_idx = []
        m_idx = []
        packed_nonzero_idx = []
        ell_nonzero_idx = []
        m_nonzero_idx = []
        neg_sign = []
        for ell in range(L):
            for m in range(ell + 1):
                pidx = hp.sphtfunc.Alm.getidx(self.lmax, ell, m)
                packed_idx.append(pidx)
                ell_idx.append(ell)
                m_idx.append(m)
                if m > 0:
                    packed_nonzero_idx.append(pidx)
                    ell_nonzero_idx.append(ell)
                    m_nonzero_idx.append(m)
                    neg_sign.append(1.0 if (m % 2 == 0) else -1.0)

        packed_idx = jnp.asarray(np.array(packed_idx, dtype=np.int32))
        ell_idx = jnp.asarray(np.array(ell_idx, dtype=np.int32))
        m_idx = jnp.asarray(np.array(m_idx, dtype=np.int32))
        if packed_nonzero_idx:
            packed_nonzero_idx = jnp.asarray(np.array(packed_nonzero_idx, dtype=np.int32))
            ell_nonzero_idx = jnp.asarray(np.array(ell_nonzero_idx, dtype=np.int32))
            m_nonzero_idx = jnp.asarray(np.array(m_nonzero_idx, dtype=np.int32))
            neg_sign = jnp.asarray(np.array(neg_sign))
        else:
            packed_nonzero_idx = None
            ell_nonzero_idx = None
            m_nonzero_idx = None
            neg_sign = None

        m_all = jnp.arange(-(L - 1), L)

        def hp_to_full_flm_batch(packed_batch):
            flm = jnp.zeros((packed_batch.shape[0], L, M), dtype=packed_batch.dtype)
            flm = flm.at[:, ell_idx, m_offset + m_idx].set(packed_batch[:, packed_idx])
            if packed_nonzero_idx is not None:
                vals_neg = neg_sign[None, :] * jnp.conj(packed_batch[:, packed_nonzero_idx])
                flm = flm.at[:, ell_nonzero_idx, m_offset - m_nonzero_idx].set(vals_neg)
            return flm

        def full_flm_to_hp_batch(flm_batch):
            packed = jnp.zeros((flm_batch.shape[0], alm_size), dtype=flm_batch.dtype)
            packed = packed.at[:, packed_idx].set(flm_batch[:, ell_idx, m_offset + m_idx])
            return packed

        def rotate_sky_flm_batch(sky_flm_batch, alpha, beta, gamma):
            real_dtype = jnp.asarray(beta).dtype
            flm_rot = jnp.zeros_like(sky_flm_batch)
            dl_iter = jnp.zeros((M, M), dtype=real_dtype)
            exp_alpha = jnp.exp(-1j * m_all * alpha)
            exp_gamma = jnp.exp(-1j * m_all * gamma)
            for el in range(L):
                dl_iter = compute_full(dl_iter, beta, L, el)
                m = jnp.arange(-el, el + 1)
                midx = m_offset + m
                sky_block = sky_flm_batch[:, el, midx] * exp_gamma[midx][None, :]
                dl_block = dl_iter[midx[:, None], midx[None, :]]
                rot_block = jnp.einsum("mn,fn->fm", dl_block, sky_block, optimize=True)
                rot_block = rot_block * exp_alpha[midx][None, :]
                flm_rot = flm_rot.at[:, el, midx].set(rot_block)
            return flm_rot

        self._hp_to_full_flm_batch_jax = jax.jit(hp_to_full_flm_batch)
        self._full_flm_to_hp_batch_jax = jax.jit(full_flm_to_hp_batch)
        self._rotate_sky_flm_batch_jax = jax.jit(rotate_sky_flm_batch)

    def _convert_efbeams_to_jax(self):
        """Keep JaxSimulator internals jax-native while preserving external IO boundaries."""
        efbeams_jax = []
        for ci, cj, beamreal, beamimag, groundPowerReal, groundPowerImag in self.efbeams:
            efbeams_jax.append(
                (
                    ci,
                    cj,
                    jnp.asarray(beamreal),
                    None if beamimag is None else jnp.asarray(beamimag),
                    jnp.asarray(groundPowerReal),
                    jnp.asarray(groundPowerImag),
                )
            )
        self.efbeams = efbeams_jax

    def _prepare_output_tensors(self):
        """Flatten (combo, real/imag) outputs into a single axis for vectorized contraction."""
        output_beams = []
        output_ground = []
        for ci, cj, beamreal, beamimag, groundPowerReal, groundPowerImag in self.efbeams:
            output_beams.append(beamreal)
            output_ground.append(groundPowerReal)
            if ci != cj:
                output_beams.append(beamimag)
                output_ground.append(groundPowerImag)
        self._output_beams = jnp.asarray(jnp.stack(output_beams, axis=0))
        self._output_ground = jnp.asarray(jnp.stack(output_ground, axis=0))
        noutput, nfreq, _ = self._output_beams.shape
        beams_flat = self._output_beams.reshape(noutput * nfreq, self._output_beams.shape[-1])
        self._output_beams_flm = self._hp_to_full_flm_batch_jax(beams_flat).reshape(
            noutput, nfreq, self.lmax + 1, 2 * self.lmax + 1
        )

    def _setup_simulation_kernels(self):
        Tground = float(self.Tground)

        @jax.jit
        def contract_single_sky(sky_time_flm, output_beams_flm, output_ground):
            contracted = jnp.real(
                jnp.einsum(
                    "bflm,flm->bf",
                    output_beams_flm,
                    jnp.conj(sky_time_flm),
                    optimize=True,
                )
            ) / (4 * jnp.pi)
            return contracted + Tground * output_ground

        @jax.jit
        def rotate_and_contract_single_time(
            sky_base_flm, alpha, beta, gamma, output_beams_flm, output_ground
        ):
            rotated_sky = self._rotate_sky_flm_batch_jax(sky_base_flm, alpha, beta, gamma)
            return contract_single_sky(rotated_sky, output_beams_flm, output_ground)

        @jax.jit
        def rotate_and_contract_time_batch(
            sky_base_flm, alpha_batch, beta_batch, gamma_batch, output_beams_flm, output_ground
        ):
            return jax.vmap(
                lambda a, b, g: rotate_and_contract_single_time(
                    sky_base_flm, a, b, g, output_beams_flm, output_ground
                )
            )(alpha_batch, beta_batch, gamma_batch)

        self._contract_single_sky_jax = contract_single_sky
        self._rotate_and_contract_single_time_jax = rotate_and_contract_single_time
        self._rotate_and_contract_time_batch_jax = rotate_and_contract_time_batch

    def _compute_zyz_angles(self, lzl, bzl, lyl, byl, Nt):
        """Compute ZYZ Euler angles that match healpy XYZ rotation convention exactly."""
        alpha = np.empty(Nt, dtype=float)
        beta = np.empty(Nt, dtype=float)
        gamma = np.empty(Nt, dtype=float)
        for ti in range(Nt):
            lz, bz, ly, by = lzl[ti], bzl[ti], lyl[ti], byl[ti]
            zhat = np.array([np.cos(bz) * np.cos(lz), np.cos(bz) * np.sin(lz), np.sin(bz)])
            yhat = np.array([np.cos(by) * np.cos(ly), np.cos(by) * np.sin(ly), np.sin(by)])
            xhat = np.cross(yhat, zhat)
            assert (np.abs(np.dot(zhat, yhat)) < 1e-10)
            R = np.array([xhat, yhat, zhat]).T
            a, b, g = rot2eul(R)
            rot = hp.rotator.Rotator(
                rot=(float(g), float(-b), float(a)),
                deg=False,
                eulertype='XYZ',
                inv=False,
            )
            a_zyz, b_zyz, g_zyz = ScipyRotation.from_matrix(np.asarray(rot.mat)).as_euler("ZYZ")
            alpha[ti] = a_zyz
            beta[ti] = b_zyz
            gamma[ti] = g_zyz
        return alpha, beta, gamma

    def simulate_at_single_time(self, sky_base, alpha=None, beta=None, gamma=None):
        if alpha is None:
            return self._contract_single_sky_jax(
                sky_base,
                self._output_beams_flm,
                self._output_ground,
            )
        return self._rotate_and_contract_single_time_jax(
            sky_base,
            alpha,
            beta,
            gamma,
            self._output_beams_flm,
            self._output_ground,
        )

    def simulate (self,times=None):
        """
        Main simulation function. Produces mock observation of sky model from self.sky_model with beams from self.beams

        :param times: List of times, defaults to lusee.observation.times if empty
        :type times: list

        :returns: Waterfall style observation data for input times and self.freq
        :rtype: numpy array
        """
        if times is None:
            times = self.obs.times
        t_sim0 = time.perf_counter()
        if self._debug_enabled:
            self._debug_print(
                f"simulate start: ntimes={len(times)} frame={self.sky_model.frame} "
                f"time_batch_size_opt={self.extra_opts.get('time_batch_size')}"
            )
            if len(times) > 0:
                self._debug_print(f"simulate times: first={times[0]} last={times[-1]}")
        if self.sky_model.frame=="galactic":
            do_rot = True
            t0 = time.perf_counter()
            cache_fn = self._transform_cache_file(times)
            force_recompute = self._transform_cache_force_recompute()
            if force_recompute and (cache_fn is not None):
                print (f"Ignoring cached transform and recomputing: {cache_fn}")
            if (cache_fn is not None) and (not force_recompute) and (os.path.isfile(cache_fn)):
                print (f"Loading cached transform from {cache_fn}...")
                lzl,bzl,lyl,byl = pickle.load(open(cache_fn,'br'))
                if (len(lzl)!=len(times)):
                    raise RuntimeError("Cache file mix-up. Array wrong length!")
                have_transform = True
            else:
                have_transform = False

            if not have_transform:
                print ("Getting pole transformations...")
                lzl,bzl = self.obs.get_l_b_from_alt_az(np.pi/2,0., times)
                print ("Getting horizon transformations...")
                lyl,byl = self.obs.get_l_b_from_alt_az(0.,0., times)
                if cache_fn is not None:
                    print (f"Saving transforms to {cache_fn}...")
                    pickle.dump((lzl,bzl,lyl,byl),open(cache_fn,'bw'))
            self._log_timing("simulate.transforms", t0)

        elif self.sky_model.frame=="MCMF":
            do_rot = False
        else:
            raise NotImplementedError

        Nt = len(times)
        t0 = time.perf_counter()
        sky_base = jnp.asarray(self.sky_model.get_alm(self.freq_ndx_sky, self.freq))
        sky_base_flm = self._hp_to_full_flm_batch_jax(sky_base)
        self._block_ready(sky_base_flm)
        self._log_timing("simulate.sky_model.get_alm", t0)
        self._debug_array_summary("sky_base", sky_base)
        self._debug_array_summary("sky_base_flm", sky_base_flm)

        if do_rot:
            t0 = time.perf_counter()
            alpha, beta, gamma = self._compute_zyz_angles(lzl, bzl, lyl, byl, Nt)
            alpha = jnp.asarray(alpha)
            beta = jnp.asarray(beta)
            gamma = jnp.asarray(gamma)
            self._block_ready((alpha, beta, gamma))
            self._log_timing("simulate.compute_zyz_angles", t0)
            self._debug_array_summary("alpha", alpha)
            self._debug_array_summary("beta", beta)
            self._debug_array_summary("gamma", gamma)
            t0 = time.perf_counter()
            time_batch_size = self._time_batch_size(Nt)
            self._debug_print(f"simulate time_batch_size={time_batch_size}")
            if self._debug_enabled and Nt > 0:
                self._debug_lowered_text(
                    "rotate_and_contract_single_time",
                    self._rotate_and_contract_single_time_jax,
                    sky_base_flm,
                    alpha[0],
                    beta[0],
                    gamma[0],
                    self._output_beams_flm,
                    self._output_ground,
                )
                probe_t0 = time.perf_counter()
                self._debug_print("probe 1/3: compile+run single rotated time")
                probe_single = self.simulate_at_single_time(
                    sky_base_flm, alpha[0], beta[0], gamma[0]
                )
                jax.block_until_ready(probe_single)
                self._debug_print(
                    f"probe 1/3 complete in {time.perf_counter() - probe_t0:.3f} s"
                )
                self._debug_array_summary("probe_single", probe_single)
                if Nt > 1:
                    probe_t0 = time.perf_counter()
                    nprobe = min(2, Nt)
                    self._debug_print(
                        f"probe 2/3: compile+run vmap over first {nprobe} rotated times"
                    )
                    probe_many = jax.vmap(
                        lambda a, b, g: self.simulate_at_single_time(sky_base_flm, a, b, g)
                    )(alpha[:nprobe], beta[:nprobe], gamma[:nprobe])
                    jax.block_until_ready(probe_many)
                    self._debug_print(
                        f"probe 2/3 complete in {time.perf_counter() - probe_t0:.3f} s"
                    )
                    self._debug_array_summary("probe_many", probe_many)
                self._debug_print(
                    f"probe 3/3: entering chunked vmap ntimes={Nt} batch_size={time_batch_size}"
                )
            if time_batch_size >= Nt:
                self.result = self._rotate_and_contract_time_batch_jax(
                    sky_base_flm,
                    alpha,
                    beta,
                    gamma,
                    self._output_beams_flm,
                    self._output_ground,
                )
            else:
                pad = (-Nt) % time_batch_size
                if pad:
                    alpha_run = jnp.pad(alpha, (0, pad), mode="edge")
                    beta_run = jnp.pad(beta, (0, pad), mode="edge")
                    gamma_run = jnp.pad(gamma, (0, pad), mode="edge")
                else:
                    alpha_run = alpha
                    beta_run = beta
                    gamma_run = gamma
                chunks = []
                for start in range(0, alpha_run.shape[0], time_batch_size):
                    chunks.append(
                        self._rotate_and_contract_time_batch_jax(
                            sky_base_flm,
                            alpha_run[start : start + time_batch_size],
                            beta_run[start : start + time_batch_size],
                            gamma_run[start : start + time_batch_size],
                            self._output_beams_flm,
                            self._output_ground,
                        )
                    )
                self.result = jnp.concatenate(chunks, axis=0)[:Nt]
            self._block_ready(self.result)
            self._log_timing("simulate.batch_rotate_and_contract", t0)
            self._debug_array_summary("result_after_batch_vmap", self.result)
            sky_t0 = None
        else:
            dummy = jnp.arange(Nt)
            t0 = time.perf_counter()
            self.result = jax.vmap(
                lambda _: self.simulate_at_single_time(sky_base_flm)
            )(dummy)
            self._block_ready(self.result)
            self._log_timing("simulate.vmap_contract_no_rotation", t0)
            self._debug_array_summary("result_after_vmap", self.result)
            sky_t0 = sky_base_flm

        if self.extra_opts.get("plot_sky_and_beam"):
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
            beamreal0 = self.efbeams[0][2]
            if do_rot:
                t0 = time.perf_counter()
                sky_t0 = self._rotate_sky_flm_batch_jax(
                    sky_base_flm, alpha[0], beta[0], gamma[0]
                )
                self._block_ready(sky_t0)
                self._log_timing("simulate.rotate_sky_t0", t0)
                self._debug_array_summary("sky_t0", sky_t0)
            else:
                sky_t0 = sky_base_flm
            sky_t0_packed = self._full_flm_to_hp_batch_jax(sky_t0)
            self._plot_sky_beam_healpix(
                sky_t0_packed[freq_idx_plot], beamreal0[freq_idx_plot], nside, self.lmax,
                save_dir=self.extra_opts.get("plot_dir", default_plot_sky_beam_dir()),
                save_filename=self.extra_opts.get("plot_filename", "sky_beam_healpix_jaxsim.png"),
                title_prefix=f"JaxSimulator at {self.freq[freq_idx_plot]} MHz ",
            )

        self._log_timing("simulate.total", t_sim0)
        return self.result
