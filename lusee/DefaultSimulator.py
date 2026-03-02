from .Observation import Observation
from .Beam import Beam
from .BeamCouplings import BeamCouplings
from .SimulatorBase import SimulatorBase, rot2eul
import numpy as np
import jax
import jax.numpy as jnp
import healpy as hp
import fitsio
import sys
import pickle
import os
from scipy.spatial.transform import Rotation as ScipyRotation



class DefaultSimulator(SimulatorBase):
    """
    Default Simulator in luseepy 
    
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
    :param extra_opts: Extra options for simulation. Supports "dump_beams" (saves instrument beams to file)
        and "cache_transform" (loads/saves beam transformations from file).
    :type extra_opts: dict
    
    """

    def __init__ (self, obs, beams, sky_model, Tground = 200.0,
                  combinations = [(0,0),(1,1),(0,2),(1,3),(1,2)], freq = None,
                  lmax = 128, cross_power = None,
                  extra_opts = {}):
        super().__init__(obs, beams, sky_model, Tground, combinations, freq)
        self.lmax = lmax
        self.extra_opts = extra_opts
        self.cross_power = cross_power if (cross_power is not None) else BeamCouplings()
        self._setup_jax_rotate_ops()
        self.prepare_beams (beams, combinations)
        self._convert_efbeams_to_jax()
        self._prepare_output_tensors()
        self._setup_simulation_kernels()

    def _setup_jax_rotate_ops(self):
        self._rotate_sky_packed_batch_jax = None
        from s2fft.utils.rotation import compute_full

        L = self.lmax + 1
        M = 2 * L - 1
        alm_size = hp.sphtfunc.Alm.getsize(self.lmax)
        m_offset = L - 1

        packed_idx = []
        ell_idx = []
        m_idx = []
        packed_pos_idx = []
        ell_pos_idx = []
        m_pos_idx = []
        neg_sign = []
        for ell in range(L):
            for m in range(ell + 1):
                pidx = hp.sphtfunc.Alm.getidx(self.lmax, ell, m)
                packed_idx.append(pidx)
                ell_idx.append(ell)
                m_idx.append(m)
                if m > 0:
                    packed_pos_idx.append(pidx)
                    ell_pos_idx.append(ell)
                    m_pos_idx.append(m)
                    neg_sign.append(1.0 if (m % 2 == 0) else -1.0)

        packed_idx = jnp.asarray(np.array(packed_idx, dtype=np.int32))
        ell_idx = jnp.asarray(np.array(ell_idx, dtype=np.int32))
        m_idx = jnp.asarray(np.array(m_idx, dtype=np.int32))
        if packed_pos_idx:
            packed_pos_idx = jnp.asarray(np.array(packed_pos_idx, dtype=np.int32))
            ell_pos_idx = jnp.asarray(np.array(ell_pos_idx, dtype=np.int32))
            m_pos_idx = jnp.asarray(np.array(m_pos_idx, dtype=np.int32))
            neg_sign = jnp.asarray(np.array(neg_sign))
        else:
            packed_pos_idx = None
            ell_pos_idx = None
            m_pos_idx = None
            neg_sign = None

        valid_m = np.zeros((L, M))
        for ell in range(L):
            valid_m[ell, m_offset - ell : m_offset + ell + 1] = 1.0
        valid_m = jnp.asarray(valid_m)
        m_all = jnp.arange(-(L - 1), L)

        def rotate_sky_packed_batch(sky_packed_batch, alpha, beta, gamma):
            real_dtype = jnp.asarray(beta).dtype
            dls = jnp.zeros((L, M, M), dtype=real_dtype)
            dl_iter = jnp.zeros((M, M), dtype=real_dtype)
            for el in range(L):
                dl_iter = compute_full(dl_iter, beta, L, el)
                dls = dls.at[el].set(dl_iter)

            flm = jnp.zeros((sky_packed_batch.shape[0], L, M), dtype=sky_packed_batch.dtype)
            vals = sky_packed_batch[:, packed_idx]
            flm = flm.at[:, ell_idx, m_offset + m_idx].set(vals)
            if packed_pos_idx is not None:
                vals_neg = neg_sign[None, :] * jnp.conj(sky_packed_batch[:, packed_pos_idx])
                flm = flm.at[:, ell_pos_idx, m_offset - m_pos_idx].set(vals_neg)

            exp_alpha = jnp.exp(-1j * m_all * alpha)
            exp_gamma = jnp.exp(-1j * m_all * gamma)
            valid = valid_m[None, :, :]
            flm_weighted = flm * exp_gamma[None, None, :] * valid
            flm_rot = jnp.einsum("lmn,fln->flm", dls, flm_weighted)
            flm_rot = flm_rot * exp_alpha[None, None, :] * valid

            packed_rot = jnp.zeros((sky_packed_batch.shape[0], alm_size), dtype=sky_packed_batch.dtype)
            packed_rot = packed_rot.at[:, packed_idx].set(flm_rot[:, ell_idx, m_offset + m_idx])
            return packed_rot

        self._rotate_sky_packed_batch_jax = jax.jit(rotate_sky_packed_batch)
        self._rotate_many_times_jax = jax.jit(
            jax.vmap(self._rotate_sky_packed_batch_jax, in_axes=(None, 0, 0, 0))
        )

    def _convert_efbeams_to_jax(self):
        """Keep DefaultSimulator internals jax-native while preserving external IO boundaries."""
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

    def _setup_simulation_kernels(self):
        lmax = self.lmax
        Tground = float(self.Tground)

        @jax.jit
        def simulate_from_rotated_sky(sky_time, output_beams, output_ground):
            # sky_time: [Nt, Nfreq, Nalm], output_beams: [Nout, Nfreq, Nalm]
            prod = output_beams[None, :, :, :] * jnp.conj(sky_time[:, None, :, :])
            contracted = (
                jnp.real(prod[..., : lmax + 1]).sum(axis=-1)
                + 2 * jnp.real(prod[..., lmax + 1 :]).sum(axis=-1)
            ) / (4 * jnp.pi)
            return contracted + Tground * output_ground[None, :, :]

        self._simulate_from_rotated_sky_jax = simulate_from_rotated_sky

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
        if self.sky_model.frame=="galactic":
            do_rot = True
            cache_fn = self.extra_opts.get("cache_transform")
            if (cache_fn is not None) and (os.path.isfile(cache_fn)):
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
                lyl,byl = self.obs.get_l_b_from_alt_az(0.,0., times)  ## astronomical azimuth = 0 = N = our y coordinate
                if cache_fn is not None:
                    print (f"Saving transforms to {cache_fn}...")
                    pickle.dump((lzl,bzl,lyl,byl),open(cache_fn,'bw'))
            
        elif self.sky_model.frame=="MCMF":
            do_rot = False
        else:
            raise NotImplementedError

        Nt = len (times)
        sky_base = jnp.asarray(self.sky_model.get_alm(self.freq_ndx_sky, self.freq))
        if do_rot:
            alpha, beta, gamma = self._compute_zyz_angles(lzl, bzl, lyl, byl, Nt)
            sky_time = self._rotate_many_times_jax(
                sky_base,
                jnp.asarray(alpha),
                jnp.asarray(beta),
                jnp.asarray(gamma),
            )
        else:
            sky_time = jnp.broadcast_to(sky_base[None, :, :], (Nt,) + sky_base.shape)

        self.result = self._simulate_from_rotated_sky_jax(
            sky_time,
            self._output_beams,
            self._output_ground,
        )
        return self.result
            
