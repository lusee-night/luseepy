"""Point-source covariance through the four-port pair-response kernel."""

import jax.numpy as jnp
import numpy as np
from scipy.constants import Boltzmann, c, physical_constants

from .Covariance import (
    assemble_open_covariance,
    load_covariance,
    pack_covariance,
)
from .InstrumentResponse import InstrumentResponse
from .FullStokesSimulator import default_target_frequencies


VACUUM_IMPEDANCE_OHM = physical_constants[
    "characteristic impedance of vacuum"
][0]


class FullStokesCalibratorSimulator:
    """Evaluate incoherent IQUV point-source covariance at arbitrary channels."""

    def __init__(
        self,
        beam,
        receiver,
        freq=None,
        products="all",
    ):
        if not isinstance(beam, InstrumentResponse):
            raise TypeError("beam must be an InstrumentResponse.")
        self.beam = beam
        self.receiver = receiver
        target = (
            default_target_frequencies(beam, None, receiver)[0]
            if freq is None
            else freq
        )
        self.freq = np.asarray(target, dtype=np.float64)
        self.products = products
        self.result = None
        self.covariance = None
        self.product_labels = None

    def simulate(
        self,
        theta_rad,
        phi_rad,
        stokes_K_sr,
        *,
        T_moon=0.0,
    ):
        """Simulate one source direction using the shared interpolated W kernel."""
        pair_kernel = self.beam.pair_stokes_at(
            theta_rad,
            phi_rad,
            self.freq,
        )
        stokes = jnp.asarray(stokes_K_sr)
        if stokes.shape == (4,):
            stokes = jnp.broadcast_to(stokes[None], (self.freq.size, 4))
        if stokes.shape != (self.freq.size, 4):
            raise ValueError(
                "stokes_K_sr must have shape (4,) or (frequency, 4)."
            )
        pair_integrals = jnp.einsum(
            "pfs,fs->fp",
            pair_kernel,
            stokes,
        )[None]
        ZA, _, Rmoon, _ = self.beam.target_matrices(self.freq)
        open_covariance = assemble_open_covariance(
            pair_integrals,
            Rmoon,
            T_moon,
        )
        ZL = self.receiver.Z(self.freq)
        covariance, M = load_covariance(open_covariance, ZA, ZL)
        packed, labels = pack_covariance(covariance, self.products)
        self.result = packed
        self.covariance = covariance
        self.product_labels = labels
        self.M = M
        return packed

    def direct_native_covariance(
        self,
        frequency_index,
        theta_rad,
        phi_rad,
        stokes_K_sr,
    ):
        """Direct loaded-H reference at one native frequency."""
        index = int(frequency_index)
        frequency = np.asarray([self.beam.freq[index]], dtype=np.float64)
        ZL = self.receiver.Z(frequency)
        loaded = self.beam.loaded_response_at(
            ZL,
            theta_rad,
            phi_rad,
            frequency,
        )[0]
        stokes_i, stokes_q, stokes_u, stokes_v = jnp.asarray(stokes_K_sr)
        wavelength = c / (frequency[0] * 1e6)
        electric_covariance = (
            Boltzmann
            * VACUUM_IMPEDANCE_OHM
            / wavelength**2
            * jnp.asarray(
                [
                    [
                        stokes_i + stokes_q,
                        stokes_u - 1j * stokes_v,
                    ],
                    [
                        stokes_u + 1j * stokes_v,
                        stokes_i - stokes_q,
                    ],
                ]
            )
        )
        return jnp.einsum(
            "ap,pq,bq->ab",
            loaded,
            electric_covariance,
            loaded.conjugate(),
        )
