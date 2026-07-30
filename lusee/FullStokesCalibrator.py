"""Point-source covariance through the four-port pair-response kernel."""

import jax.numpy as jnp
import numpy as np
from scipy.constants import Boltzmann, c, physical_constants

from .Covariance import (
    apply_receiver_loading,
    assemble_open_covariance,
    covariance_projection_diagnostics,
    matrix_condition_number,
    pack_covariance,
    project_hermitian,
)
from .InstrumentResponse import InstrumentResponse
from .FullStokesSimulator import default_target_frequencies
from .ResponsePhysics import validate_response_matrices


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
        T_ant=None,
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
        ZA, Rsky, Rmoon, Rloss, _ = self.beam.target_matrices(self.freq)
        validate_response_matrices(
            np.asarray(ZA),
            np.asarray(Rsky),
            np.asarray(Rmoon),
            np.asarray(Rloss),
            field_rsky=np.asarray(Rsky),
            loss_model=self.beam.header.get("LOSSMODEL"),
        )
        if T_ant is None:
            loss_model = str(
                self.beam.header.get("LOSSMODEL", "")
            ).strip().lower()
            if loss_model == "pec" or np.all(
                np.asarray(self.beam.Rloss_native) == 0
            ):
                T_ant = 0.0
            else:
                raise ValueError(
                    "A lossy four-port response requires an explicit T_ant."
                )
        open_covariance = assemble_open_covariance(
            pair_integrals,
            Rmoon,
            Rloss,
            T_moon=T_moon,
            T_ant=T_ant,
        )
        ZL = self.receiver.Z(self.freq)
        unprojected_covariance, M = apply_receiver_loading(
            open_covariance,
            ZA,
            ZL,
        )
        covariance = project_hermitian(unprojected_covariance)
        diagnostics = covariance_projection_diagnostics(
            unprojected_covariance
        )
        packed, labels = pack_covariance(covariance, self.products)
        self.result = packed
        self.covariance = covariance
        self.product_labels = labels
        self.M = M
        self.covariance_antihermitian_absolute = diagnostics[
            "antihermitian_absolute"
        ]
        self.covariance_antihermitian_relative = diagnostics[
            "antihermitian_relative"
        ]
        self.covariance_eigenvalues = diagnostics["eigenvalues"]
        self.covariance_minimum_eigenvalue_ratio = diagnostics[
            "minimum_eigenvalue_ratio"
        ]
        self.loading_condition_number = matrix_condition_number(ZA + ZL)
        self.Rmoon = Rmoon
        self.Rloss = Rloss
        self.T_moon = T_moon
        self.T_ant = T_ant
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
