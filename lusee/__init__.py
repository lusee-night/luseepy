"""A package for simulation of LuSEE-Night."""

from .Observation   import Observation
from .CalibratorTrack import CalibratorTrack
from .Data          import Data
from .Satellite     import Satellite, ObservedSatellite
from .Beam          import Beam, grid2healpix, grid2healpix_alm_fast
from .BeamInterpolator import BeamInterpolator
from .BeamGauss     import BeamGauss
from .beam_fine_frequency import linear_resample_beam_freq_mhz
from .NpWrapper     import NpWrapper
from .CachedBeam   import CachedBeam
from .BeamCouplings import BeamCouplings
from .DefaultSimulator import TopoNumpySimulator
from .TopoJaxSimulator import TopoJaxSimulator
from .CalibratorSimulator import CalibratorSimulator
from .NumpySimulator import NumpySimulator
try:
    from .CroSimulator import CroSimulator
except (ModuleNotFoundError, ImportError) as e:
    if "croissant" in str(e).lower() or "s2fft" in str(e).lower():
        CroSimulator = None  # optional: install croissant (and s2fft) to use CroSimulator
    else:
        raise

from .RRLSkyModels import (
    RRL_DEFAULT_LINE_FWHM_KHZ,
    RRL_DEFAULT_LINE_PEAK_K,
    RRL_DEFAULT_LINE_SIGMA_MHZ,
    ULSAPlusEnvelopeSky,
    ULSAPlusRRLSky,
    build_ulsa_rrl_sky,
    default_rrl_catalog_path,
    default_ulsa_path,
    carbon_rrl_alpha_transitions_in_frequency_band_mhz,
    carbon_rrl_frequency_mhz,
    carbon_rrl_transitions_in_frequency_band_mhz,
    hydrogen_rrl_alpha_quantum_numbers_from_frequency_mhz,
    hydrogen_rrl_alpha_transitions_in_frequency_band_mhz,
    hydrogen_rrl_frequency_mhz,
    load_rrl_region_positions_gal_deg,
    COLD_GAS_VYDULA2024,
    HOT_GAS_VYDULA2024,
    Vydula2024EnvelopeParams,
    rrl_envelope_T_rrl_k_mhz,
    rrl_smooth_envelope_weight_mhz,
    rydberg_line_spectrum_mhz,
    vydula2024_envelope_params_from_config,
    vydula2024_envelope_params_from_gas_case,
)
from .RRLAnalysis import (
    RRLAnalysisPipeline,
    RRLPipelineStages,
    build_rrl_analysis_pipeline,
    resample_waterfall_frequency,
)
from .RRLSim import RRLSimulator
from .SkyModels     import FitsSky
from .SkyModels     import GalCenter
from .SkyModels     import HarmonicPointSourceSky

from . import SkyModels     as sky
from . import MapMaker      as mapmaker


from . import MonoSkyModels as monosky 
from .PCAanalyzer import PCAanalyzer, CompositePCAanalyzer
from .Throughput import Throughput
from .frequencies import (
    ALL_FREQUENCIES_MHZ,
    ALL_FREQUENCY_INDICES,
    canonical_frequencies,
    canonical_frequency_indices,
    fine_uniform_frequency_mhz,
    frequency_indices_from_config,
    frequency_indices_from_values,
    observation_frequency_mhz_from_config,
)

__version__ = '1.3'
__comment__ = '1.3 dev'
