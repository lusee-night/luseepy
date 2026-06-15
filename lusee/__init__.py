"""A package for simulation of LuSEE-Night."""

from .Observation   import Observation
from .CalibratorTrack import CalibratorTrack
from .Data          import Data
from .Satellite     import Satellite, ObservedSatellite
from .Beam          import Beam, grid2healpix, grid2healpix_alm_fast
from .BeamInterpolator import BeamInterpolator
from .BeamGauss     import BeamGauss
from .NpWrapper     import NpWrapper
from .LabeledArray  import (
    LabeledArray, label, relabel, asarray, units_of, frame_of, is_labeled,
    describe,
    FRAME_GALACTIC, FRAME_EQUATORIAL, FRAME_MCMF, FRAME_TOPO, FRAME_MEPA,
)
from .CachedBeam   import CachedBeam
from .BeamCouplings import BeamCouplings
from .DefaultSimulator import TopoNumpySimulator
from .TopoJaxSimulator import TopoJaxSimulator
from .CalibratorSimulator import CalibratorSimulator
from .NumpySimulator import NumpySimulator
try:
    from .CroSimulator import CroSimulator
except (ModuleNotFoundError, ImportError) as e:
    if (
        "croissant" in str(e).lower()
        or "s2fft" in str(e).lower()
        or "spiceypy" in str(e).lower()
    ):
        CroSimulator = None  # optional: install croissant/s2fft/spiceypy to use CroSimulator
    else:
        raise

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
    FrequencyMap,
    canonical_frequencies,
    canonical_frequency_indices,
    canonicalize_frequencies,
    frequencies_from_config,
    frequency_indices_from_values,
)

__version__ = '1.3'
__comment__ = '1.3 dev'
