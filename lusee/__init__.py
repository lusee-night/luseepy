"""A package for simulation of LuSEE-Night."""

from .Observation   import Observation
from .Data          import Data
from .Satellite     import Satellite, ObservedSatellite
from .Beam          import Beam, grid2healpix, grid2healpix_alm_fast
from .BeamInterpolator import BeamInterpolator
from .BeamGauss     import BeamGauss
from .BeamCouplings import BeamCouplings
from .DefaultSimulator import DefaultSimulator
try:
    from .CroSimulator import CroSimulator
except (ModuleNotFoundError, ImportError) as e:
    if "croissant" in str(e).lower() or "s2fft" in str(e).lower():
        CroSimulator = None  # optional: install croissant (and s2fft) to use CroSimulator
    else:
        raise

from .SkyModels     import FitsSky
from .SkyModels     import GalCenter

from . import SkyModels     as sky


from . import MonoSkyModels as monosky 
from .PCAanalyzer import PCAanalyzer, CompositePCAanalyzer
from .Throughput import Throughput

__version__ = '1.3'
__comment__ = '1.3 dev'


