# -- FIXME --

# -mxp- commented out some imports since the hardcoded cache breaks
# execution on Singularity and otherwise has undesirable side
# effects. There is a separate cache in the LunarCalendar class
# now, which can be propagated to other modules. It behaves
# gracefully e.g. optionally cleaned up, has db name capability etc.

from .Observation   import Observation
from .Data          import Data
from .Satellite     import Satellite, ObservedSatellite
from .Beam          import Beam, grid2healpix, grid2healpix_alm_fast, grid2healpix_alm_reference, getLegendre, project_to_theta_phi
from .BeamGauss     import BeamGauss
from .BeamCouplings import BeamCouplings
from .Simulation    import Simulator

from .SkyModels     import FitsSky
from .SkyModels     import GalCenter

from . import SkyModels     as sky


from . import MonoSkyModels as monosky 
from .PCAanalyzer import PCAanalyzer, CompositePCAanalyzer
from .Throughput import Throughput

__version__ = '0.2'
__comment__ = 'The first official update after the improved name convention, and addition of Readthedocs functionality'




