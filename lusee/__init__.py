"""A package for simulation of LuSEE-Night.

Top-level attribute access (e.g. ``lusee.Observation``, ``lusee.Beam``)
triggers a lazy import of the relevant submodule. This keeps light-weight
subpackages like :mod:`lusee.ingest` importable without dragging in the
heavy simulation dependency stack (astropy, lunarsky, jax, ...).
"""

from __future__ import annotations

__version__ = '1.3'
__comment__ = '1.3 dev'

# Map: attribute name -> (submodule, attribute-in-submodule or None for the module itself).
_LAZY = {
    # Observation / time
    'Observation':         ('Observation', 'Observation'),
    'CalibratorTrack':     ('CalibratorTrack', 'CalibratorTrack'),
    'Data':                ('Data', 'Data'),

    # Satellite
    'Satellite':           ('Satellite', 'Satellite'),
    'ObservedSatellite':   ('Satellite', 'ObservedSatellite'),

    # Beam family
    'Beam':                ('Beam', 'Beam'),
    'grid2healpix':        ('Beam', 'grid2healpix'),
    'grid2healpix_alm_fast': ('Beam', 'grid2healpix_alm_fast'),
    'BeamInterpolator':    ('BeamInterpolator', 'BeamInterpolator'),
    'BeamGauss':           ('BeamGauss', 'BeamGauss'),
    'NpWrapper':           ('NpWrapper', 'NpWrapper'),
    'CachedBeam':          ('CachedBeam', 'CachedBeam'),
    'BeamCouplings':       ('BeamCouplings', 'BeamCouplings'),

    # Simulators
    'TopoNumpySimulator':  ('DefaultSimulator', 'TopoNumpySimulator'),
    'TopoJaxSimulator':    ('TopoJaxSimulator', 'TopoJaxSimulator'),
    'CalibratorSimulator': ('CalibratorSimulator', 'CalibratorSimulator'),
    'NumpySimulator':      ('NumpySimulator', 'NumpySimulator'),
    'CroSimulator':        ('CroSimulator', 'CroSimulator'),

    # Sky models
    'FitsSky':                 ('SkyModels', 'FitsSky'),
    'GalCenter':               ('SkyModels', 'GalCenter'),
    'HarmonicPointSourceSky':  ('SkyModels', 'HarmonicPointSourceSky'),
    'sky':                     ('SkyModels', None),
    'mapmaker':                ('MapMaker', None),
    'monosky':                 ('MonoSkyModels', None),

    # Analysis / utilities
    'PCAanalyzer':           ('PCAanalyzer', 'PCAanalyzer'),
    'CompositePCAanalyzer':  ('PCAanalyzer', 'CompositePCAanalyzer'),
    'Throughput':            ('Throughput', 'Throughput'),

    # Frequencies module
    'ALL_FREQUENCIES_MHZ':            ('frequencies', 'ALL_FREQUENCIES_MHZ'),
    'ALL_FREQUENCY_INDICES':          ('frequencies', 'ALL_FREQUENCY_INDICES'),
    'canonical_frequencies':          ('frequencies', 'canonical_frequencies'),
    'canonical_frequency_indices':    ('frequencies', 'canonical_frequency_indices'),
    'frequency_indices_from_config':  ('frequencies', 'frequency_indices_from_config'),
    'frequency_indices_from_values':  ('frequencies', 'frequency_indices_from_values'),

    # Subpackages
    'ingest': ('ingest', None),
}


def __getattr__(name):
    """PEP 562 lazy attribute loader."""
    try:
        modname, attrname = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module 'lusee' has no attribute {name!r}") from None

    import importlib
    try:
        mod = importlib.import_module(f'.{modname}', __name__)
    except (ModuleNotFoundError, ImportError) as e:
        # CroSimulator is optional: install croissant (and s2fft) to use it.
        if name == 'CroSimulator' and (
            'croissant' in str(e).lower() or 's2fft' in str(e).lower()
        ):
            value = None
        # ingest is optional: install h5py to use it.
        elif name == 'ingest':
            value = None
        else:
            raise
    else:
        value = mod if attrname is None else getattr(mod, attrname)

    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals().keys()) | set(_LAZY.keys()))
