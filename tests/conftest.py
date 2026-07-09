import pytest
import os


@pytest.fixture(scope="session")
def drive_dir():
    d = os.environ.get("LUSEE_DRIVE_DIR")
    if not d:
        pytest.skip("LUSEE_DRIVE_DIR not set")
    return d


@pytest.fixture(scope="session", autouse=True)
def furnish_spice_moon_frame():
    """Furnish the SPICE Moon PA/ME frame kernels into the global pool.

    croissant calls ``spice.pxform("MOON_ME", ...)`` against the global spiceypy
    kernel pool, but neither croissant nor lunarsky (which keeps its own internal
    pool) populates it. Without this, every CroSimulator test dies with
    SpiceUNKNOWNFRAME. The kernels ship with lunarsky. No-op if lunarsky/spiceypy
    are absent or MOON_ME already resolves.
    """
    try:
        import spiceypy as spice
        import lunarsky
    except Exception:
        return
    data = os.path.join(os.path.dirname(lunarsky.__file__), "data")
    for rel in ("pck/moon_pa_de421_1900-2050.bpc", "fk/satellites/moon_080317.tf"):
        path = os.path.join(data, rel)
        if os.path.isfile(path):
            spice.furnsh(path)
