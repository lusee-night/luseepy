import pytest
import os


@pytest.fixture(scope="session")
def drive_dir():
    d = os.environ.get("LUSEE_DRIVE_DIR")
    if not d:
        pytest.skip("LUSEE_DRIVE_DIR not set")
    return d


LEGACY_BEAM_RELPATH = (
    "Simulations/OldBeamModels/LanderRegolithComparison/"
    "eight_layer_regolith/hfss_lbl_3m_75deg.fits"
)


@pytest.fixture(scope="session")
def legacy_beam_path(drive_dir):
    """Path to the deprecated scalar reference beam, or skip.

    The legacy scalar beams are deprecated and absent from the four-port
    CI Drive snapshot; tests that still exercise the v1/v2 reader against
    real data skip cleanly when only the new artifacts are present.
    """
    path = os.path.join(drive_dir, LEGACY_BEAM_RELPATH)
    if not os.path.isfile(path):
        pytest.skip("legacy scalar beam not present in this Drive checkout")
    return path
