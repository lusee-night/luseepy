import pytest
import os


@pytest.fixture(scope="session")
def drive_dir():
    d = os.environ.get("LUSEE_DRIVE_DIR")
    if not d:
        pytest.skip("LUSEE_DRIVE_DIR not set")
    return d
