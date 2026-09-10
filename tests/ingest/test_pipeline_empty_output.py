"""Empty decoded sessions retain a consistent failure audit when written."""

import json

import pytest

from test_pipeline_time_reference import write_landing_reference
from test_waveform_ingest import hello
from uncrater import id

from lusee.ingest.pipeline import process_session


@pytest.mark.parametrize("formats", [(), ("h5",), ("fits",), ("h5", "fits"), ("h5", "plots")])
def test_hello_only_session_writes_failed_outputs_without_internal_error(tmp_path, formats):
    session = tmp_path / "session"
    cdi = session / "cdi_output"
    cdi.mkdir(parents=True)
    (cdi / f"00000_{int(id.AppID_uC_Start):04x}.bin").write_bytes(hello(10))
    options = {f"{format}_dir": tmp_path / format for format in formats}
    result = process_session(session, name="empty", manifest_dir=tmp_path / "manifests",
                             landing_time_file=write_landing_reference(tmp_path), **options)
    assert result.status == "failed"
    assert result.status_issue_codes == ["decode.no_usable_products"]
    assert result.issue_counts["decode.no_usable_products"] == 1
    manifest = json.loads((tmp_path / "manifests" / "empty.json").read_text())
    assert manifest["status"] == "failed"
    assert result.h5_path is None and result.fits_path is None
    assert not (tmp_path / "plots").exists()
    for format in formats:
        assert not list((tmp_path / format).glob(f"*.{format}"))


def test_failed_overwrite_removes_only_requested_stale_products(tmp_path):
    from test_waveform_ingest import codes
    from test_decode_auxiliary import waveform_blob

    session = tmp_path / "session"
    cdi = session / "cdi_output"
    cdi.mkdir(parents=True)
    waveform = cdi / "00000_02f0.bin"
    waveform.write_bytes(waveform_blob(codes()))
    options = dict(name="capture", landing_time_file=write_landing_reference(tmp_path),
                   h5_dir=tmp_path / "h5", fits_dir=tmp_path / "fits")
    first = process_session(session, **options)
    h5 = tmp_path / "h5" / "capture.h5"
    fits = tmp_path / "fits" / "capture.fits"
    assert first.status == "partial" and h5.is_file() and fits.is_file()
    untouched = tmp_path / "h5" / "other.h5"
    untouched.write_bytes(b"another capture")
    plot_dir = tmp_path / "plots" / "capture"
    plot_dir.mkdir(parents=True)
    (plot_dir / "old.png").write_bytes(b"old plot")
    options["plots_dir"] = tmp_path / "plots"
    waveform.unlink()
    (cdi / f"00000_{int(id.AppID_uC_Start):04x}.bin").write_bytes(hello(10))
    with pytest.raises(FileExistsError):
        process_session(session, **options)
    assert h5.is_file() and fits.is_file() and plot_dir.is_dir()
    failed = process_session(session, overwrite=True, **options)
    assert failed.status == "failed"
    assert failed.h5_path is None and failed.fits_path is None
    assert not h5.exists() and not fits.exists() and not plot_dir.exists()
    assert untouched.read_bytes() == b"another capture"
