"""FLASH identity extraction uses the same schema proof as collection decoding."""

import json

import numpy as np
import pytest

from test_ccsds_framing import make_frame
from test_decode_spectra import metadata_blob, spectrum_blob
from test_pipeline_time_reference import write_landing_reference
from uncrater.schema_registry import SchemaConflictError, binding_for_key
from uncrater.decode_status import PacketDecodeError

from lusee.ingest.constants import BANK_FILENAME
from lusee.ingest.decode import read_uncrater_session
from lusee.ingest.pipeline import process_flash


def hello_blob(binding, uid):
    value = binding.pystruct.startup_hello()
    value.SW_version = binding.canonical_schema_id
    value.unique_packet_id = uid
    value.time_32 = uid * 65536
    return bytes(value)


def hk_blob(binding, kind, uid):
    value = getattr(binding.pystruct, f"housekeeping_data_{kind}")()
    value.base.version = binding.canonical_schema_id
    value.base.unique_packet_id = uid
    value.base.housekeeping_type = kind
    blob = bytes(value)
    return blob + b"\0" * (-len(blob) % 4)


def packets_for(binding, *, split=False):
    packets = [(0x209, hello_blob(binding, 1)),
               (0x20F, metadata_blob(binding, uid=2)),
               (0x210, spectrum_blob(2, np.full(512, 4, dtype="<u4")))]
    if split:
        packets.extend([(0x206, hk_blob(binding, 1, 3)),
                        (0x209, hello_blob(binding, 100))])
    packets.append((0x206, hk_blob(binding, 0, 103)))
    return packets


def save_flash(root, packets):
    bank = root / "b05" / BANK_FILENAME
    bank.parent.mkdir(parents=True)
    bank.write_bytes(b"".join(
        make_frame(np.frombuffer(blob, dtype="<u2").byteswap().tobytes(), appid=appid, sequence=index)
        for index, (appid, blob) in enumerate(packets)
    ))


@pytest.mark.parametrize("key", ["306-early", "306-final", "307"])
@pytest.mark.parametrize("explicit", [False, True])
def test_flash_retains_science_and_startup_with_input_schema(tmp_path, key, explicit):
    binding = binding_for_key(key)
    packets = packets_for(binding)
    direct = tmp_path / "direct"
    direct.mkdir()
    for index, (appid, blob) in enumerate(packets):
        (direct / f"{index:05d}_{appid:04x}.bin").write_bytes(blob)
    expected = read_uncrater_session(direct)
    assert len(expected.spectra) == 1
    flash = tmp_path / "FLASH"
    save_flash(flash, packets)
    options = {"schema_variant": binding.variant} if explicit else {}
    result = process_flash(flash, landing_time_file=write_landing_reference(tmp_path),
                           sessions_root=tmp_path / "sessions", h5_dir=tmp_path / "h5", **options)
    assert len(result) == 1
    directory, = (tmp_path / "sessions").glob("session_*")
    products = read_uncrater_session(directory)
    assert len(products.spectra) == 1
    assert products.decode_provenance.binding_key == key
    assert products.start_raw_seconds == 1.0
    np.testing.assert_array_equal(products.spectra[0].data, expected.spectra[0].data)
    document = json.loads((directory / "packet_map.json").read_text())
    assert document["schema_resolution"]["binding_key"] == key
    assert {item["original_appid"] for item in document["packets"]} >= {0x209, 0x20F}


@pytest.mark.parametrize("key", ["306-early", "306-final"])
def test_reopened_subset_keeps_evidence_from_other_session(tmp_path, key):
    binding = binding_for_key(key)
    flash = tmp_path / "FLASH"
    save_flash(flash, packets_for(binding, split=True))
    result = process_flash(flash, landing_time_file=write_landing_reference(tmp_path),
                           sessions_root=tmp_path / "sessions", h5_dir=tmp_path / "h5")
    assert len(result) == 2
    directory = sorted((tmp_path / "sessions").glob("session_*"))[0]
    products = read_uncrater_session(directory)
    assert len(products.spectra) == 1
    assert len(products.housekeeping) == 1
    assert products.decode_provenance.binding_key == key
    report = json.loads(products.decode_provenance.canonical_report_json)
    assert report["input_schema"]["evidence"]
    assert report["input_schema"]["binding_key"] == key
    with pytest.raises(PacketDecodeError, match="schema_conflict"):
        read_uncrater_session(directory, strict=True,
                              schema_variant="final" if key.endswith("early") else "early")
    document = json.loads((directory / "packet_map.json").read_text())
    document["schema_resolution"]["abi_fingerprint"] = "0" * 64
    (directory / "packet_map.json").write_text(json.dumps(document))
    with pytest.raises(SchemaConflictError, match="Stored input schema"):
        read_uncrater_session(directory)


@pytest.mark.parametrize("key", ["306-early", "306-final"])
def test_versionless_waveform_session_reopens_with_input_proof(tmp_path, key):
    from test_waveform_ingest import codes, logical, metadata
    from test_decode_auxiliary import waveform_blob
    from lusee.ingest.session import Session, write_uncrater_session
    from lusee.ingest.uncrater_adapter import resolve_input_schema

    binding = binding_for_key(key)
    proof = resolve_input_schema([logical(appid, blob) for appid, blob in packets_for(binding)])
    session = Session(0, [logical(0x2F0, waveform_blob(codes()), uid=6),
                          logical(0x2FA, metadata(7), uid=7)], schema_resolution=proof)
    directory = tmp_path / "session"
    write_uncrater_session(session, directory)
    products = read_uncrater_session(directory)
    assert len(products.waveforms) == 1
    assert products.waveforms[0].unique_packet_id == 7
    assert products.decode_provenance.reported_schema_ids == (0x306,)
    assert products.decode_provenance.binding_key == key
