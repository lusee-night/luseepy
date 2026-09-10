"""Waveform retention and metadata repair through extraction and both formats."""

import json
from dataclasses import replace

import h5py
import numpy as np
import pytest
from test_ccsds_framing import make_frame
from test_decode_auxiliary import BINDING, waveform_blob, waveform_metadata_blob, write_packet
from test_layout_v4_hdf5 import make_request
from test_pipeline_time_reference import write_landing_reference

from lusee.ingest.collation import assign_identities
from lusee.ingest.constants import BANK_FILENAME
from lusee.ingest.decode import read_uncrater_session
from lusee.ingest.fits_writer import write_fits
from lusee.ingest.hdf5_writer import write_hdf5
from lusee.ingest.obs_factory import load_bundle
from lusee.ingest.packet_map import PacketMapError
from lusee.ingest.pipeline import process_flash
from lusee.ingest.reassembly import LogicalPacket
from lusee.ingest.session import Session, associate_session_waveforms, split_sessions, write_uncrater_session
from lusee.ingest.write_request import family_statuses_for_products


def codes():
    data = np.zeros(16384, dtype="<u2")
    data[:4] = [17, 8192, 16383, 4095]
    return data


def metadata(uid):
    return waveform_metadata_blob(uid, time_32=65536, time_16=0, timestamp=2**64 - 1)


def request_for(products, *, overwrite=False):
    return replace(
        make_request(overwrite=overwrite), products=products, issues=products.issues,
        family_statuses=family_statuses_for_products(products, family_issue_ids=products.family_issue_ids),
    )


def logical(appid, blob, sequence=0, uid=None):
    return LogicalPacket(appid, sequence, sequence, blob, True, uid, "b05")


def test_waveform_without_predecessor_survives_identity_and_packet_map(tmp_path):
    packets = assign_identities([logical(0x2F2, waveform_blob(codes()))])
    assert len(packets) == 1
    directory = tmp_path / "session"
    write_uncrater_session(Session(0, packets), directory)
    products = read_uncrater_session(directory)
    row = products.waveforms[0]
    assert row.unique_packet_id == 0
    assert row.provenance.uid_source == ""
    assert row.raw_seconds is None and row.adc_timestamp is None
    assert row.provenance.source_packets[0].bank == "b05"
    assert products.quality_status == "partial"
    np.testing.assert_array_equal(row.data[:4], [17, -8192, -1, 4095])


@pytest.mark.parametrize("uid", [0, 0xFFFFFFFF])
def test_packet_map_repairs_inherited_uid_after_ordering(tmp_path, uid):
    packets = [
        logical(0x2F2, waveform_blob(codes()), uid=70),
        logical(0x2FA, metadata(uid), uid=uid),
    ]
    directory = tmp_path / "session"
    write_uncrater_session(Session(0, packets), directory)
    packet_map = json.loads((directory / "packet_map.json").read_text())
    assert [item["unique_packet_id"] for item in packet_map["packets"]] == [uid, uid]
    assert [packet.appid for packet in packets] == [0x2F2, 0x2FA]
    products = read_uncrater_session(directory)
    assert products.waveforms[0].unique_packet_id == uid
    assert products.waveforms[0].provenance.uid_source_role == "waveform_metadata"
    packet_map["packets"][0]["unique_packet_id"] = 10
    (directory / "packet_map.json").write_text(json.dumps(packet_map))
    with pytest.raises(PacketMapError, match="product UID disagrees"):
        read_uncrater_session(directory)


def test_source_order_survives_uid_sequence_sort(tmp_path):
    packets = []
    for ch in range(4):
        packets.append(logical(0x2F0 + ch, waveform_blob(codes()), sequence=30 - ch, uid=3))
    packets.append(logical(0x2FA, metadata(4), uid=4))
    packets = assign_identities(packets)
    assert [p.appid for p in packets[:4]] == [0x2F3, 0x2F2, 0x2F1, 0x2F0]
    directory = tmp_path / "session"
    write_uncrater_session(Session(0, packets), directory)
    products = read_uncrater_session(directory)
    assert len(products.waveforms) == 4
    assert {row.unique_packet_id for row in products.waveforms} == {4}
    assert all(row.raw_seconds == 1 for row in products.waveforms)


@pytest.mark.parametrize("format", ["h5", "fits"])
def test_unresolved_values_round_trip_and_later_metadata_overwrites(tmp_path, format):
    cdi = tmp_path / "cdi_output"
    cdi.mkdir()
    write_packet(cdi, 0, 0x2F2, waveform_blob(codes()))
    products = read_uncrater_session(tmp_path)
    destination = tmp_path / f"waveform.{format}"
    writer = write_hdf5 if format == "h5" else write_fits
    writer(request_for(products), destination)
    bundle = load_bundle(destination)
    assert bundle.waveform_unique_ids.dtype == np.uint32
    assert bundle.waveform_unique_ids.tolist() == [0]
    assert np.isnan(bundle.waveform_raw_times).all()
    assert np.isnan(bundle.waveform_mjd_times).all()
    assert bundle.waveform_adc_timestamps.dtype == np.uint64
    assert bundle.waveform_adc_timestamps.tolist() == [0]
    assert not bundle.waveform_adc_timestamp_valid.any()
    assert bundle.product_records["waveforms"][0].provenance.uid_source == ""
    np.testing.assert_array_equal(bundle.waveform_data[0, :4], [17, -8192, -1, 4095])
    if format == "h5":
        with h5py.File(destination) as h5:
            assert h5["provenance/product_rows/uid_source"].asstr()[0] == ""
            assert not h5["waveform/raw_time_valid"][0]
    write_packet(cdi, 1, 0x2FA, metadata(0))
    products = read_uncrater_session(tmp_path)
    writer(request_for(products, overwrite=True), destination)
    bundle = load_bundle(destination)
    assert bundle.waveform_unique_ids.tolist() == [0]
    assert bundle.waveform_adc_timestamp_valid.all()
    assert bundle.waveform_adc_timestamps.tolist() == [2**64 - 1]
    assert bundle.waveform_raw_times.tolist() == [1.0]
    assert bundle.product_records["waveforms"][0].provenance.uid_source_role == "waveform_metadata"


def test_unmatched_metadata_is_retained_in_hdf5_decoder_report(tmp_path):
    cdi = tmp_path / "cdi_output"
    cdi.mkdir()
    write_packet(cdi, 0, 0x2F0, waveform_blob(codes()))
    write_packet(cdi, 1, 0x2FA, metadata(7))
    write_packet(cdi, 2, 0x2FA, metadata(8))
    products = read_uncrater_session(tmp_path)
    destination = tmp_path / "ambiguous.h5"
    write_hdf5(request_for(products), destination)
    bundle = load_bundle(destination)
    report = json.loads(bundle.decoder_provenance["record"].canonical_report_json)
    orphan_metadata = [row["metadata"] for row in report["unresolved_waveforms"] if row["metadata"]]
    assert [row["unique_packet_id"] for row in orphan_metadata] == [7, 8]
    assert all(row["timestamp"] == 2**64 - 1 for row in orphan_metadata)
    assert bundle.waveform_unique_ids.tolist() == [0]


def test_flash_waveform_ingestion_and_full_input_rerun(tmp_path):
    flash = tmp_path / "FLASH"
    flash.mkdir()
    def frame(appid, blob, sequence=0):
        swapped = np.frombuffer(blob, dtype="<u2").byteswap().tobytes()
        return make_frame(swapped, appid=appid, sequence=sequence)
    payload = frame(0x2F2, waveform_blob(codes()))
    bank = flash / "b05" / BANK_FILENAME
    bank.parent.mkdir()
    bank.write_bytes(payload)
    reference = write_landing_reference(tmp_path, reference_raw_seconds=0.0)
    options = dict(landing_time_file=reference, sessions_root=tmp_path / "sessions",
                   h5_dir=tmp_path / "h5", fits_dir=tmp_path / "fits")
    process_flash(flash, **options)
    h5_path, = (tmp_path / "h5").glob("*.h5")
    first = load_bundle(h5_path)
    assert first.waveform_data.shape == (1, 16384)
    assert not first.waveform_adc_timestamp_valid.any()
    bank.write_bytes(payload + frame(0x2FA, metadata(12)))
    process_flash(flash, overwrite=True, **options)
    h5_path, = (tmp_path / "h5").glob("*.h5")
    second = load_bundle(h5_path)
    assert second.waveform_unique_ids.tolist() == [12]
    assert second.waveform_adc_timestamp_valid.all()
    np.testing.assert_array_equal(first.waveform_data, second.waveform_data)


def hello(uid, *, version=BINDING.canonical_schema_id):
    value = BINDING.pystruct.startup_hello()
    value.SW_version = version
    value.unique_packet_id = uid
    value.time_32 = uid * 65536
    return bytes(value)


def test_corrupt_metadata_slots_survive_identity_assignment(tmp_path):
    packets = []
    for channel in range(4):
        packets.append(logical(0x2F0 + channel, waveform_blob(codes()), sequence=2 * channel))
        packets.append(logical(0x2FA, b"xx" if channel < 3 else metadata(104), sequence=2 * channel + 1))
    with pytest.warns(RuntimeWarning, match="failed to extract unique_packet_id"):
        packets = assign_identities(packets)
    assert len(packets) == 8
    directory = tmp_path / "session"
    write_uncrater_session(Session(0, packets), directory)
    rows = read_uncrater_session(directory).waveforms
    assert {row.channel: row.unique_packet_id for row in rows} == {0: 0, 1: 0, 2: 0, 3: 104}
    assert [row.channel for row in rows if row.provenance.uid_source] == [3]


def test_bank_concatenation_routes_waveforms_to_validated_metadata_session(tmp_path):
    packets = [logical(int(BINDING.appids.AppID_uC_Start), hello(100), sequence=2), logical(int(BINDING.appids.AppID_End_Of_Sequence), b"", sequence=99),
               logical(int(BINDING.appids.AppID_uC_Start), hello(200), sequence=2), logical(int(BINDING.appids.AppID_End_Of_Sequence), b"", sequence=99)]
    for uid in (101, 201):
        for channel in range(4):
            packet = logical(0x2F0 + channel, waveform_blob(np.full(16384, uid, dtype="<u2")), sequence=6 + channel)
            packet.bank = "b06"
            packets.append(packet)
        packet = logical(0x2FA, metadata(uid), sequence=10)
        packet.bank = "b06"
        packets.append(packet)
    sessions = split_sessions(assign_identities(packets))
    assert len(sessions) == 2
    for session, uid in zip(sessions, (101, 201)):
        directory = tmp_path / str(session.ordinal)
        write_uncrater_session(session, directory)
        rows = read_uncrater_session(directory).waveforms
        assert len(rows) == 4
        assert {row.unique_packet_id for row in rows} == {uid}
        assert all(np.all(row.data == uid) for row in rows)
        document = json.loads((directory / "packet_map.json").read_text())
        assert all(item["waveform_metadata_source_order"] is not None
                   for item in document["packets"] if 0x2F0 <= item["original_appid"] <= 0x2F3)


def test_session_subset_does_not_rematch_a_globally_ambiguous_block(tmp_path):
    packets = [logical(0x2F0, waveform_blob(codes()), uid=10),
               logical(0x2FA, metadata(11), uid=11),
               logical(0x2F0, waveform_blob(codes()), uid=20)]
    for index, packet in enumerate(packets):
        packet.file_index = index
    sessions = [Session(0, packets[:2]), Session(1, packets[2:])]
    associate_session_waveforms(sessions)
    directory = tmp_path / "session"
    write_uncrater_session(sessions[0], directory)
    row, = read_uncrater_session(directory).waveforms
    assert row.unique_packet_id == 0
    assert row.provenance.uid_source == ""


def test_packet_map_repair_uses_requested_diagnostic_binding(tmp_path):
    packets = [logical(int(BINDING.appids.AppID_uC_Start), hello(1, version=0x999), uid=1),
               logical(0x2F0, waveform_blob(codes()), uid=1),
               logical(0x2FA, metadata(2), uid=2)]
    directory = tmp_path / "session"
    write_uncrater_session(Session(0, packets), directory, diagnostic_override=True)
    row, = read_uncrater_session(directory, diagnostic_override=True).waveforms
    assert row.unique_packet_id == 2


@pytest.mark.parametrize("loss", ["crc", "reassembly"])
def test_flash_transport_loss_cannot_masquerade_as_a_quartet(tmp_path, loss):
    flash = tmp_path / "FLASH"
    bank = flash / "b05" / BANK_FILENAME
    bank.parent.mkdir(parents=True)
    frames = []
    for channel in range(4):
        frames.append(make_frame(np.frombuffer(waveform_blob(codes()), dtype="<u2").byteswap().tobytes(),
                                 appid=0x2F0 + channel, sequence=2 * channel))
        blob = metadata(101 + channel)
        if channel < 3 and loss == "reassembly":
            blob = b"x"  # Odd science length is dropped before identity assignment
        else:
            blob = np.frombuffer(blob, dtype="<u2").byteswap().tobytes()
        frames.append(make_frame(blob, appid=0x2FA, sequence=2 * channel + 1,
                                 corrupt_crc=channel < 3 and loss == "crc"))
    bank.write_bytes(b"".join(frames))
    reference = write_landing_reference(tmp_path, reference_raw_seconds=0.0)
    with pytest.warns(RuntimeWarning):
        process_flash(flash, landing_time_file=reference, sessions_root=tmp_path / "sessions",
                      h5_dir=tmp_path / "h5")
    path, = (tmp_path / "h5").glob("*.h5")
    bundle = load_bundle(path)
    assert bundle.waveform_data.shape == (4, 16384)
    assert bundle.waveform_unique_ids.tolist() == [0, 0, 0, 0]
    assert not bundle.waveform_adc_timestamp_valid.any()
    assert all(row.provenance.uid_source == "" for row in bundle.product_records["waveforms"])
