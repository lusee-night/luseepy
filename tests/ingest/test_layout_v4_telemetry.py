from __future__ import annotations

from dataclasses import replace

import h5py
import numpy as np
import pytest
from astropy.io import fits
from test_layout_v4_hdf5 import make_request

from lusee.ingest import fits_writer, viz
from lusee.ingest import telemetry as telemetry_module
from lusee.ingest.clock_reference import (
    ClockReference,
    ClockReferenceSet,
    ClockSource,
)
from lusee.ingest.hdf5_writer import write_hdf5
from lusee.ingest.issues import IssueAction, IssueCollector, IssueSeverity
from lusee.ingest.layout_v4_reader import LayoutV4ValidationError
from lusee.ingest.layout_v4_tree import (
    assert_layout_trees_equal,
    build_layout_v4_tree,
)
from lusee.ingest.obs_factory import IngestData, _load_fits, _load_h5, load_bundle
from lusee.ingest.telemetry import (
    TelemetryBlock,
    TelemetryCollection,
    TelemetryCounts,
    TelemetryCoverage,
    TelemetryDecodeResult,
    TelemetryDecoderInfo,
    TelemetryDecoderStatus,
    TelemetryFieldMetadata,
    TelemetryInputState,
)
from lusee.ingest.write_request import family_statuses_for_products


def telemetry_request(raw_second=1000.25):
    base = make_request()
    clock_reference = ClockReferenceSet(
        format_version=1,
        reference_event="landing",
        clock_reference_isot="2027-05-01T00:00:00",
        time_scale="utc",
        clocks=(
            ClockReference(ClockSource.SPECTROMETER, 10.0),
            ClockReference(ClockSource.DCB, 1000.0),
        ),
        source="synthetic telemetry layout test",
        assumed=True,
        source_sha256="e" * 64,
    )
    whole_second = int(np.floor(raw_second))
    subsecond = round((raw_second - whole_second) * 65536.0)
    raw_seconds = np.asarray([raw_second], dtype=np.float64)
    mjd_times = np.asarray(
        clock_reference.to_mjd(raw_seconds, clock_source=ClockSource.DCB),
        dtype=np.float64,
    )
    fpga = TelemetryBlock(
        source_kind="b01_0x314",
        field_names=("temperature", "raw_adc"),
        input_indices=np.asarray([0], dtype=np.int64),
        mission_seconds=np.asarray([whole_second], dtype=np.uint32),
        lusee_subsecs=np.asarray([subsecond], dtype=np.uint16),
        raw_seconds=raw_seconds,
        mjd_times=mjd_times,
        mjd_time_valid=np.asarray([True], dtype=np.bool_),
        raw_counts=np.asarray([[0, 0]], dtype=np.uint16),
        values=np.asarray([[0.0, np.nan]], dtype=np.float64),
        valid=np.asarray([[True, False]], dtype=np.bool_),
    )
    encoder = TelemetryBlock(
        source_kind="b01_0x325",
        field_names=(),
        input_indices=np.empty(0, dtype=np.int64),
        mission_seconds=np.empty(0, dtype=np.uint32),
        lusee_subsecs=np.empty(0, dtype=np.uint16),
        raw_seconds=np.empty(0, dtype=np.float64),
        mjd_times=np.empty(0, dtype=np.float64),
        mjd_time_valid=np.empty(0, dtype=np.bool_),
        raw_counts=np.empty((0, 0), dtype=np.uint16),
        values=np.empty((0, 0), dtype=np.float64),
        valid=np.empty((0, 0), dtype=np.bool_),
    )
    collector = IssueCollector()
    collector.record(
        code="telemetry_decoder.encoder_layout_unvalidated",
        severity=IssueSeverity.WARNING,
        stage="telemetry_decode",
        message="encoder measurements remain unavailable",
        action=IssueAction.REJECTED,
        appid=0x325,
    )
    telemetry = TelemetryDecodeResult(
        input_source="b01",
        input_state=TelemetryInputState.PRESENT,
        decoder_status=TelemetryDecoderStatus.AVAILABLE,
        coverage=TelemetryCoverage.PARTIAL,
        decoder_info=TelemetryDecoderInfo(
            api_version=1,
            decoder_name="generic-test-decoder",
            decoder_version="test-1",
            claimed_appids=(0x314, 0x325),
        ),
        field_metadata=(
            TelemetryFieldMetadata(
                "temperature", "K", "continuous", "linear", "thermal"
            ),
            TelemetryFieldMetadata(
                "raw_adc",
                "count",
                "uncalibrated_raw_count",
                "none",
                None,
            ),
        ),
        fpga=fpga,
        encoder=encoder,
        counts=TelemetryCounts(
            source="b01",
            scalar_counts=(
                ("input_packet_count", 2),
                ("claimed_packet_count", 2),
                ("unclaimed_packet_count", 0),
                ("fpga_input_packet_count", 1),
                ("fpga_output_record_count", 1),
                ("fpga_dropped_packet_count", 0),
                ("encoder_input_packet_count", 1),
                ("encoder_output_record_count", 0),
                ("encoder_rejected_packet_count", 1),
            ),
            claimed_appid_counts=((0x314, 1), (0x325, 1)),
        ),
        issues=collector.issues,
    )
    return replace(
        base,
        clock_reference_set=clock_reference,
        clock_reference_unavailable_reason=None,
        issues=telemetry.issues,
        family_statuses=family_statuses_for_products(
            base.products,
            family_issue_ids={},
            telemetry=telemetry,
        ),
        telemetry=telemetry,
    )


def present_empty_telemetry_request():
    base = telemetry_request()
    source = base.telemetry.fpga
    fpga = TelemetryBlock(
        source_kind=source.source_kind,
        field_names=source.field_names,
        input_indices=np.empty(0, dtype=np.int64),
        mission_seconds=np.empty(0, dtype=np.uint32),
        lusee_subsecs=np.empty(0, dtype=np.uint16),
        raw_seconds=np.empty(0, dtype=np.float64),
        mjd_times=np.empty(0, dtype=np.float64),
        mjd_time_valid=np.empty(0, dtype=np.bool_),
        raw_counts=np.empty((0, len(source.field_names)), dtype=np.uint16),
        values=np.empty((0, len(source.field_names)), dtype=np.float64),
        valid=np.empty((0, len(source.field_names)), dtype=np.bool_),
    )
    telemetry = replace(
        base.telemetry,
        input_state=TelemetryInputState.PRESENT_EMPTY,
        coverage=TelemetryCoverage.PRESENT_EMPTY,
        fpga=fpga,
        counts=TelemetryCounts(
            source="b01",
            scalar_counts=(
                ("input_packet_count", 0),
                ("claimed_packet_count", 0),
                ("unclaimed_packet_count", 0),
                ("fpga_input_packet_count", 0),
                ("fpga_output_record_count", 0),
                ("fpga_dropped_packet_count", 0),
                ("encoder_input_packet_count", 0),
                ("encoder_output_record_count", 0),
                ("encoder_rejected_packet_count", 0),
            ),
        ),
        issues=(),
    )
    return replace(
        base,
        telemetry=telemetry,
        issues=(),
        family_statuses=family_statuses_for_products(
            base.products,
            family_issue_ids={},
            telemetry=telemetry,
        ),
    )


def unassigned_telemetry_request():
    base = telemetry_request()
    telemetry = base.telemetry.with_blocks(
        fpga=base.telemetry.fpga.slice_rows(
            np.zeros(base.telemetry.fpga.row_count, dtype=np.bool_)
        ),
        unassigned_fpga=base.telemetry.fpga,
    )
    return replace(
        base,
        telemetry=telemetry,
        family_statuses=family_statuses_for_products(
            base.products,
            family_issue_ids={},
            telemetry=telemetry,
        ),
    )


def failed_telemetry_request(status):
    base = telemetry_request()
    collector = IssueCollector()
    issue = collector.record(
        code=f"telemetry_decoder.{status.value}",
        severity=IssueSeverity.ERROR,
        stage="telemetry_decode",
        message="synthetic telemetry decoder failure",
        action=IssueAction.REJECTED,
    )
    telemetry = TelemetryDecodeResult(
        input_source="b01",
        input_state=TelemetryInputState.PRESENT,
        decoder_status=status,
        coverage=TelemetryCoverage(status.value),
        decoder_info=base.telemetry.decoder_info,
        field_metadata=base.telemetry.field_metadata,
        issues=(issue,),
    )
    return replace(
        base,
        telemetry=telemetry,
        issues=(issue,),
        family_statuses=family_statuses_for_products(
            base.products,
            family_issue_ids={},
            telemetry=telemetry,
        ),
    )


def legacy_sidecar_telemetry_request():
    base = telemetry_request()
    fpga = replace(
        base.telemetry.fpga,
        source_kind="legacy_binary_sidecar",
    )
    telemetry = TelemetryDecodeResult(
        input_source="legacy_sidecar",
        input_state=TelemetryInputState.PRESENT,
        decoder_status=TelemetryDecoderStatus.AVAILABLE,
        coverage=TelemetryCoverage.DECODED,
        decoder_info=base.telemetry.decoder_info,
        field_metadata=base.telemetry.field_metadata,
        fpga=fpga,
        counts=TelemetryCounts(
            source="legacy_sidecar",
            scalar_counts=(
                ("input_byte_count", 32),
                ("complete_record_count", 1),
                ("trailing_byte_count", 0),
                ("output_record_count", 1),
                ("dropped_record_count", 0),
            ),
        ),
    )
    return replace(
        base,
        telemetry=telemetry,
        issues=(),
        family_statuses=family_statuses_for_products(
            base.products,
            family_issue_ids={},
            telemetry=telemetry,
        ),
    )


def test_hdf5_telemetry_tree_preserves_typed_matrices(tmp_path):
    request = telemetry_request()
    destination = tmp_path / "telemetry.h5"

    write_hdf5(request, destination)

    with h5py.File(destination, "r") as h5:
        assert h5.attrs["quality_status"] == "partial"
        telemetry = h5["telemetry"]
        assert telemetry.attrs["input_state"] == "present"
        assert telemetry.attrs["decoder_status"] == "available"
        assert telemetry.attrs["coverage"] == "partial"
        assert telemetry["issue_refs/issue_index"][:].tolist() == [0]
        fpga = telemetry["fpga"]
        assert fpga["raw_counts"].shape == (1, 2)
        assert fpga["raw_counts"].dtype == np.dtype(np.uint16)
        assert fpga["values"].dtype == np.dtype(np.float64)
        assert fpga["valid"].dtype == np.dtype(np.bool_)
        assert fpga["values"][0, 0] == 0.0
        assert fpga["valid"][0, 0]
        assert np.isnan(fpga["values"][0, 1])
        assert not fpga["valid"][0, 1]
        assert fpga["field_metadata/name"].asstr()[:].tolist() == [
            "temperature",
            "raw_adc",
        ]
        assert telemetry["encoder/raw_counts"].shape == (0, 0)
        assert telemetry["encoder"].attrs["measurement_status"] == "unvalidated"
        assert "unassigned_fpga" not in telemetry

    loaded = _load_h5(destination)
    assert loaded.telemetry.fpga.raw_counts.tolist() == [[0, 0]]
    assert loaded.telemetry.fpga.valid.tolist() == [[True, False]]
    assert loaded.telemetry.issue_ids == ("issue-00000001",)


def test_v4_plot_grouping_uses_persisted_metadata(tmp_path, monkeypatch):
    destination = tmp_path / "telemetry.h5"
    write_hdf5(telemetry_request(), destination)
    bundle = _load_h5(destination)

    def fail_live_lookup():
        raise AssertionError("v4 plotting must not consult the live decoder")

    monkeypatch.setattr(telemetry_module, "field_groups", fail_live_lookup)

    assert viz._telemetry_field_groups(bundle) == {
        "thermal": ("temperature",),
        "other": ("raw_adc",),
    }


def test_present_empty_plot_is_not_reported_as_written(tmp_path):
    destination = tmp_path / "empty-telemetry.h5"
    output = tmp_path / "telemetry.png"
    write_hdf5(present_empty_telemetry_request(), destination)

    with pytest.raises(FileNotFoundError, match="zero samples"):
        viz.plot_dcb_telemetry(_load_h5(destination), output)

    assert not output.exists()


def test_unassigned_only_telemetry_is_labeled_in_both_plot_paths(
    tmp_path,
    monkeypatch,
):
    destination = tmp_path / "unassigned-telemetry.h5"
    output = tmp_path / "telemetry.png"
    write_hdf5(unassigned_telemetry_request(), destination)
    bundle = _load_h5(destination)

    unassigned = viz._unassigned_telemetry_values(bundle)
    assert unassigned["raw_seconds"].tolist() == [1000.25]
    assert unassigned["temperature"].tolist() == [0.0]

    class FakeAxis:
        def __init__(self):
            self.labels = []
            self.title = None
            self.xlabel = None

        def plot(self, *args, label=None, **kwargs):
            self.labels.append(label)

        def set_title(self, title):
            self.title = title

        def set_xlabel(self, label):
            self.xlabel = label

        def set_ylabel(self, label):
            pass

        def legend(self, **kwargs):
            pass

        def grid(self, *args, **kwargs):
            pass

    class FakeFigure:
        def tight_layout(self):
            pass

        def savefig(self, *args, **kwargs):
            pass

    class FakePyplot:
        def __init__(self):
            self.axes = []

        def subplots(self, nrows, ncols, **kwargs):
            self.axes = [FakeAxis() for _ in range(nrows)]
            return FakeFigure(), self.axes

        def close(self, fig):
            pass

    fake_pyplot = FakePyplot()
    monkeypatch.setattr(viz, "_require_matplotlib", lambda: fake_pyplot)

    assert viz.plot_dcb_telemetry(bundle, output) == output
    assert all("(unassigned)" in axis.title for axis in fake_pyplot.axes)
    assert fake_pyplot.axes[-1].xlabel == (
        "seconds since first unassigned telemetry sample"
    )

    data = object.__new__(IngestData)
    data.dcb_telemetry = bundle.dcb_fpga
    data.telemetry_fpga = bundle.telemetry_fpga
    axis = FakeAxis()
    data.plot_dcb(ax=axis)
    assert axis.labels == ["temperature", "raw_adc"]
    assert axis.title == "Unassigned DCB telemetry"
    assert axis.xlabel == "seconds since first unassigned telemetry sample"


@pytest.mark.parametrize(
    ("suffix", "writer", "loader"),
    [
        (".h5", write_hdf5, _load_h5),
        (".fits", fits_writer.write_fits, _load_fits),
    ],
)
@pytest.mark.parametrize(
    "status",
    [TelemetryDecoderStatus.BROKEN, TelemetryDecoderStatus.INCOMPATIBLE],
)
def test_failed_decoder_roundtrip_preserves_resolved_identity_and_metadata(
    tmp_path,
    suffix,
    writer,
    loader,
    status,
):
    request = failed_telemetry_request(status)
    path = tmp_path / f"failed-telemetry{suffix}"

    writer(request, path)
    telemetry = loader(path).telemetry

    assert telemetry.decoder_status is status
    assert telemetry.decoder_info == request.telemetry.decoder_info
    assert telemetry.field_metadata == request.telemetry.field_metadata
    assert telemetry.issue_ids == request.telemetry.issue_ids


def test_fits_uses_the_identical_telemetry_logical_tree(tmp_path):
    request = telemetry_request()
    destination = tmp_path / "telemetry.fits"

    fits_writer.write_fits(request, destination)

    expected = build_layout_v4_tree(
        request,
        destination_preexisted=False,
    )
    with fits.open(destination, uint=True, memmap=False) as hdul:
        observed = fits_writer._read_layout_tree(hdul, fits)
    assert_layout_trees_equal(
        expected,
        observed,
        context="telemetry HDF5/FITS logical parity",
    )
    loaded = _load_fits(destination)
    assert loaded.telemetry.fpga.values[0, 0] == 0.0
    assert np.isnan(loaded.telemetry.fpga.values[0, 1])


def test_hdf5_reader_rejects_mask_value_corruption(tmp_path):
    destination = tmp_path / "corrupt-telemetry.h5"
    write_hdf5(telemetry_request(), destination)
    with h5py.File(destination, "r+") as h5:
        h5["telemetry/fpga/values"][0, 1] = 0.0

    with pytest.raises(LayoutV4ValidationError, match="finite exactly where"):
        _load_h5(destination)


def test_hdf5_reader_rejects_reserved_engineering_field_name(tmp_path):
    destination = tmp_path / "reserved-telemetry-field.h5"
    write_hdf5(telemetry_request(), destination)
    with h5py.File(destination, "r+") as h5:
        h5["telemetry/fpga/field_metadata/name"][0] = "raw_seconds"

    with pytest.raises(LayoutV4ValidationError, match="reserved for public arrays"):
        _load_h5(destination)


def test_fits_reader_rejects_semantic_corruption_with_valid_checksum(tmp_path):
    destination = tmp_path / "corrupt-telemetry.fits"
    fits_writer.write_fits(telemetry_request(), destination)
    with fits.open(destination, mode="update", uint=True, memmap=False) as hdul:
        target = next(
            hdu
            for hdu in hdul[1:]
            if hdu.header.get("LUSEEPTH") == "/telemetry/fpga"
            and "values" in hdu.columns.names
        )
        target.data["values"][0, 1] = 0.0
        target.add_checksum(override_datasum=True)

    with pytest.raises(LayoutV4ValidationError, match="finite exactly where"):
        _load_fits(destination)


def test_multifile_reader_keeps_source_local_indices_and_boundaries(tmp_path):
    first = tmp_path / "a-telemetry.h5"
    second = tmp_path / "b-telemetry.h5"
    write_hdf5(telemetry_request(), first)
    write_hdf5(telemetry_request(), second)

    merged = load_bundle((second, first))

    assert merged.telemetry is None
    assert len(merged.telemetry_sessions) == 2
    assert merged.telemetry_session_sources == (first, second)
    assert merged.telemetry_fpga.session_row_counts == (1, 1)
    assert merged.telemetry_fpga.session_boundaries == ((0, 1), (1, 2))
    assert merged.telemetry_fpga.session_index.tolist() == [0, 1]
    assert merged.telemetry_fpga.input_indices.tolist() == [0, 0]
    assert merged.telemetry_fpga.values.shape == (2, 2)
    assert merged.telemetry_fpga.raw_counts.shape == (2, 2)
    assert merged.dcb_fpga["raw_seconds"].tolist() == [1000.25, 1000.25]


def test_telemetry_only_files_sort_by_persisted_absolute_time(tmp_path):
    early = tmp_path / "z-early.h5"
    late = tmp_path / "a-late.h5"
    write_hdf5(telemetry_request(1000.25), early)
    write_hdf5(telemetry_request(2000.25), late)

    merged = load_bundle((late, early))

    assert merged.telemetry_session_sources == (early, late)
    assert merged.telemetry_fpga.raw_seconds.tolist() == [1000.25, 2000.25]


def test_same_input_unassigned_rows_are_not_duplicated_across_sessions(tmp_path):
    first = tmp_path / "a-unassigned.h5"
    second = tmp_path / "b-unassigned.h5"
    write_hdf5(unassigned_telemetry_request(), first)
    write_hdf5(unassigned_telemetry_request(), second)

    merged = load_bundle((second, first))
    values = merged.telemetry_fpga.unassigned_engineering_values()

    assert merged.telemetry_fpga.session_unassigned_row_counts == (1, 1)
    assert merged.telemetry_fpga.session_input_identities == (
        ("sha256", "fixture-sha256"),
        ("sha256", "fixture-sha256"),
    )
    assert values["raw_seconds"].tolist() == [1000.25]
    assert values["session_index"].tolist() == [-1]
    assert values["source_index"].tolist() == [0]


def test_ambiguous_unassigned_rows_keep_unassigned_label_with_selector():
    result = unassigned_telemetry_request().telemetry
    collection = TelemetryCollection(
        (result, result),
        session_sources=(None, None),
        session_input_identities=(None, None),
    )

    with pytest.raises(ValueError, match="session_index"):
        collection.unassigned_engineering_values()

    values = collection.unassigned_engineering_values(session_index=1)

    assert values["session_index"].tolist() == [-1]
    assert values["source_index"].tolist() == [1]


def test_multifile_reader_preserves_mixed_b01_and_sidecar_sources(tmp_path):
    b01_path = tmp_path / "a-b01.h5"
    sidecar_path = tmp_path / "b-sidecar.h5"
    write_hdf5(telemetry_request(), b01_path)
    write_hdf5(legacy_sidecar_telemetry_request(), sidecar_path)

    merged = load_bundle((sidecar_path, b01_path))

    assert merged.telemetry_fpga.session_row_counts == (1, 1)
    assert merged.telemetry_fpga.source_kind == (
        "b01_0x314",
        "legacy_binary_sidecar",
    )
    assert [result.input_source for result in merged.telemetry_sessions] == [
        "b01",
        "legacy_sidecar",
    ]


def test_multifile_reader_keeps_present_empty_session_boundary(tmp_path):
    empty_path = tmp_path / "a-empty.h5"
    data_path = tmp_path / "b-data.h5"
    write_hdf5(present_empty_telemetry_request(), empty_path)
    write_hdf5(telemetry_request(), data_path)

    merged = load_bundle((data_path, empty_path))

    assert merged.telemetry_fpga.session_row_counts == (1, 0)
    assert merged.telemetry_fpga.session_boundaries == ((0, 1), (1, 1))
    assert [result.coverage.value for result in merged.telemetry_sessions] == [
        "partial",
        "present_empty",
    ]


def test_ingest_data_repr_uses_typed_telemetry_row_count():
    telemetry = present_empty_telemetry_request().telemetry
    full = telemetry_request().telemetry
    empty_selector = np.zeros(full.fpga.row_count, dtype=np.bool_)
    unassigned = full.with_blocks(
        fpga=full.fpga.slice_rows(empty_selector),
        unassigned_fpga=full.fpga,
    )

    def data_for(result):
        data = object.__new__(IngestData)
        data.spectra = None
        data.tr_spectra = None
        data.zoom_spectra = None
        data.source_paths = []
        data.telemetry_fpga = TelemetryCollection(
            (result,),
            session_sources=(None,),
        )
        data.dcb_telemetry = data.telemetry_fpga.engineering_values()
        return data

    assert "telemetry=False" in repr(data_for(telemetry))
    assert "telemetry=True" in repr(data_for(unassigned))


def test_reader_rejects_noncanonical_telemetry_family_status(tmp_path):
    destination = tmp_path / "corrupt-family-status.h5"
    write_hdf5(present_empty_telemetry_request(), destination)
    with h5py.File(destination, "r+") as h5:
        families = h5["status/families/family"].asstr()[:]
        index = int(np.flatnonzero(families == "dcb_telemetry")[0])
        h5["status/families/coverage"][index] = "absent_in_input"

    with pytest.raises(LayoutV4ValidationError, match="contradicts typed"):
        _load_h5(destination)


def test_reader_rejects_b01_count_equation_corruption(tmp_path):
    destination = tmp_path / "corrupt-telemetry-counts.h5"
    write_hdf5(telemetry_request(), destination)
    with h5py.File(destination, "r+") as h5:
        h5["telemetry/counts"].attrs["fpga_dropped_packet_count"] = np.uint64(1)

    with pytest.raises(LayoutV4ValidationError, match="count equations disagree"):
        _load_h5(destination)
