"""Strict high-level reader coverage for ingest layout v4."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest
from test_layout_v4_hdf5 import make_all_family_request, make_request

from lusee.ingest.fits_writer import write_fits
from lusee.ingest.frequency_contract import (
    UnresolvedFrequencyCoordinateError,
    spectrometer_frequency_window,
)
from lusee.ingest.hdf5_writer import write_hdf5
from lusee.ingest.issues import (
    IngestIssue,
    IssueAction,
    IssueSeverity,
)
from lusee.ingest.layout_v4_reader import LayoutV4ValidationError
from lusee.ingest.obs_factory import (
    IngestData,
    MixedFrequencyGridError,
    SessionBundle,
    _bundle_sort_key,
    _concat_bundles,
    load,
    load_bundle,
)
from lusee.ingest.products import DataQuality
from lusee.ingest.viz import plot_adc_stats, plot_spectra_mean
from lusee.ingest.write_request import WriteRequest, family_statuses_for_products


@pytest.fixture(params=("h5", "fits"))
def all_family_file(tmp_path: Path, request: pytest.FixtureRequest) -> Path:
    write_request = make_all_family_request()
    destination = tmp_path / f"all-products.{request.param}"
    writer = write_hdf5 if request.param == "h5" else write_fits
    writer(write_request, destination)
    return destination


def make_issue_request() -> WriteRequest:
    base = make_request()
    issue = IngestIssue(
        issue_id="issue-00000001",
        code="decode.fixture_issue",
        severity=IssueSeverity.WARNING,
        stage="decode",
        message="fixture issue",
        action=IssueAction.KEPT,
    )
    products = base.products
    products.decode_provenance = replace(
        products.decode_provenance,
        issue_counts=(("fixture_issue", 1),),
    )
    products.issues = (issue,)
    products.quality_status = DataQuality.PARTIAL
    return replace(
        base,
        issues=(issue,),
        family_statuses=family_statuses_for_products(
            products,
            family_issue_ids={"housekeeping": (issue.issue_id,)},
        ),
    )


def test_v4_bundle_reads_every_persisted_family(all_family_file: Path):
    bundle = load_bundle(all_family_file)

    assert bundle.spectra.shape == (1, 16, 2048)
    assert bundle.tr_spectra.shape == (1, 16, 4, 4)
    assert bundle.zoom_spectra.shape == (1, 4, 64)
    assert bundle.waveform_data.shape == (1, 16384)
    assert bundle.waveform_adc_timestamps.dtype == np.dtype(np.uint64)
    assert bundle.waveform_adc_timestamps[0] == np.iinfo(np.uint64).max
    assert bundle.grimm_spectra.shape == (1, 2, 16, 4)
    assert len(bundle.housekeeping_fields) == 2
    assert set(bundle.calibrator) == {"metadata", "data", "raw_pfb", "debug"}
    assert bundle.clock_reference_set is not None
    assert bundle.spectra_frequency_windows[0].navgf == 2


def test_v4_bundle_rejects_non_nan_normal_tail(tmp_path: Path):
    destination = tmp_path / "bad-tail.h5"
    write_hdf5(make_all_family_request(), destination)
    with h5py.File(destination, "r+") as handle:
        handle["spectra/data"][0, 3, -1] = np.float32(1.0)

    with pytest.raises(LayoutV4ValidationError, match="tail is not NaN"):
        load_bundle(destination)


@pytest.mark.parametrize("suffix", ("h5", "fits"))
@pytest.mark.parametrize("layout_version", (None, 5, 3.9))
def test_reader_rejects_missing_or_unknown_layout_before_auxiliary_decode(
    tmp_path: Path,
    suffix: str,
    layout_version: object,
):
    destination = tmp_path / f"unsupported-{layout_version}.{suffix}"
    if suffix == "h5":
        with h5py.File(destination, "w") as handle:
            if layout_version is not None:
                handle.attrs["layout_version"] = layout_version
            handle.create_group("tr_spectra").create_dataset(
                "data",
                data=np.empty((0, 16, 1, 1), dtype=np.float64),
            )
    else:
        from astropy.io import fits

        primary = fits.PrimaryHDU()
        if layout_version is not None:
            primary.header["LAYOUTV"] = layout_version
        fits.HDUList(
            [
                primary,
                fits.ImageHDU(
                    data=np.empty((0, 16, 1, 1), dtype=np.float64),
                    name="TR_SPECTRA",
                ),
            ]
        ).writeto(destination)

    with pytest.raises(ValueError, match="layout version"):
        load_bundle(destination)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("empty_source_role", "product provenance row"),
        ("housekeeping_time", "housekeeping row"),
        ("bad_family_quality", "family status"),
        ("adc_spectrum_clock", "MJD coverage|normal-spectrum row"),
        ("calibrator_packet_count", "calibrator metadata row"),
        ("missing_session_validity", "session-invariant attributes"),
        ("missing_decoder_identity", "decoder_name"),
        ("stale_clock_reason", "clock-reference contract"),
        ("noncanonical_clock_json", "record is not canonical"),
        ("bad_compression_level", "compression level"),
        ("root_quality_mismatch", "clean root quality"),
    ),
)
def test_v4_bundle_rejects_semantic_corruption(
    tmp_path: Path,
    mutation: str,
    message: str,
):
    destination = tmp_path / f"{mutation}.h5"
    if mutation == "bad_compression_level":
        write_request = make_request()
    elif mutation == "root_quality_mismatch":
        write_request = make_issue_request()
    else:
        write_request = make_all_family_request()
    write_hdf5(write_request, destination)
    with h5py.File(destination, "r+") as handle:
        if mutation == "empty_source_role":
            handle["provenance/source_packets/role"][0] = ""
        elif mutation == "housekeeping_time":
            handle["housekeeping/hk_type"][0] = np.uint16(3)
        elif mutation == "bad_family_quality":
            families = handle["status/families/family"].asstr()[...]
            index = int(np.flatnonzero(families == "spectra")[0])
            handle["status/families/quality"][index] = "banana"
        elif mutation == "adc_spectrum_clock":
            families = handle["provenance/product_rows/family"].asstr()[...]
            index = int(np.flatnonzero(families == "spectra")[0])
            handle["provenance/product_rows/clock_source"][index] = "adc"
        elif mutation == "calibrator_packet_count":
            handle["calibrator/metadata/from_debug"][0] = np.bool_(True)
        elif mutation == "missing_session_validity":
            del handle["session_invariants"].attrs["start_raw_seconds_valid"]
        elif mutation == "missing_decoder_identity":
            del handle["provenance/decoder"].attrs["decoder_name"]
        elif mutation == "stale_clock_reason":
            handle["clock_reference"].attrs["unavailable_reason"] = "stale"
        elif mutation == "noncanonical_clock_json":
            attrs = handle["clock_reference"].attrs
            attrs["canonical_record_json"] = (
                " " + attrs["canonical_record_json"]
            )
        elif mutation == "bad_compression_level":
            handle["run_provenance"].attrs["hdf5_compression_level"] = 99
        else:
            handle.attrs["quality_status"] = "clean"

    with pytest.raises(LayoutV4ValidationError, match=message):
        load_bundle(destination)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("empty_code", "issue code is empty"),
        ("noncanonical_details", "not canonical"),
        ("invalid_optional_fill", "byte_offset absence"),
    ),
)
def test_v4_bundle_rejects_corrupt_issue_rows(
    tmp_path: Path,
    mutation: str,
    message: str,
):
    destination = tmp_path / f"issue-{mutation}.h5"
    write_hdf5(make_issue_request(), destination)
    with h5py.File(destination, "r+") as handle:
        if mutation == "empty_code":
            handle["issues/code"][0] = ""
        elif mutation == "noncanonical_details":
            handle["issues/details_json"][0] = "{ }"
        else:
            handle["issues/byte_offset"][0] = np.uint64(17)

    with pytest.raises(LayoutV4ValidationError, match=message):
        load_bundle(destination)


def test_v4_high_level_reader_keeps_frequency_coordinate_unresolved(
    all_family_file: Path,
):
    data = load(all_family_file)

    assert data.spectra.shape == (1, 16, 1024)
    assert data.freq is None
    assert data.frequency_coordinate_status == "unresolved"
    telemetry = {
        name: np.ones(1, dtype=np.float64)
        for name in data._GAIN_TELEMETRY_KEYS
    }
    with pytest.raises(UnresolvedFrequencyCoordinateError):
        data.to_physical(levels="H", telemetry=telemetry, gain=object())


def test_mixed_equal_width_windows_split_without_resampling(tmp_path: Path):
    reference_set = make_all_family_request().clock_reference_set
    contracts = (
        spectrometer_frequency_window(3),
        spectrometer_frequency_window(4),
    )
    raw = np.asarray([100.0, 101.0], dtype=np.float64)
    bundle = SessionBundle(
        spectra=np.ones((2, 16, 2048), dtype=np.float32),
        spectra_unique_ids=np.asarray([1, 2], dtype=np.uint32),
        spectra_raw_times=raw,
        spectra_raw_time_valid=np.ones(2, dtype=np.bool_),
        spectra_mjd_times=reference_set.to_mjd(
            raw, clock_source="spectrometer"
        ),
        spectra_mjd_time_valid=np.ones(2, dtype=np.bool_),
        spectra_frequency_counts=np.asarray([512, 512], dtype=np.uint16),
        spectra_navgf=np.asarray([3, 4], dtype=np.uint8),
        spectra_frequency_window_index=np.asarray([0, 1], dtype=np.uint8),
        spectra_frequency_windows=contracts,
        spectra_units="SDU",
        spectra_representation="gain_model_input_sdu",
        spectra_normalization_version=1,
        constants={
            "lun_lat_deg": -23.814,
            "lun_long_deg": 182.258,
            "lun_height_m": 0.0,
        },
        clock_reference_set=reference_set,
        layout_version=4,
    )
    bundle.product_records = {"spectra": ("grid-3", "grid-4")}
    bundle.product_provenance = {
        "product_rows": {
            "family": np.asarray(["spectra", "spectra"], dtype=object),
            "row_index": np.asarray([0, 1], dtype=np.uint64),
        },
        "records": ("provenance-3", "provenance-4"),
    }
    bundle.family_status = {
        "family": np.asarray(["spectra"], dtype=object),
        "persisted_rows": np.asarray([2], dtype=np.uint64),
    }

    with pytest.raises(MixedFrequencyGridError, match="Navgf"):
        IngestData(bundle)
    splits = bundle.split_by_frequency_grid()
    assert [int(split.spectra_navgf[0]) for split in splits] == [3, 4]
    assert [IngestData(split).Nfreq for split in splits] == [512, 512]
    assert [split.product_records["spectra"] for split in splits] == [
        ("grid-3",),
        ("grid-4",),
    ]
    for split in splits:
        assert split.product_provenance["scope"] == "frequency_grid_subset"
        assert "product_rows" not in split.product_provenance
        assert (
            split.product_provenance["records_by_family"]
            is split.product_records
        )
        assert "product_rows" in split.product_provenance["original_artifact"]
        assert split.family_status["scope"] == "frequency_grid_subset"
        assert split.family_status["persisted_rows_by_family"]["spectra"] == 1
        np.testing.assert_array_equal(
            split.family_status["original_artifact"]["persisted_rows"],
            np.asarray([2], dtype=np.uint64),
        )
    written = plot_spectra_mean(bundle, tmp_path / "mixed", products=[0])
    assert {path.name for path in written} == {
        "spectra_mean_navgf3_p00.png",
        "spectra_mean_navgf4_p00.png",
    }

    first_source = tmp_path / "first.h5"
    second_source = tmp_path / "second.h5"
    first = bundle._spectra_subset(np.asarray([0]))
    first.source_path = first_source
    first.source_paths = (first_source,)
    second = bundle._spectra_subset(np.asarray([1]))
    second.source_path = second_source
    second.source_paths = (second_source,)
    merged_splits = _concat_bundles((first, second)).split_by_frequency_grid()

    assert [split.session_spectra_counts for split in merged_splits] == [
        (1, 0),
        (0, 1),
    ]
    assert [
        split.product_provenance["records_by_family"]["spectra"]
        for split in merged_splits
    ] == [("grid-3",), ("grid-4",)]
    assert IngestData(merged_splits[0]).session_boundaries() == [
        (0, 1, first_source),
        (1, 1, second_source),
    ]
    assert IngestData(merged_splits[1]).session_boundaries() == [
        (0, 0, first_source),
        (0, 1, second_source),
    ]


def test_v4_plotters_use_the_validated_bundle(
    all_family_file: Path,
    tmp_path: Path,
):
    means = plot_spectra_mean(
        all_family_file,
        tmp_path / "means",
        products=[3],
    )
    adc = plot_adc_stats(all_family_file, tmp_path / "adc.png")

    assert len(means) == 1 and means[0].is_file()
    assert adc.is_file()


def test_time_source_must_align_with_indexed_normal_rows(
    all_family_file: Path,
):
    bundle = load_bundle(all_family_file)
    mismatched = replace(
        bundle,
        tr_spectra=np.repeat(bundle.tr_spectra, 2, axis=0),
        tr_raw_times=np.asarray([100.0, 101.0], dtype=np.float64),
        tr_raw_time_valid=np.ones(2, dtype=np.bool_),
        tr_mjd_times=bundle.clock_reference_set.to_mjd(
            np.asarray([100.0, 101.0]), clock_source="spectrometer"
        ),
        tr_mjd_time_valid=np.ones(2, dtype=np.bool_),
    )

    with pytest.raises(ValueError, match="not row-aligned"):
        IngestData(mismatched, time_source="tr_spectra")
    with pytest.raises(ValueError, match="unknown time_source"):
        IngestData(bundle, time_source="invented")


def test_multi_session_optional_rows_remain_aligned():
    first = SessionBundle(
        zoom_spectra=np.ones((1, 4, 64), dtype=np.float32),
        zoom_raw_times=np.asarray([1.0]),
        spectra=np.ones((1, 16, 2), dtype=np.float32),
        interp_telemetry={"v1": np.asarray([2.0])},
    )
    second = SessionBundle(
        zoom_spectra=np.ones((1, 4, 64), dtype=np.float32),
        spectra=np.ones((1, 16, 2), dtype=np.float32),
    )

    merged = _concat_bundles((first, second))

    assert merged.zoom_spectra.shape[0] == 2
    np.testing.assert_array_equal(
        merged.zoom_raw_times, np.asarray([1.0, np.nan])
    )
    np.testing.assert_array_equal(
        merged.zoom_raw_time_valid, np.asarray([True, False])
    )
    np.testing.assert_array_equal(
        merged.interp_telemetry["v1"], np.asarray([2.0, np.nan])
    )


def test_multi_session_calibrator_keeps_one_public_array_contract(
    tmp_path: Path,
):
    paths = (tmp_path / "first.h5", tmp_path / "second.h5")
    for path in paths:
        write_hdf5(make_all_family_request(), path)

    merged = load_bundle(paths)

    assert isinstance(merged.calibrator["metadata"], dict)
    assert merged.calibrator["metadata"]["unique_ids"].shape == (4,)
    assert merged.calibrator["data"]["data"].shape == (2, 4, 512)
    assert len(merged.product_records["calibrator_metadata"]) == 4


def test_multi_session_clock_unavailable_reasons_are_diagnostic(
    tmp_path: Path,
):
    paths = (tmp_path / "first.h5", tmp_path / "second.h5")
    for path in paths:
        write_hdf5(make_request(), path)
    with h5py.File(paths[1], "r+") as handle:
        handle["clock_reference"].attrs["unavailable_reason"] = (
            "second fixture has no clock anchor"
        )

    merged = load_bundle(paths)

    assert merged.clock_reference_set is None
    assert merged.clock_reference_unavailable_reasons == (
        "fixture has no clock anchor",
        "second fixture has no clock anchor",
    )
    assert "multiple input sessions" in merged.clock_reference_unavailable_reason


def test_merged_bundle_preserves_high_level_session_boundaries(
    tmp_path: Path,
):
    paths = (tmp_path / "first.h5", tmp_path / "second.h5")
    for path in paths:
        write_hdf5(make_all_family_request(), path)

    data = IngestData(load_bundle(paths))

    assert data.session_boundaries() == [
        (0, 1, paths[0]),
        (1, 2, paths[1]),
    ]


def test_multi_session_order_uses_recorded_session_start(tmp_path: Path):
    paths = (tmp_path / "first.h5", tmp_path / "second.h5")
    requests = []
    for session_start, first_spectrum_time in ((100.0, 200.0), (150.0, 160.0)):
        write_request = make_all_family_request()
        products = write_request.products
        products.start_time_16 = 0
        products.start_time_32 = int(session_start * 65536)
        products.start_raw_seconds = session_start
        spectrum = products.spectra[0]
        products.spectra[0] = replace(
            spectrum,
            raw_seconds=first_spectrum_time,
            metadata=replace(
                spectrum.metadata,
                time_16=0,
                time_32=int(first_spectrum_time * 65536),
            ),
        )
        requests.append(write_request)
    for path, write_request in zip(paths, requests):
        write_hdf5(write_request, path)

    merged = load_bundle((paths[1], paths[0]))

    np.testing.assert_array_equal(
        merged.spectra_raw_times,
        np.asarray([200.0, 160.0]),
    )
    assert merged.session_sources == paths
    assert merged.session_spectra_counts == (1, 1)


def test_bundle_sort_uses_first_stored_science_time_not_minimum(tmp_path: Path):
    first = SessionBundle(
        spectra_raw_times=np.asarray([20.0, 10.0]),
        source_path=tmp_path / "first.h5",
    )
    second = SessionBundle(
        spectra_raw_times=np.asarray([15.0]),
        source_path=tmp_path / "second.h5",
    )

    ordered = sorted((first, second), key=_bundle_sort_key)

    assert ordered == [second, first]


def test_legacy_sort_ignores_unverified_finite_mjd(tmp_path: Path):
    first = SessionBundle(
        spectra_raw_times=np.asarray([20.0]),
        spectra_mjd_times=np.asarray([10.0]),
        clock_reference_set=None,
        source_path=tmp_path / "first.h5",
        layout_version=3,
    )
    second = SessionBundle(
        spectra_raw_times=np.asarray([15.0]),
        spectra_mjd_times=np.asarray([25.0]),
        clock_reference_set=None,
        source_path=tmp_path / "second.h5",
        layout_version=3,
    )

    ordered = sorted((first, second), key=_bundle_sort_key)

    assert ordered == [second, first]
