from __future__ import annotations

from dataclasses import replace

import h5py
import numpy as np
import pytest
from test_layout_v4_hdf5 import make_request

from lusee.ingest import fits_writer
from lusee.ingest.clock_reference import (
    ClockReference,
    ClockReferenceSet,
    ClockSource,
)
from lusee.ingest.hdf5_writer import write_hdf5
from lusee.ingest.layout_v4_reader import (
    LayoutV4ValidationError,
    read_layout_v4_fits,
    read_layout_v4_hdf5,
    validate_layout_v4_tree,
)
from lusee.ingest.layout_v4_tree import (
    assert_layout_trees_equal,
    build_layout_v4_tree,
)
from lusee.ingest.obs_factory import load_bundle
from lusee.ingest.telemetry import TELEMETRY_FIELD_COUNT, TelemetryData


TELEMETRY_DATASETS = {
    "field_names",
    "units",
    "source_indices",
    "mission_seconds",
    "lusee_subsecs",
    "mjd_times",
    "raw_counts",
    "values",
    "valid",
}


def clock_reference() -> ClockReferenceSet:
    return ClockReferenceSet(
        format_version=1,
        reference_event="landing",
        clock_reference_isot="2027-05-01T00:00:00",
        time_scale="utc",
        clocks=(ClockReference(ClockSource.DCB, 1000.0),),
        source="synthetic telemetry layout test",
        assumed=True,
        source_sha256="e" * 64,
    )


def telemetry_data(
    *,
    row_count: int = 2,
    source_kind: str = "b01_0x314",
    reference: ClockReferenceSet | None = None,
) -> TelemetryData:
    names = tuple(f"field_{index:02d}" for index in range(TELEMETRY_FIELD_COUNT))
    units = tuple(
        "V" if index % 2 else "degC"
        for index in range(TELEMETRY_FIELD_COUNT)
    )
    if row_count == 0:
        mission_seconds = np.empty(0, dtype=np.uint32)
        subseconds = np.empty(0, dtype=np.uint16)
    else:
        mission_seconds = np.arange(1000, 1000 + row_count, dtype=np.uint32)
        subseconds = np.arange(row_count, dtype=np.uint16) * np.uint16(32768)
    raw_seconds = (
        mission_seconds.astype(np.float64)
        + subseconds.astype(np.float64) / 65536.0
    )
    mjd_times = (
        np.asarray(
            reference.to_mjd(raw_seconds, clock_source=ClockSource.DCB),
            dtype=np.float64,
        )
        if reference is not None
        else np.full(row_count, np.nan, dtype=np.float64)
    )
    raw_counts = np.arange(
        row_count * TELEMETRY_FIELD_COUNT,
        dtype=np.uint16,
    ).reshape(row_count, TELEMETRY_FIELD_COUNT)
    values = raw_counts.astype(np.float64)
    valid = np.ones(values.shape, dtype=np.bool_)
    if row_count:
        values[-1, -1] = np.nan
        valid[-1, -1] = False
    return TelemetryData(
        source_kind=source_kind,
        field_names=names,
        units=units,
        source_indices=np.arange(0, 2 * row_count, 2, dtype=np.int64),
        mission_seconds=mission_seconds,
        lusee_subsecs=subseconds,
        mjd_times=mjd_times,
        raw_counts=raw_counts,
        values=values,
        valid=valid,
    )


def telemetry_request(
    *,
    row_count: int = 2,
    source_kind: str = "b01_0x314",
    with_dcb_reference: bool = True,
):
    base = make_request()
    reference = clock_reference() if with_dcb_reference else None
    telemetry = telemetry_data(
        row_count=row_count,
        source_kind=source_kind,
        reference=reference,
    )
    return replace(
        base,
        telemetry=telemetry,
        clock_reference_set=reference,
        clock_reference_unavailable_reason=(
            None if reference is not None else "DCB clock unavailable"
        ),
    )


def test_absent_telemetry_is_omitted(tmp_path):
    destination = tmp_path / "absent.h5"

    write_hdf5(make_request(), destination)

    with h5py.File(destination, "r") as h5:
        assert "telemetry" not in h5


def test_fixed_telemetry_table_round_trips_with_hdf5_fits_parity(tmp_path):
    request = telemetry_request()
    hdf5_path = tmp_path / "telemetry.h5"
    fits_path = tmp_path / "telemetry.fits"

    write_hdf5(request, hdf5_path)
    fits_writer.write_fits(request, fits_path)

    with h5py.File(hdf5_path, "r") as h5:
        group = h5["telemetry"]
        assert set(group.attrs) == {"source_kind"}
        assert group.attrs["source_kind"] == "b01_0x314"
        assert set(group) == TELEMETRY_DATASETS
        assert group["source_indices"].dtype == np.dtype(np.int64)
        assert group["mission_seconds"].dtype == np.dtype(np.uint32)
        assert group["lusee_subsecs"].dtype == np.dtype(np.uint16)
        assert group["mjd_times"].dtype == np.dtype(np.float64)
        assert group["raw_counts"].shape == (2, TELEMETRY_FIELD_COUNT)
        assert group["values"].shape == (2, TELEMETRY_FIELD_COUNT)
        assert group["valid"].shape == (2, TELEMETRY_FIELD_COUNT)
        assert "raw_seconds" not in group
        assert "mjd_time_valid" not in group

    observed_hdf5 = read_layout_v4_hdf5(hdf5_path)
    observed_fits = read_layout_v4_fits(fits_path)
    assert_layout_trees_equal(
        observed_hdf5,
        observed_fits,
        context="telemetry HDF5/FITS parity",
    )


def test_zero_row_sidecar_remains_present(tmp_path):
    request = telemetry_request(
        row_count=0,
        source_kind="legacy_binary_sidecar",
    )
    hdf5_path = tmp_path / "empty-sidecar.h5"
    fits_path = tmp_path / "empty-sidecar.fits"

    write_hdf5(request, hdf5_path)
    fits_writer.write_fits(request, fits_path)
    observed_hdf5 = read_layout_v4_hdf5(hdf5_path)
    observed_fits = read_layout_v4_fits(fits_path)
    assert_layout_trees_equal(
        observed_hdf5,
        observed_fits,
        context="empty telemetry HDF5/FITS parity",
    )

    with h5py.File(hdf5_path, "r") as h5:
        group = h5["telemetry"]
        assert group.attrs["source_kind"] == "legacy_binary_sidecar"
        assert group["field_names"].shape == (TELEMETRY_FIELD_COUNT,)
        assert group["source_indices"].shape == (0,)
        assert group["raw_counts"].shape == (0, TELEMETRY_FIELD_COUNT)
        assert group["values"].shape == (0, TELEMETRY_FIELD_COUNT)
        assert group["valid"].shape == (0, TELEMETRY_FIELD_COUNT)


def test_strict_reader_rejects_noncanonical_telemetry_tree():
    tree = build_layout_v4_tree(
        telemetry_request(),
        destination_preexisted=False,
    )
    telemetry = tree["telemetry"]
    telemetry.create_dataset("raw_seconds", data=np.asarray([1.0, 2.0]))

    with pytest.raises(
        LayoutV4ValidationError,
        match="telemetry tree is not canonical",
    ):
        validate_layout_v4_tree(tree)


def test_strict_reader_repeats_dcb_mjd_equation():
    tree = build_layout_v4_tree(
        telemetry_request(),
        destination_preexisted=False,
    )
    mjd_times = tree["telemetry/mjd_times"]
    mjd_times.data[0] = np.nextafter(mjd_times.data[0], np.inf)

    with pytest.raises(
        LayoutV4ValidationError,
        match="MJD contradicts the DCB clock reference",
    ):
        validate_layout_v4_tree(tree)


def test_strict_reader_requires_nan_mjd_without_dcb_reference():
    tree = build_layout_v4_tree(
        telemetry_request(with_dcb_reference=False),
        destination_preexisted=False,
    )
    validate_layout_v4_tree(tree)
    tree["telemetry/mjd_times"].data[0] = 60000.0

    with pytest.raises(
        LayoutV4ValidationError,
        match="MJD requires a DCB clock reference",
    ):
        validate_layout_v4_tree(tree)


def test_multi_file_reader_preserves_per_file_sources_and_absence(tmp_path):
    reference = clock_reference()
    first = telemetry_request()
    second_telemetry = telemetry_data(
        source_kind="legacy_binary_sidecar",
        reference=reference,
    )
    second = replace(first, telemetry=second_telemetry)
    absent = replace(first, telemetry=None)
    paths = (
        tmp_path / "01-b01.h5",
        tmp_path / "02-sidecar.fits",
        tmp_path / "03-absent.h5",
    )
    write_hdf5(first, paths[0])
    fits_writer.write_fits(second, paths[1])
    write_hdf5(absent, paths[2])

    bundle = load_bundle(paths)

    assert tuple(
        None if item is None else item.source_kind
        for item in bundle.telemetry_sessions
    ) == ("b01_0x314", "legacy_binary_sidecar", None)
    assert bundle.telemetry is None
    assert bundle.dcb_fpga["mission_seconds"].shape == (4,)
    assert bundle.dcb_fpga["field_00"].shape == (4,)


@pytest.mark.parametrize("contract_field", ("field_names", "units"))
def test_multi_file_reader_rejects_incompatible_field_contracts(
    tmp_path,
    contract_field,
):
    first = telemetry_request()
    telemetry = first.telemetry
    assert telemetry is not None
    changed = list(getattr(telemetry, contract_field))
    changed[-1] = "different"
    second = replace(
        first,
        telemetry=replace(telemetry, **{contract_field: tuple(changed)}),
    )
    paths = (tmp_path / "01.h5", tmp_path / "02.h5")
    write_hdf5(first, paths[0])
    write_hdf5(second, paths[1])

    with pytest.raises(
        ValueError,
        match="incompatible field names or units",
    ):
        load_bundle(paths)
