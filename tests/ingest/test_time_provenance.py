"""Absolute-time provenance for layout v4 and legacy reader adapters."""

from __future__ import annotations

import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from astropy.time import Time
from test_layout_v4_hdf5 import make_all_family_request

from lusee.ingest.clock_reference import (
    ClockReference,
    ClockReferenceSet,
    ClockReferenceUnavailableError,
    LegacyClockReferenceSet,
)
from lusee.ingest.constants import NCHANNELS, NPRODUCTS
from lusee.ingest.fits_writer import write_fits
from lusee.ingest.frequency_contract import spectrometer_frequency_window
from lusee.ingest.hdf5_writer import write_hdf5
from lusee.ingest.obs_factory import IngestData, LegacyIngestWarning, load_bundle
from lusee.ingest.viz import _normal_cube, plot_spectra_mean

h5py = pytest.importorskip("h5py")


def write_legacy_hdf5(
    path: Path,
    layout_version: int,
    *,
    complete_clock: bool = True,
    navgf: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Write the smallest hand-built layout-v2/v3 normal-spectrum file."""
    raw_times = np.array([100.0, 101.0], dtype=np.float64)
    raw_anchor = 100.0
    mjd_anchor = 61000.0
    mjd_times = mjd_anchor + (raw_times - raw_anchor) / 86400.0
    bitslices = np.full((raw_times.size, NPRODUCTS), 30, dtype=np.uint8)
    decoded = np.full(
        (raw_times.size, NPRODUCTS, NCHANNELS),
        4.0,
        dtype=np.float32,
    )

    with h5py.File(path, "w") as handle:
        handle.attrs["layout_version"] = np.uint16(layout_version)
        constants = handle.create_group("constants")
        constants.attrs["lun_lat_deg"] = np.float64(-23.814)
        constants.attrs["lun_long_deg"] = np.float64(182.258)
        constants.attrs["lun_height_m"] = np.float64(0.0)
        if complete_clock:
            constants.attrs["raw_time_subtract_seconds"] = np.float64(raw_anchor)
            constants.attrs["mjd_epoch_offset_days"] = np.float64(mjd_anchor)
            constants.attrs["time_scale"] = "utc"

        spectra = handle.create_group("spectra")
        stored = decoded if layout_version == 2 else np.ldexp(decoded, -1)
        data = spectra.create_dataset("data", data=stored)
        if layout_version == 3:
            data.attrs["units"] = "SDU"
            data.attrs["representation"] = "gain_model_input_sdu"
            data.attrs["bitslice_restored"] = np.uint8(1)
            data.attrs["bitslice_reference"] = np.uint8(31)
            data.attrs["normalization_version"] = np.uint16(1)
        spectra.create_dataset(
            "unique_ids",
            data=np.arange(raw_times.size, dtype=np.uint32),
        )
        spectra.create_dataset("raw_times", data=raw_times)
        if complete_clock:
            spectra.create_dataset("mjd_times", data=mjd_times)
        metadata = spectra.create_group("metadata")
        metadata.create_dataset(
            "actual_bitslice",
            data=(bitslices[:, None, :] if layout_version == 2 else bitslices),
        )
        gains = np.ones((raw_times.size, 4), dtype=np.uint8)
        metadata.create_dataset(
            "actual_gain",
            data=(gains[:, None, :] if layout_version == 2 else gains),
        )
        metadata.create_dataset(
            "Navgf",
            data=np.full(raw_times.size, navgf, dtype=np.uint8),
        )

    return raw_times, mjd_times


@pytest.mark.parametrize(
    ("suffix", "writer"),
    (("h5", write_hdf5), ("fits", write_fits)),
)
def test_layout_v4_round_trips_exact_clock_reference(
    tmp_path: Path,
    suffix: str,
    writer,
):
    request = make_all_family_request()
    reference_set = request.clock_reference_set
    path = tmp_path / f"session.{suffix}"

    writer(request, path)

    bundle = load_bundle(path)
    assert bundle.clock_reference_set == reference_set
    data = IngestData(path)
    assert data.clock_reference_set == reference_set
    np.testing.assert_allclose(
        data.times.mjd,
        reference_set.to_mjd(
            request.products.spectra[0].raw_seconds,
            clock_source="spectrometer",
        ),
        rtol=0.0,
        atol=1e-12,
    )
    assert data.time_provenance == {
        "scale": "utc",
        "source": "spectrometer",
        "assumed": False,
    }


def test_layout_v4_assume_scale_cannot_replace_clock_reference(tmp_path: Path):
    anchored_request = make_all_family_request()
    request = replace(
        anchored_request,
        clock_reference_set=None,
        clock_reference_unavailable_reason="fixture has no clock anchor",
    )
    path = tmp_path / "unanchored.h5"
    write_hdf5(request, path)

    with pytest.raises(
        ClockReferenceUnavailableError,
        match="assume_scale alone is insufficient",
    ):
        IngestData(path, assume_scale="utc")

    legacy = LegacyClockReferenceSet(
        clock_reference_raw_seconds=100.0,
        mjd_epoch_offset_days=61000.0,
        time_scale="utc",
        source="legacy test fixture",
        assumed=False,
        mapping_sha256="a" * 64,
    )
    with pytest.raises(TypeError, match="production ClockReferenceSet"):
        IngestData(path, clock_reference_set=legacy)
    with pytest.raises(TypeError, match="stored clock reference"):
        IngestData(replace(load_bundle(path), clock_reference_set=legacy))

    data = IngestData(
        path,
        clock_reference_set=anchored_request.clock_reference_set,
    )
    assert data.clock_reference_set == anchored_request.clock_reference_set


def test_assume_scale_alone_cannot_authorize_legacy_unix_epoch(tmp_path: Path):
    path = tmp_path / "unanchored-v3.h5"
    write_legacy_hdf5(path, 3, complete_clock=False)

    with pytest.raises(
        ClockReferenceUnavailableError,
        match="assume_scale alone cannot authorize absolute time",
    ):
        IngestData(path, assume_scale="utc")


def test_legacy_zero_placeholder_pair_cannot_authorize_absolute_time(
    tmp_path: Path,
):
    path = tmp_path / "placeholder-v3.h5"
    write_legacy_hdf5(path, 3, complete_clock=False)
    with h5py.File(path, "r+") as handle:
        constants = handle["constants"].attrs
        constants["raw_time_subtract_seconds"] = np.float64(0.0)
        constants["mjd_epoch_offset_days"] = np.float64(0.0)
        constants["time_scale"] = "unknown"

    with pytest.raises(
        ClockReferenceUnavailableError,
        match="assume_scale alone cannot authorize absolute time",
    ):
        IngestData(path, assume_scale="utc")


def test_mission_epoch_cannot_imply_zero_raw_clock_anchor(tmp_path: Path):
    path = tmp_path / "unanchored-v3.h5"
    write_legacy_hdf5(path, 3, complete_clock=False)

    with pytest.raises(ValueError, match="unverified zero raw-clock anchor"):
        IngestData(path, mission_epoch="2025-01-01T00:00:00")


@pytest.mark.parametrize("layout_version", (2, 3))
def test_layout_v2_v3_clock_constants_remain_readable(
    tmp_path: Path,
    layout_version: int,
):
    path = tmp_path / f"legacy-v{layout_version}.h5"
    raw_times, expected_mjd = write_legacy_hdf5(path, layout_version)

    warning_type = RuntimeWarning if layout_version == 2 else LegacyIngestWarning
    with pytest.warns(warning_type) as caught:
        data = IngestData(path)
    assert any("legacy_unverified" in str(item.message) for item in caught)

    assert data.layout_version == layout_version
    np.testing.assert_array_equal(data.raw_times, raw_times)
    np.testing.assert_allclose(
        data.times.mjd,
        expected_mjd,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_array_equal(
        data.spectra,
        np.full(
            (raw_times.size, NPRODUCTS, NCHANNELS),
            2.0,
            dtype=np.float32,
        ),
    )
    assert data.time_provenance == {
        "scale": "utc",
        "source": "spectrometer",
        "assumed": False,
    }
    reference = data.clock_reference_set.require_reference("spectrometer")
    assert reference.clock_reference_raw_seconds == 100.0
    assert isinstance(data.clock_reference_set, LegacyClockReferenceSet)
    assert data.clock_reference_set.reference_event == "legacy_mjd_offset"
    assert data.clock_reference_set.verified_landing is False
    assert data.clock_reference_set.reference_for("dcb") is None


@pytest.mark.parametrize("layout_version", (2, 3))
@pytest.mark.parametrize("navgf", (2, 3, 4))
def test_layout_v2_v3_fixed_backing_width_uses_navgf_prefix(
    tmp_path: Path,
    layout_version: int,
    navgf: int,
):
    path = tmp_path / f"legacy-v{layout_version}-navgf{navgf}.h5"
    write_legacy_hdf5(path, layout_version, navgf=navgf)

    with pytest.warns(RuntimeWarning):
        data = IngestData(path)

    expected = spectrometer_frequency_window(navgf).output_count
    assert data.spectra.shape == (2, NPRODUCTS, expected)
    assert data.Nfreq == expected
    np.testing.assert_array_equal(
        data.navgf,
        np.full(2, navgf, dtype=np.uint8),
    )
    np.testing.assert_allclose(
        np.diff(data.freq),
        0.025 * spectrometer_frequency_window(navgf).stride,
        rtol=0.0,
        atol=1e-14,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        caller_bundle = load_bundle(path)
        caller_bundle.spectra_navgf = None
        caller_data = IngestData(caller_bundle)
    np.testing.assert_array_equal(caller_data.navgf, data.navgf)
    np.testing.assert_array_equal(caller_data.freq, data.freq)


@pytest.mark.parametrize("layout_version", (2, 3))
def test_layout_v2_v3_mixed_navgf_split_counts_actual_rows(
    tmp_path: Path,
    layout_version: int,
):
    path = tmp_path / f"legacy-v{layout_version}-mixed.h5"
    write_legacy_hdf5(path, layout_version)
    with h5py.File(path, "r+") as handle:
        handle["spectra/metadata/Navgf"][...] = np.asarray(
            [3, 4], dtype=np.uint8
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        groups = load_bundle(path).split_by_frequency_grid()

    assert [group.spectra.shape[0] for group in groups] == [1, 1]
    assert [
        group.family_status["persisted_rows_by_family"]["spectra"]
        for group in groups
    ] == [1, 1]


@pytest.mark.parametrize("layout_version", (2, 3))
def test_layout_v2_v3_multi_source_counts_follow_navgf(
    tmp_path: Path,
    layout_version: int,
):
    paths = (
        tmp_path / f"legacy-v{layout_version}-first.h5",
        tmp_path / f"legacy-v{layout_version}-second.h5",
    )
    for path in paths:
        write_legacy_hdf5(path, layout_version, navgf=2)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        bundle = load_bundle(paths)
        data = IngestData(bundle)

    np.testing.assert_array_equal(
        bundle.spectra_frequency_counts,
        np.full(4, 1024, dtype=np.int64),
    )
    assert data.spectra.shape == (4, NPRODUCTS, 1024)


@pytest.mark.parametrize("layout_version", (2, 3))
def test_layout_v2_v3_multi_source_rejects_undersized_source_before_padding(
    tmp_path: Path,
    layout_version: int,
):
    paths = (
        tmp_path / f"legacy-v{layout_version}-short.h5",
        tmp_path / f"legacy-v{layout_version}-full.h5",
    )
    for path in paths:
        write_legacy_hdf5(path, layout_version, navgf=2)
    with h5py.File(paths[0], "r+") as handle:
        spectra = handle["spectra"]
        attributes = dict(spectra["data"].attrs)
        del spectra["data"]
        data = spectra.create_dataset(
            "data",
            data=np.ones((2, NPRODUCTS, 512), dtype=np.float32),
        )
        for name, value in attributes.items():
            data.attrs[name] = value

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with pytest.raises(ValueError, match="source backing width 512"):
            load_bundle(paths)


@pytest.mark.parametrize("layout_version", (2, 3))
def test_layout_v2_v3_mixed_navgf_plot_uses_canonical_tag(
    tmp_path: Path,
    layout_version: int,
):
    path = tmp_path / f"legacy-v{layout_version}-mixed-plot.h5"
    write_legacy_hdf5(path, layout_version)
    with h5py.File(path, "r+") as handle:
        handle["spectra/metadata/Navgf"][...] = np.asarray(
            [3, 4], dtype=np.uint8
        )

    with pytest.warns(RuntimeWarning):
        written = plot_spectra_mean(
            path,
            tmp_path / f"legacy-v{layout_version}-mixed-plot",
            products=[0],
        )

    assert {item.name for item in written} == {
        "spectra_mean_navgf3_p00.png",
        "spectra_mean_navgf4_p00.png",
    }


@pytest.mark.parametrize("layout_version", (2, 3))
def test_layout_v2_v3_plot_uses_legacy_width_with_warning(
    tmp_path: Path,
    layout_version: int,
):
    path = tmp_path / f"legacy-plot-v{layout_version}.h5"
    write_legacy_hdf5(path, layout_version, navgf=2)

    with pytest.warns(RuntimeWarning) as caught:
        written = plot_spectra_mean(
            path,
            tmp_path / f"legacy-plot-v{layout_version}.png",
            products=[0],
        )

    assert any(
        isinstance(item.message, LegacyIngestWarning)
        and "legacy_unverified" in str(item.message)
        for item in caught
    )
    assert len(written) == 1
    assert written[0].is_file()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cube = _normal_cube(load_bundle(path))
    assert cube.shape == (2, NPRODUCTS, 1024)


def test_equivalent_caller_clock_reference_supersedes_legacy_provenance(
    tmp_path: Path,
):
    path = tmp_path / "legacy-v3.h5"
    _, expected_mjd = write_legacy_hdf5(path, 3)
    caller = ClockReferenceSet(
        format_version=1,
        reference_event="landing",
        clock_reference_isot=Time(
            61000.0,
            format="mjd",
            scale="utc",
        ).isot,
        time_scale="utc",
        clocks=(
            ClockReference(
                clock_source="spectrometer",
                clock_reference_raw_seconds=100.0,
            ),
        ),
        source="explicit caller fixture",
        assumed=False,
        source_sha256="f" * 64,
    )

    with pytest.warns(LegacyIngestWarning):
        data = IngestData(path, clock_reference_set=caller)

    assert data.clock_reference_set is caller
    np.testing.assert_allclose(
        data.times.mjd,
        expected_mjd,
        rtol=0.0,
        atol=1e-12,
    )

    contradictory = replace(
        caller,
        clocks=(
            ClockReference(
                clock_source="spectrometer",
                clock_reference_raw_seconds=101.0,
            ),
        ),
    )
    with pytest.warns(LegacyIngestWarning), pytest.raises(
        ValueError,
        match="contradicts the migrated",
    ):
        IngestData(path, clock_reference_set=contradictory)


def test_legacy_clock_reference_rejects_extreme_finite_raw_times():
    with pytest.raises(ValueError, match="MJD anchor is not representable"):
        LegacyClockReferenceSet(
            clock_reference_raw_seconds=0.0,
            mjd_epoch_offset_days=1e100,
            time_scale="utc",
            source="legacy test fixture",
            assumed=False,
            mapping_sha256="a" * 64,
        )

    overflow = LegacyClockReferenceSet(
        clock_reference_raw_seconds=-1e308,
        mjd_epoch_offset_days=61000.0,
        time_scale="utc",
        source="legacy test fixture",
        assumed=False,
        mapping_sha256="a" * 64,
    )
    with pytest.raises(ValueError, match="unrepresentable time delta"):
        overflow.to_time(1e308, clock_source="spectrometer")

    out_of_range = replace(overflow, clock_reference_raw_seconds=0.0)
    with pytest.raises(ValueError):
        out_of_range.to_time(1e300, clock_source="spectrometer")
