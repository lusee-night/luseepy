"""Validation tests for the format-neutral layout-v4 write request."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from lusee.ingest.clock_reference import (
    ClockReference,
    ClockReferenceSet,
    ClockSource,
)
from lusee.ingest.decode import CalDataSample as LegacyCalDataSample
from lusee.ingest.decode import Products
from lusee.ingest.issues import IngestIssue, IssueAction, IssueSeverity
from lusee.ingest.products import (
    DataQuality,
    DecodeProvenance,
    HKSample,
    ProductProvenance,
    SourcePacketProvenance,
    ValidatedCounts,
)
from lusee.ingest.telemetry import TelemetryData
from lusee.ingest.write_request import (
    ALL_FAMILIES,
    UNSUPPORTED_FAMILIES,
    FamilyCoverage,
    FamilyStatus,
    LunarLocation,
    RunProvenance,
    WriteRequest,
    family_statuses_for_products,
)


def make_decode_provenance() -> DecodeProvenance:
    return DecodeProvenance.from_report(
        distribution_version="1.2.3",
        decoder_source_commit="a" * 40,
        reported_schema_ids=(0x307,),
        selected_schema_id=0x307,
        binding_key="307",
        schema_variant=None,
        schema_assumed=False,
        binding_source_release="3r09",
        binding_source_commit="b" * 40,
        abi_fingerprint="c" * 64,
        execution_mode="collect",
        input_packet_count=1,
        valid_packet_count=1,
        appid_counts=((0x212, 1),),
        issue_counts=(),
        canonical_report={"fixture": "strict"},
    )


def make_housekeeping() -> HKSample:
    provenance = ProductProvenance(
        source_packets=(
            SourcePacketProvenance(
                role="housekeeping",
                original_appid=0x212,
                packet_index=0,
            ),
        ),
        uid=7,
        uid_source="housekeeping.unique_packet_id",
        uid_source_role="housekeeping",
        reported_schema_ids=(0x307,),
        selected_schema_id=0x307,
        time_valid=False,
    )
    return HKSample(
        hk_type=1,
        version=0x307,
        unique_packet_id=7,
        errors=0,
        raw_seconds=None,
        fields={"temperature": 21.5},
        field_present={"temperature": True},
        provenance=provenance,
    )


def make_products() -> Products:
    return Products(
        housekeeping=[make_housekeeping()],
        decode_provenance=make_decode_provenance(),
        quality_status=DataQuality.CLEAN,
        validated_counts=ValidatedCounts(
            input_packets=1,
            valid_packets=1,
            product_rows=(("housekeeping", 1),),
        ),
        issues=(),
    )


def make_run_provenance() -> RunProvenance:
    return RunProvenance(
        input_identity="fixture-sha256",
        input_identity_kind="sha256",
        input_identity_unavailable_reason=None,
        source_kind="uncrater_session",
    )


def request_values(products: Products) -> dict[str, object]:
    return {
        "products": products,
        "clock_reference_set": None,
        "clock_reference_unavailable_reason": "fixture has no clock anchor",
        "location": LunarLocation(-23.814, 182.258, 0.0),
        "run_provenance": make_run_provenance(),
        "issues": products.issues,
        "family_statuses": family_statuses_for_products(
            products,
            family_issue_ids={},
        ),
    }


def make_clock_reference_set() -> ClockReferenceSet:
    return ClockReferenceSet(
        format_version=1,
        reference_event="landing",
        clock_reference_isot="2026-01-01T00:00:00",
        time_scale="utc",
        clocks=(),
        source="test fixture",
        assumed=False,
        source_sha256="d" * 64,
    )


def make_telemetry(*, mjd_times=None) -> TelemetryData:
    if mjd_times is None:
        mjd_times = np.array([np.nan], dtype=np.float64)
    return TelemetryData(
        source_kind="b01_0x314",
        field_names=tuple(f"field_{index}" for index in range(57)),
        units=("V",) * 57,
        source_indices=np.array([3], dtype=np.int64),
        mission_seconds=np.array([10], dtype=np.uint32),
        lusee_subsecs=np.array([32768], dtype=np.uint16),
        mjd_times=mjd_times,
        raw_counts=np.zeros((1, 57), dtype=np.uint16),
        values=np.zeros((1, 57), dtype=np.float64),
        valid=np.ones((1, 57), dtype=np.bool_),
    )


def make_issue() -> IngestIssue:
    return IngestIssue(
        issue_id="issue-0001",
        code="fixture_issue",
        severity=IssueSeverity.WARNING,
        stage="decode",
        message="fixture issue",
        action=IssueAction.DROPPED,
    )


def test_write_request_retains_early_context_issues():
    products = make_products()
    issue = IngestIssue(
        issue_id="issue-early-0001",
        code="framing.synthetic_damage",
        severity=IssueSeverity.WARNING,
        stage="framing",
        message="one source frame was dropped",
        action=IssueAction.DROPPED,
    )
    values = request_values(products)
    values["issues"] = (issue,)
    values["context_issues"] = (issue,)

    request = WriteRequest(**values)

    assert request.issues == (issue,)
    assert request.context_issues == (issue,)
    assert request.quality_status is DataQuality.PARTIAL


def test_write_request_requires_context_issues_in_exact_union():
    products = make_products()
    issue = IngestIssue(
        issue_id="issue-early-0001",
        code="framing.synthetic_damage",
        severity=IssueSeverity.WARNING,
        stage="framing",
        message="one source frame was dropped",
        action=IssueAction.DROPPED,
    )
    values = request_values(products)
    values["issues"] = (issue,)

    with pytest.raises(ValueError, match="exact source/context union"):
        WriteRequest(**values)


def test_write_request_preserves_existing_positional_constructor_order():
    products = make_products()
    values = request_values(products)
    request = WriteRequest(
        products,
        None,
        values["clock_reference_unavailable_reason"],
        values["location"],
        values["run_provenance"],
        values["issues"],
        values["family_statuses"],
        None,
        False,
        "gzip",
        1,
    )

    assert request.context_issues == ()


def test_minimal_strict_products_make_valid_write_request():
    products = make_products()

    request = WriteRequest(**request_values(products))

    assert request.products is products
    assert request.issues == ()
    assert request.clock_reference_set is None
    assert request.clock_reference_unavailable_reason == ("fixture has no clock anchor")
    assert tuple(status.family for status in request.family_statuses) == tuple(
        sorted(ALL_FAMILIES)
    )


def test_write_request_requires_complete_sorted_family_statuses():
    products = make_products()
    values = request_values(products)
    values["family_statuses"] = values["family_statuses"][:-1]

    with pytest.raises(ValueError, match="every known family in sorted order"):
        WriteRequest(**values)


def test_invalid_or_dropped_family_requires_issue_reference():
    with pytest.raises(ValueError, match="requires an issue reference"):
        FamilyStatus(
            family="spectra",
            supported=True,
            coverage=FamilyCoverage.INVALID_OR_DROPPED,
            quality=DataQuality.FAILED,
            decoded_rows=0,
        )


def test_clean_family_rejects_issue_reference():
    with pytest.raises(ValueError, match="clean family cannot reference issues"):
        FamilyStatus(
            family="spectra",
            supported=True,
            coverage=FamilyCoverage.ABSENT_IN_INPUT,
            quality=DataQuality.CLEAN,
            decoded_rows=0,
            issue_ids=("issue-0001",),
        )


def test_partial_family_requires_issue_reference():
    with pytest.raises(ValueError, match="partial family requires an issue"):
        FamilyStatus(
            family="housekeeping",
            supported=True,
            coverage=FamilyCoverage.PERSISTED,
            quality=DataQuality.PARTIAL,
            decoded_rows=1,
        )


def test_family_status_helper_classifies_every_coverage_state():
    products = make_products()

    statuses = family_statuses_for_products(
        products,
        family_issue_ids={"spectra": ("issue-0001",)},
    )
    status_by_family = {status.family: status for status in statuses}

    assert tuple(status_by_family) == tuple(sorted(ALL_FAMILIES))
    assert status_by_family["housekeeping"].coverage is FamilyCoverage.PERSISTED
    assert status_by_family["housekeeping"].quality is DataQuality.CLEAN
    assert status_by_family["housekeeping"].decoded_rows == 1
    assert status_by_family["spectra"].coverage is FamilyCoverage.INVALID_OR_DROPPED
    assert status_by_family["spectra"].quality is DataQuality.FAILED
    assert status_by_family["spectra"].issue_ids == ("issue-0001",)
    assert status_by_family["tr_spectra"].coverage is (FamilyCoverage.ABSENT_IN_INPUT)
    for family in UNSUPPORTED_FAMILIES:
        assert status_by_family[family].coverage is FamilyCoverage.UNSUPPORTED
        assert status_by_family[family].supported is False


def test_write_request_rejects_legacy_cal_data():
    products = make_products()
    products.cal_data.append(
        LegacyCalDataSample(
            packet_idx=0,
            channel_idx=0,
            data=np.array([1.0], dtype=np.float32),
        )
    )

    with pytest.raises(ValueError, match="anonymous cal_data"):
        WriteRequest(**request_values(products))


def test_write_request_rejects_partial_hello_invariants():
    products = make_products()
    products.sw_version = 0x307

    with pytest.raises(ValueError, match="all present or all absent"):
        WriteRequest(**request_values(products))


def test_write_request_rejects_inconsistent_hello_split_time():
    products = make_products()
    products.sw_version = 0x307
    products.fw_version = 1
    products.fw_id = 2
    products.fw_date = 3
    products.fw_time = 4
    products.start_unique_packet_id = 5
    products.start_time_32 = 65536
    products.start_time_16 = 0
    products.start_raw_seconds = 2.0

    with pytest.raises(ValueError, match="disagrees with Hello split time"):
        WriteRequest(**request_values(products))


def test_write_request_rejects_issue_integer_overflow():
    products = make_products()
    values = request_values(products)
    values["issues"] = (replace(make_issue(), appid=0x800),)

    with pytest.raises(ValueError, match="unsigned 11-bit"):
        WriteRequest(**values)


def test_write_request_rejects_untyped_telemetry():
    products = make_products()
    values = request_values(products)
    values["telemetry"] = {"raw_seconds": np.array([1.0], dtype=np.float64)}

    with pytest.raises(TypeError, match="TelemetryData or None"):
        WriteRequest(**values)


def test_write_request_rejects_issue_mismatch():
    products = make_products()
    values = request_values(products)
    values["issues"] = (make_issue(),)

    with pytest.raises(ValueError, match="exact source/context union"):
        WriteRequest(**values)


def test_telemetry_does_not_join_issue_union_or_family_statuses():
    products = make_products()
    values = request_values(products)
    values["telemetry"] = make_telemetry()

    request = WriteRequest(**values)

    assert request.issues == ()
    assert request.quality_status is DataQuality.CLEAN
    assert "dcb_telemetry" not in {
        status.family for status in request.family_statuses
    }


def test_partial_quality_requires_recorded_issue():
    products = make_products()
    products.quality_status = DataQuality.PARTIAL

    with pytest.raises(ValueError, match="partial product quality"):
        WriteRequest(**request_values(products))


def test_clean_quality_rejects_degraded_supported_family():
    products = make_products()
    issue = make_issue()
    products.issues = (issue,)
    values = request_values(products)
    values["family_statuses"] = family_statuses_for_products(
        products,
        family_issue_ids={"spectra": (issue.issue_id,)},
    )

    with pytest.raises(ValueError, match="clean product quality"):
        WriteRequest(**values)


def test_write_request_rejects_unknown_family_issue_reference():
    products = make_products()
    values = request_values(products)
    values["family_statuses"] = family_statuses_for_products(
        products,
        family_issue_ids={"spectra": ("issue-unknown",)},
    )

    with pytest.raises(ValueError, match="references an unknown issue"):
        WriteRequest(**values)


def test_write_request_rejects_finite_telemetry_time_without_dcb_reference():
    products = make_products()
    values = request_values(products)
    values["telemetry"] = make_telemetry(
        mjd_times=np.array([60000.0], dtype=np.float64)
    )

    with pytest.raises(ValueError, match="requires a DCB clock reference"):
        WriteRequest(**values)


def test_write_request_requires_exact_dcb_telemetry_time():
    products = make_products()
    reference_set = ClockReferenceSet(
        format_version=1,
        reference_event="landing",
        clock_reference_isot="2026-01-01T00:00:00",
        time_scale="utc",
        clocks=(ClockReference(ClockSource.DCB, 10.0),),
        source="test fixture",
        assumed=False,
        source_sha256="e" * 64,
    )
    raw_seconds = np.array([10.5], dtype=np.float64)
    expected = np.asarray(
        reference_set.to_mjd(raw_seconds, clock_source=ClockSource.DCB),
        dtype=np.float64,
    )
    values = request_values(products)
    values["clock_reference_set"] = reference_set
    values["clock_reference_unavailable_reason"] = None
    values["telemetry"] = make_telemetry(mjd_times=expected)

    assert WriteRequest(**values).telemetry is values["telemetry"]

    values["telemetry"] = make_telemetry(mjd_times=expected + 1.0)
    with pytest.raises(ValueError, match="contradict the DCB clock reference"):
        WriteRequest(**values)


def test_write_request_requires_row_issues_in_family_status():
    issue = make_issue()
    row = make_housekeeping()
    row = replace(
        row,
        provenance=replace(
            row.provenance,
            decoder_issue_ids=(issue.issue_id,),
        ),
    )
    products = make_products()
    products.housekeeping = [row]
    products.issues = (issue,)
    products.quality_status = DataQuality.PARTIAL
    values = request_values(products)

    with pytest.raises(ValueError, match="omits row issue references"):
        WriteRequest(**values)


@pytest.mark.parametrize("record_both", [False, True])
def test_write_request_rejects_ambiguous_clock_reference(record_both: bool):
    products = make_products()
    values = request_values(products)
    values["clock_reference_set"] = make_clock_reference_set() if record_both else None
    values["clock_reference_unavailable_reason"] = (
        "also unavailable" if record_both else None
    )

    with pytest.raises(ValueError, match="record exactly one"):
        WriteRequest(**values)


@pytest.mark.parametrize("record_both", [False, True])
def test_run_provenance_rejects_ambiguous_input_identity(record_both: bool):
    with pytest.raises(ValueError, match="record exactly one"):
        RunProvenance(
            input_identity="fixture" if record_both else None,
            input_identity_kind="sha256" if record_both else None,
            input_identity_unavailable_reason=(
                "also unavailable" if record_both else None
            ),
            source_kind="uncrater_session",
        )
