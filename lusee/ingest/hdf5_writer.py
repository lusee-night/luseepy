"""HDF5 writer for the validated ingest layout v4 contract."""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import fields as dataclass_fields
from pathlib import Path

import numpy as np

from .clock_reference import ClockSource
from .constants import (
    BITSLICE_REFERENCE,
    HDF5_LAYOUT_VERSION,
    NCHANNELS,
    NPRODUCTS,
    SPECTRA_NORMALIZATION_VERSION,
    WAVEFORM_SAMPLES,
)
from .dependencies import import_optional_dependency
from .frequency_contract import FrequencyWindowContract
from .write_request import FAMILY_TYPES, WriteRequest

log = logging.getLogger(__name__)


def _dataset_kwargs(request: WriteRequest, data: np.ndarray) -> dict[str, object]:
    if data.ndim == 0 or data.size == 0 or request.hdf5_compression is None:
        return {}
    return {
        "compression": request.hdf5_compression,
        "compression_opts": request.hdf5_compression_level,
    }


def _create_dataset(
    group,
    name: str,
    data: object,
    request: WriteRequest,
    *,
    dtype: object | None = None,
):
    array = np.asarray(data if dtype is None else np.asarray(data, dtype=dtype))
    return group.create_dataset(
        name,
        data=data,
        dtype=dtype,
        **_dataset_kwargs(request, array),
    )


def _write_strings(
    group,
    name: str,
    values: Sequence[str],
    request: WriteRequest,
    h5py,
):
    data = np.asarray(tuple(values), dtype=object)
    return _create_dataset(
        group,
        name,
        data,
        request,
        dtype=h5py.string_dtype(encoding="utf-8"),
    )


def _write_optional_attr(group, name: str, value: object | None) -> None:
    group.attrs[f"{name}_valid"] = np.bool_(value is not None)
    if value is not None:
        group.attrs[name] = value


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _write_session_invariants(h5, request: WriteRequest) -> None:
    products = request.products
    group = h5.create_group("session_invariants")
    fields = {
        "software_version": (
            None if products.sw_version is None else np.uint32(products.sw_version)
        ),
        "firmware_version": (
            None if products.fw_version is None else np.uint32(products.fw_version)
        ),
        "firmware_id": (None if products.fw_id is None else np.uint32(products.fw_id)),
        "firmware_date": (
            None if products.fw_date is None else np.uint32(products.fw_date)
        ),
        "firmware_time": (
            None if products.fw_time is None else np.uint32(products.fw_time)
        ),
        "start_unique_packet_id": (
            None
            if products.start_unique_packet_id is None
            else np.uint32(products.start_unique_packet_id)
        ),
        "start_time_32": (
            None
            if products.start_time_32 is None
            else np.uint32(products.start_time_32)
        ),
        "start_time_16": (
            None
            if products.start_time_16 is None
            else np.uint16(products.start_time_16)
        ),
        "start_raw_seconds": (
            None
            if products.start_raw_seconds is None
            else np.float64(products.start_raw_seconds)
        ),
    }
    for name, value in fields.items():
        _write_optional_attr(group, name, value)


def _write_constants(h5, request: WriteRequest) -> None:
    group = h5.create_group("constants")
    group.attrs["lun_lat_deg"] = np.float64(request.location.latitude_deg)
    group.attrs["lun_long_deg"] = np.float64(request.location.longitude_deg)
    group.attrs["lun_height_m"] = np.float64(request.location.height_m)


def _write_clock_reference(h5, request: WriteRequest, h5py) -> None:
    group = h5.create_group("clock_reference")
    reference_set = request.clock_reference_set
    group.attrs["available"] = np.bool_(reference_set is not None)
    if reference_set is None:
        group.attrs["unavailable_reason"] = request.clock_reference_unavailable_reason
        return
    group.attrs["format_version"] = np.uint16(reference_set.format_version)
    group.attrs["reference_event"] = reference_set.reference_event
    group.attrs["clock_reference_isot"] = reference_set.clock_reference_isot
    group.attrs["time_scale"] = reference_set.time_scale
    group.attrs["source"] = reference_set.source
    group.attrs["assumed"] = np.bool_(reference_set.assumed)
    group.attrs["source_sha256"] = reference_set.source_sha256
    _write_strings(
        group,
        "clock_sources",
        [item.clock_source.value for item in reference_set.clocks],
        request,
        h5py,
    )
    _create_dataset(
        group,
        "clock_reference_raw_seconds",
        np.asarray(
            [item.clock_reference_raw_seconds for item in reference_set.clocks],
            dtype=np.float64,
        ),
        request,
    )
    group.attrs["canonical_record_json"] = _canonical_json(reference_set.as_record())


def _write_run_provenance(
    h5,
    request: WriteRequest,
    *,
    destination_preexisted: bool,
) -> None:
    group = h5.create_group("run_provenance")
    for name, value in request.run_provenance.as_record().items():
        _write_optional_attr(group, name, value)
    group.attrs["overwrite_requested"] = np.bool_(request.overwrite)
    group.attrs["destination_preexisted"] = np.bool_(destination_preexisted)
    _write_optional_attr(group, "hdf5_compression", request.hdf5_compression)
    _write_optional_attr(
        group,
        "hdf5_compression_level",
        request.hdf5_compression_level,
    )


def _write_decoder_provenance(h5, request: WriteRequest, h5py) -> None:
    provenance = request.products.decode_provenance
    group = h5.require_group("provenance").create_group("decoder")
    for name in (
        "decoder_name",
        "distribution_version",
        "decoder_source_commit",
        "binding_key",
        "schema_variant",
        "binding_source_release",
        "binding_source_commit",
        "abi_fingerprint",
        "canonical_report_json",
    ):
        _write_optional_attr(group, name, getattr(provenance, name))
    group.attrs["selected_schema_id"] = np.uint16(provenance.selected_schema_id)
    group.attrs["schema_assumed"] = np.bool_(provenance.schema_assumed)
    group.attrs["execution_mode"] = provenance.execution_mode.value
    group.attrs["input_packet_count"] = np.uint64(provenance.input_packet_count)
    group.attrs["valid_packet_count"] = np.uint64(provenance.valid_packet_count)
    _create_dataset(
        group,
        "reported_schema_ids",
        np.asarray(provenance.reported_schema_ids, dtype=np.uint16),
        request,
    )
    _create_dataset(
        group,
        "appids",
        np.asarray([item[0] for item in provenance.appid_counts], dtype=np.uint16),
        request,
    )
    _create_dataset(
        group,
        "appid_counts",
        np.asarray([item[1] for item in provenance.appid_counts], dtype=np.uint64),
        request,
    )
    _write_strings(
        group,
        "issue_codes",
        [item[0] for item in provenance.issue_counts],
        request,
        h5py,
    )
    _create_dataset(
        group,
        "issue_code_counts",
        np.asarray([item[1] for item in provenance.issue_counts], dtype=np.uint64),
        request,
    )


def _optional_integer_column(
    values: Sequence[int | None], dtype
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.asarray([value is not None for value in values], dtype=np.bool_)
    data = np.zeros(len(values), dtype=dtype)
    for index, value in enumerate(values):
        if value is not None:
            data[index] = value
    return data, valid


def _optional_string_column(
    values: Sequence[str | None],
) -> tuple[list[str], np.ndarray]:
    valid = np.asarray([value is not None for value in values], dtype=np.bool_)
    return ["" if value is None else value for value in values], valid


def _write_issues(h5, request: WriteRequest, h5py) -> dict[str, int]:
    group = h5.create_group("issues")
    issues = request.issues
    group.attrs["count"] = np.uint64(len(issues))
    issue_index = {issue.issue_id: index for index, issue in enumerate(issues)}
    string_fields = (
        "issue_id",
        "code",
        "stage",
        "message",
        "input_identity",
        "bank",
        "session",
    )
    for name in string_fields:
        values = [getattr(issue, name) for issue in issues]
        if name in ("issue_id", "code", "stage", "message"):
            _write_strings(group, name, values, request, h5py)
        else:
            data, valid = _optional_string_column(values)
            _write_strings(group, name, data, request, h5py)
            _create_dataset(group, f"{name}_valid", valid, request)
    _write_strings(
        group,
        "severity",
        [issue.severity.value for issue in issues],
        request,
        h5py,
    )
    _write_strings(
        group,
        "action",
        [issue.action.value for issue in issues],
        request,
        h5py,
    )
    integer_fields = (
        ("byte_offset", np.uint64),
        ("frame_index", np.uint64),
        ("packet_index", np.uint64),
        ("appid", np.uint16),
        ("sequence_count", np.uint16),
        ("uid", np.uint32),
    )
    for name, dtype in integer_fields:
        data, valid = _optional_integer_column(
            [getattr(issue, name) for issue in issues], dtype
        )
        _create_dataset(group, name, data, request)
        _create_dataset(group, f"{name}_valid", valid, request)
    _write_strings(
        group,
        "details_json",
        [_canonical_json(issue.as_dict()["details"]) for issue in issues],
        request,
        h5py,
    )
    return issue_index


def _iter_product_rows(request: WriteRequest):
    for family, _ in FAMILY_TYPES:
        for row_index, row in enumerate(getattr(request.products, family)):
            yield family, row_index, row


def _write_product_provenance(
    h5,
    request: WriteRequest,
    h5py,
    issue_index: Mapping[str, int],
) -> dict[tuple[str, int], int]:
    root = h5.require_group("provenance")
    group = root.create_group("product_rows")
    entries = list(_iter_product_rows(request))
    provenance_index = {
        (family, row_index): index
        for index, (family, row_index, _) in enumerate(entries)
    }
    _write_strings(
        group,
        "family",
        [item[0] for item in entries],
        request,
        h5py,
    )
    _create_dataset(
        group,
        "row_index",
        np.asarray([item[1] for item in entries], dtype=np.uint64),
        request,
    )
    _create_dataset(
        group,
        "unique_ids",
        np.asarray([item[2].unique_packet_id for item in entries], dtype=np.uint32),
        request,
    )
    provenances = [item[2].provenance for item in entries]
    _write_strings(
        group,
        "uid_source",
        [provenance.uid_source for provenance in provenances],
        request,
        h5py,
    )
    for name in (
        "uid_source_role",
        "time_source",
        "time_source_role",
        "clock_source",
    ):
        values, valid = _optional_string_column(
            [getattr(provenance, name) for provenance in provenances]
        )
        _write_strings(group, name, values, request, h5py)
        _create_dataset(group, f"{name}_valid", valid, request)
    _create_dataset(
        group,
        "time_valid",
        np.asarray([item.time_valid for item in provenances], dtype=np.bool_),
        request,
    )
    _create_dataset(
        group,
        "selected_schema_ids",
        np.asarray(
            [item.selected_schema_id for item in provenances],
            dtype=np.uint16,
        ),
        request,
    )

    schema_group = root.create_group("product_schema_refs")
    schema_rows = [
        (index, schema_id)
        for index, provenance in enumerate(provenances)
        for schema_id in provenance.reported_schema_ids
    ]
    _create_dataset(
        schema_group,
        "provenance_index",
        np.asarray([item[0] for item in schema_rows], dtype=np.uint64),
        request,
    )
    _create_dataset(
        schema_group,
        "schema_id",
        np.asarray([item[1] for item in schema_rows], dtype=np.uint16),
        request,
    )

    packet_group = root.create_group("source_packets")
    packet_rows = [
        (provenance_ndx, order, packet)
        for provenance_ndx, provenance in enumerate(provenances)
        for order, packet in enumerate(provenance.source_packets)
    ]
    _create_dataset(
        packet_group,
        "provenance_index",
        np.asarray([item[0] for item in packet_rows], dtype=np.uint64),
        request,
    )
    _create_dataset(
        packet_group,
        "source_order",
        np.asarray([item[1] for item in packet_rows], dtype=np.uint16),
        request,
    )
    _write_strings(
        packet_group,
        "role",
        [item[2].role for item in packet_rows],
        request,
        h5py,
    )
    _create_dataset(
        packet_group,
        "original_appid",
        np.asarray([item[2].original_appid for item in packet_rows], dtype=np.uint16),
        request,
    )
    optional_packet_fields = (
        ("normalized_appid", np.uint16),
        ("packet_index", np.uint64),
        ("frame_start", np.uint64),
        ("frame_stop", np.uint64),
        ("byte_offset_start", np.uint64),
        ("byte_offset_stop", np.uint64),
    )
    for name, dtype in optional_packet_fields:
        data, valid = _optional_integer_column(
            [getattr(item[2], name) for item in packet_rows], dtype
        )
        _create_dataset(packet_group, name, data, request)
        _create_dataset(packet_group, f"{name}_valid", valid, request)
    for name in ("filename", "bank"):
        data, valid = _optional_string_column(
            [getattr(item[2], name) for item in packet_rows]
        )
        _write_strings(packet_group, name, data, request, h5py)
        _create_dataset(packet_group, f"{name}_valid", valid, request)

    issue_group = root.create_group("product_issue_refs")
    issue_rows = [
        (provenance_ndx, issue_index[issue_id])
        for provenance_ndx, provenance in enumerate(provenances)
        for issue_id in provenance.decoder_issue_ids
    ]
    _create_dataset(
        issue_group,
        "provenance_index",
        np.asarray([item[0] for item in issue_rows], dtype=np.uint64),
        request,
    )
    _create_dataset(
        issue_group,
        "issue_index",
        np.asarray([item[1] for item in issue_rows], dtype=np.uint64),
        request,
    )
    return provenance_index


def _write_family_status(
    h5,
    request: WriteRequest,
    h5py,
    issue_index: Mapping[str, int],
) -> None:
    group = h5.create_group("status").create_group("families")
    statuses = request.family_statuses
    _write_strings(
        group,
        "family",
        [status.family for status in statuses],
        request,
        h5py,
    )
    _create_dataset(
        group,
        "supported",
        np.asarray([status.supported for status in statuses], dtype=np.bool_),
        request,
    )
    _write_strings(
        group,
        "coverage",
        [status.coverage.value for status in statuses],
        request,
        h5py,
    )
    _write_strings(
        group,
        "quality",
        [status.quality.value for status in statuses],
        request,
        h5py,
    )
    _create_dataset(
        group,
        "decoded_rows",
        np.asarray([status.decoded_rows for status in statuses], dtype=np.uint64),
        request,
    )
    _create_dataset(
        group,
        "persisted_rows",
        np.asarray(
            [
                status.decoded_rows if status.coverage.value == "persisted" else 0
                for status in statuses
            ],
            dtype=np.uint64,
        ),
        request,
    )
    reasons, reason_valid = _optional_string_column(
        [status.reason for status in statuses]
    )
    _write_strings(group, "reason", reasons, request, h5py)
    _create_dataset(group, "reason_valid", reason_valid, request)
    ref_group = h5["status"].create_group("family_issue_refs")
    refs = [
        (family_index, issue_index[issue_id])
        for family_index, status in enumerate(statuses)
        for issue_id in status.issue_ids
    ]
    _create_dataset(
        ref_group,
        "family_index",
        np.asarray([item[0] for item in refs], dtype=np.uint64),
        request,
    )
    _create_dataset(
        ref_group,
        "issue_index",
        np.asarray([item[1] for item in refs], dtype=np.uint64),
        request,
    )


def _mjd_values(
    raw_seconds: np.ndarray,
    raw_valid: np.ndarray,
    clock_sources: Sequence[str | None],
    request: WriteRequest,
) -> tuple[np.ndarray, np.ndarray]:
    mjd = np.full(raw_seconds.shape, np.nan, dtype=np.float64)
    valid = np.zeros(raw_seconds.shape, dtype=np.bool_)
    reference_set = request.clock_reference_set
    if reference_set is None:
        return mjd, valid
    for source in sorted({item for item in clock_sources if item is not None}):
        reference = reference_set.reference_for(source)
        if reference is None:
            continue
        select = raw_valid & np.asarray(
            [item == source for item in clock_sources], dtype=np.bool_
        )
        if not np.any(select):
            continue
        mjd[select] = reference_set.to_mjd(raw_seconds[select], clock_source=source)
        valid[select] = True
    return mjd, valid


def _write_common_rows(
    group,
    family: str,
    rows: Sequence[object],
    request: WriteRequest,
    provenance_index: Mapping[tuple[str, int], int],
) -> None:
    count = len(rows)
    group.attrs["count"] = np.uint64(count)
    _create_dataset(
        group,
        "unique_ids",
        np.asarray([row.unique_packet_id for row in rows], dtype=np.uint32),
        request,
    )
    raw_valid = np.asarray(
        [row.raw_seconds is not None for row in rows], dtype=np.bool_
    )
    raw_seconds = np.asarray(
        [np.nan if row.raw_seconds is None else row.raw_seconds for row in rows],
        dtype=np.float64,
    )
    _create_dataset(group, "raw_seconds", raw_seconds, request)
    _create_dataset(group, "raw_time_valid", raw_valid, request)
    clock_sources = [row.provenance.clock_source for row in rows]
    mjd, mjd_valid = _mjd_values(raw_seconds, raw_valid, clock_sources, request)
    _create_dataset(group, "mjd_times", mjd, request)
    _create_dataset(group, "mjd_time_valid", mjd_valid, request)
    _create_dataset(
        group,
        "original_indices",
        np.arange(count, dtype=np.uint64),
        request,
    )
    _create_dataset(
        group,
        "provenance_index",
        np.asarray(
            [provenance_index[(family, index)] for index in range(count)],
            dtype=np.uint64,
        ),
        request,
    )


def _field_array(value: object) -> np.ndarray:
    if isinstance(value, tuple):
        value = np.asarray(value)
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise TypeError("normalized field values must not produce object arrays")
    return array


def _write_array_variant(
    group,
    name: str,
    arrays: Sequence[np.ndarray],
    request: WriteRequest,
    h5py,
) -> None:
    stacked = np.stack(arrays)
    if stacked.dtype.kind == "U":
        _create_dataset(
            group,
            name,
            stacked.astype(object),
            request,
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
    else:
        _create_dataset(group, name, stacked, request)


def _write_mapping_union(
    parent,
    values: Sequence[Mapping[str, object] | None],
    present: np.ndarray,
    request: WriteRequest,
    h5py,
) -> None:
    keys = sorted({key for value in values if value is not None for key in value})
    fields_group = parent.create_group("fields")
    presence_group = parent.create_group("field_present")
    for key in keys:
        field_values = [None if value is None else value.get(key) for value in values]
        field_present = np.asarray(
            [
                bool(present[index])
                and values[index] is not None
                and key in values[index]
                and field_values[index] is not None
                for index in range(len(values))
            ],
            dtype=np.bool_,
        )
        _create_dataset(presence_group, key, field_present, request)
        _write_one_field(
            fields_group,
            key,
            field_values,
            field_present,
            request,
            h5py,
        )


def _write_one_field(
    parent,
    name: str,
    values: Sequence[object | None],
    present: np.ndarray,
    request: WriteRequest,
    h5py,
) -> None:
    group = parent.create_group(name)
    present_values = [values[index] for index in np.flatnonzero(present)]
    if not present_values:
        group.attrs["kind"] = "untyped_absent"
        return
    if all(isinstance(value, Mapping) for value in present_values):
        group.attrs["kind"] = "mapping"
        _write_mapping_union(
            group,
            [value if isinstance(value, Mapping) else None for value in values],
            present,
            request,
            h5py,
        )
        return
    if any(isinstance(value, Mapping) for value in present_values):
        raise TypeError(f"field {name} mixes mappings and arrays")

    arrays = {index: _field_array(values[index]) for index in np.flatnonzero(present)}
    variants: dict[tuple[str, tuple[int, ...]], list[int]] = {}
    for index, array in arrays.items():
        variants.setdefault((array.dtype.str, array.shape), []).append(index)
    group.attrs["kind"] = "array_variants"
    group.attrs["variant_count"] = np.uint32(len(variants))
    variant_index = np.full(len(values), -1, dtype=np.int32)
    for number, key in enumerate(sorted(variants, key=lambda item: repr(item))):
        row_indices = variants[key]
        variant_index[row_indices] = number
        variant_group = group.create_group(f"variant_{number:03d}")
        variant_group.attrs["numpy_dtype"] = key[0]
        variant_group.attrs["value_shape_json"] = _canonical_json(list(key[1]))
        _create_dataset(
            variant_group,
            "row_indices",
            np.asarray(row_indices, dtype=np.uint64),
            request,
        )
        _write_array_variant(
            variant_group,
            "data",
            [arrays[index] for index in row_indices],
            request,
            h5py,
        )
    _create_dataset(group, "variant_index", variant_index, request)


def _write_field_union(
    parent,
    rows: Sequence[Mapping[str, object]],
    presence: Sequence[Mapping[str, bool]],
    request: WriteRequest,
    h5py,
) -> None:
    keys = sorted({key for row in rows for key in row})
    fields_group = parent.create_group("fields")
    presence_group = parent.create_group("field_present")
    for key in keys:
        values = [row.get(key) for row in rows]
        present = np.asarray(
            [bool(mask.get(key, False)) for mask in presence],
            dtype=np.bool_,
        )
        for index, is_present in enumerate(present):
            if is_present != (values[index] is not None):
                raise ValueError(
                    f"field union presence disagrees for {key!r} row {index}"
                )
        _create_dataset(presence_group, key, present, request)
        _write_one_field(fields_group, key, values, present, request, h5py)


def _write_metadata_union(parent, rows, request: WriteRequest, h5py) -> None:
    mappings = []
    presence = []
    for row in rows:
        metadata = row.metadata
        mapping = {
            item.name: getattr(metadata, item.name)
            for item in dataclass_fields(metadata)
            if item.init
        }
        mapping["adc_statistics_valid"] = metadata.adc_statistics_valid
        mapping["current_fields_present"] = metadata.current_fields_present
        mappings.append(mapping)
        presence.append({key: value is not None for key, value in mapping.items()})
    _write_field_union(
        parent,
        mappings,
        presence,
        request,
        h5py,
    )


def _write_frequency_windows(group, rows, request: WriteRequest) -> None:
    contracts: dict[int, FrequencyWindowContract] = {
        row.navgf: row.frequency_contract for row in rows
    }
    ordered = [contracts[navgf] for navgf in sorted(contracts)]
    contract_group = group.create_group("frequency_windows")
    if ordered:
        first = ordered[0]
        contract_group.attrs["contract_name"] = first.contract_name
        contract_group.attrs["contract_version"] = np.uint16(first.contract_version)
        contract_group.attrs["source_commit"] = first.source_commit
        contract_group.attrs["frequency_coordinate_status"] = (
            first.frequency_coordinate_status
        )
        contract_group.attrs["integer_arithmetic"] = first.as_record()[
            "integer_arithmetic"
        ]
    _create_dataset(
        contract_group,
        "navgf",
        np.asarray([item.navgf for item in ordered], dtype=np.uint8),
        request,
    )
    for name, dtype in (
        ("native_count", np.uint16),
        ("output_count", np.uint16),
        ("stride", np.uint16),
        ("divisor", np.uint16),
    ):
        _create_dataset(
            contract_group,
            name,
            np.asarray([getattr(item, name) for item in ordered], dtype=dtype),
            request,
        )
    offsets = np.zeros((len(ordered), 4), dtype=np.uint8)
    offset_valid = np.zeros((len(ordered), 4), dtype=np.bool_)
    weights = np.full((len(ordered), 4), np.nan, dtype=np.float64)
    for index, item in enumerate(ordered):
        count = len(item.included_offsets)
        offsets[index, :count] = item.included_offsets
        offset_valid[index, :count] = True
        weights[index, :count] = item.nominal_response_weights
    _create_dataset(contract_group, "included_offsets", offsets, request)
    _create_dataset(contract_group, "included_offset_valid", offset_valid, request)
    _create_dataset(contract_group, "nominal_response_weights", weights, request)
    window_index = {item.navgf: index for index, item in enumerate(ordered)}
    _create_dataset(
        group,
        "frequency_window_index",
        np.asarray([window_index[row.navgf] for row in rows], dtype=np.uint8),
        request,
    )


def _write_spectra(h5, request, provenance_index, h5py) -> None:
    rows = request.products.spectra
    if not rows:
        return
    group = h5.create_group("spectra")
    _write_common_rows(group, "spectra", rows, request, provenance_index)
    data = np.full((len(rows), NPRODUCTS, NCHANNELS), np.nan, dtype=np.float32)
    for index, row in enumerate(rows):
        data[index, :, : row.frequency_contract.output_count] = row.data
    dataset = _create_dataset(group, "data", data, request)
    dataset.attrs["units"] = rows[0].units
    dataset.attrs["representation"] = rows[0].representation
    dataset.attrs["bitslice_restored"] = np.bool_(True)
    dataset.attrs["bitslice_reference"] = np.uint8(BITSLICE_REFERENCE)
    dataset.attrs["normalization_version"] = np.uint16(SPECTRA_NORMALIZATION_VERSION)
    _create_dataset(
        group,
        "frequency_counts",
        np.asarray(
            [row.frequency_contract.output_count for row in rows],
            dtype=np.uint16,
        ),
        request,
    )
    _create_dataset(
        group,
        "navgf",
        np.asarray([row.navgf for row in rows], dtype=np.uint8),
        request,
    )
    _write_frequency_windows(group, rows, request)
    _write_metadata_union(group.create_group("metadata"), rows, request, h5py)


def _write_tr_spectra(h5, request, provenance_index, h5py) -> None:
    rows = request.products.tr_spectra
    if not rows:
        return
    group = h5.create_group("tr_spectra")
    _write_common_rows(group, "tr_spectra", rows, request, provenance_index)
    navg2, tr_length = rows[0].navg2, rows[0].tr_length
    group.attrs["navg2"] = np.uint32(navg2)
    group.attrs["tr_length"] = np.uint32(tr_length)
    group.attrs["native_dtype"] = "int32"
    group.attrs["units"] = rows[0].units
    group.attrs["representation"] = rows[0].representation
    data = np.full(
        (len(rows), NPRODUCTS, navg2, tr_length),
        np.nan,
        dtype=np.float64,
    )
    for index, row in enumerate(rows):
        data[index, row.product_present] = row.data[row.product_present]
    _create_dataset(group, "data", data, request)
    _create_dataset(
        group,
        "navg2_per_sample",
        np.full(len(rows), navg2, dtype=np.uint32),
        request,
    )
    _create_dataset(
        group,
        "tr_length_per_sample",
        np.full(len(rows), tr_length, dtype=np.uint32),
        request,
    )
    _write_metadata_union(group.create_group("metadata"), rows, request, h5py)


def _write_zoom(h5, request, provenance_index, h5py) -> None:
    rows = request.products.zoom_spectra
    if not rows:
        return
    parent = h5.require_group("calibrator")
    group = parent.create_group("zoom_spectra")
    _write_common_rows(group, "zoom_spectra", rows, request, provenance_index)
    data = np.stack([row.data for row in rows]).astype(np.float32, copy=False)
    dataset = _create_dataset(group, "data", data, request)
    dataset.attrs["units"] = rows[0].units
    dataset.attrs["representation"] = rows[0].representation
    dataset.attrs["component_labels"] = np.asarray(rows[0].component_labels, dtype="S3")
    _create_dataset(
        group,
        "pfb_bins",
        np.asarray([row.pfb_bin for row in rows], dtype=np.uint16),
        request,
    )


def _write_waveforms(h5, request, provenance_index, h5py) -> None:
    rows = request.products.waveforms
    if not rows:
        return
    group = h5.create_group("waveform")
    _write_common_rows(group, "waveforms", rows, request, provenance_index)
    data = np.stack([row.data for row in rows]).astype(np.int16, copy=False)
    if data.shape != (len(rows), WAVEFORM_SAMPLES):
        raise ValueError("waveform rows changed after WriteRequest validation")
    dataset = _create_dataset(group, "data", data, request)
    dataset.attrs["units"] = rows[0].units
    dataset.attrs["representation"] = rows[0].representation
    _create_dataset(
        group,
        "channel",
        np.asarray([row.channel for row in rows], dtype=np.uint8),
        request,
    )
    _create_dataset(
        group,
        "adc_timestamps",
        np.asarray([
            0 if row.adc_timestamp is None else row.adc_timestamp for row in rows
        ], dtype=np.uint64),
        request,
    )
    _create_dataset(
        group,
        "adc_timestamp_valid",
        np.asarray([row.adc_timestamp is not None for row in rows], dtype=np.bool_),
        request,
    )
    group.attrs["adc_clock_source"] = ClockSource.ADC.value


def _write_grimm(h5, request, provenance_index, h5py) -> None:
    rows = request.products.grimm_spectra
    if not rows:
        return
    group = h5.create_group("grimm_spectra")
    _write_common_rows(group, "grimm_spectra", rows, request, provenance_index)
    navg2_max = max(row.navg2 for row in rows)
    data = np.zeros((len(rows), navg2_max, NPRODUCTS, 4), dtype=np.int32)
    average_valid = np.zeros((len(rows), navg2_max), dtype=np.bool_)
    for index, row in enumerate(rows):
        data[index, : row.navg2] = row.data
        average_valid[index, : row.navg2] = True
    dataset = _create_dataset(group, "data", data, request)
    dataset.attrs["units"] = rows[0].units
    dataset.attrs["representation"] = rows[0].representation
    dataset.attrs["axis_labels"] = np.asarray(rows[0].axis_labels, dtype="S24")
    dataset.attrs["value_axis_labels"] = np.asarray(
        rows[0].value_axis_labels, dtype="S16"
    )
    _create_dataset(
        group,
        "navg2_per_sample",
        np.asarray([row.navg2 for row in rows], dtype=np.uint32),
        request,
    )
    _create_dataset(group, "average_valid", average_valid, request)


def _write_housekeeping(h5, request, provenance_index, h5py) -> None:
    rows = request.products.housekeeping
    if not rows:
        return
    group = h5.create_group("housekeeping")
    _write_common_rows(group, "housekeeping", rows, request, provenance_index)
    _create_dataset(
        group,
        "hk_type",
        np.asarray([row.hk_type for row in rows], dtype=np.uint16),
        request,
    )
    _create_dataset(
        group,
        "version",
        np.asarray([row.version for row in rows], dtype=np.uint16),
        request,
    )
    _create_dataset(
        group,
        "firmware_errors",
        np.asarray([row.errors for row in rows], dtype=np.uint32),
        request,
    )
    _write_field_union(
        group,
        [row.fields for row in rows],
        [row.field_present for row in rows],
        request,
        h5py,
    )


def _page_mjd(
    page_raw_seconds: np.ndarray,
    request: WriteRequest,
) -> tuple[np.ndarray, np.ndarray]:
    shape = page_raw_seconds.shape
    flattened = page_raw_seconds.reshape(-1)
    raw_valid = np.ones(flattened.shape, dtype=np.bool_)
    sources = [ClockSource.SPECTROMETER.value] * flattened.size
    mjd, valid = _mjd_values(flattened, raw_valid, sources, request)
    return mjd.reshape(shape), valid.reshape(shape)


def _write_calibrator_metadata(h5, request, provenance_index, h5py) -> None:
    rows = request.products.calibrator_metadata
    if not rows:
        return
    group = h5.require_group("calibrator").create_group("metadata")
    _write_common_rows(group, "calibrator_metadata", rows, request, provenance_index)
    _create_dataset(
        group,
        "from_debug",
        np.asarray([row.from_debug for row in rows], dtype=np.bool_),
        request,
    )
    _write_field_union(
        group,
        [row.fields for row in rows],
        [row.field_present for row in rows],
        request,
        h5py,
    )


def _write_calibrator_data(h5, request, provenance_index, h5py) -> None:
    rows = request.products.calibrator_data
    if not rows:
        return
    group = h5.require_group("calibrator").create_group("data")
    _write_common_rows(group, "calibrator_data", rows, request, provenance_index)
    group.attrs["units"] = rows[0].units
    group.attrs["representation"] = rows[0].representation
    group.attrs["page_count"] = np.uint8(rows[0].page_count)
    group.attrs["page_clock_source"] = ClockSource.SPECTROMETER.value
    group.attrs["channel_labels"] = np.asarray(rows[0].channel_labels, dtype=np.uint8)
    data = np.stack([row.data for row in rows])
    _create_dataset(group, "data_real", data.real.astype(np.float64), request)
    _create_dataset(group, "data_imag", data.imag.astype(np.float64), request)
    _create_dataset(
        group,
        "g_nacc",
        np.asarray([row.g_nacc for row in rows], dtype=np.int32),
        request,
    )
    _create_dataset(
        group,
        "gphase",
        np.stack([row.gphase for row in rows]).astype(np.int32, copy=False),
        request,
    )
    page_raw = np.stack([row.page_raw_seconds for row in rows])
    _create_dataset(group, "page_raw_seconds", page_raw, request)
    page_mjd, page_valid = _page_mjd(page_raw, request)
    _create_dataset(group, "page_mjd_times", page_mjd, request)
    _create_dataset(group, "page_mjd_time_valid", page_valid, request)


def _write_calibrator_raw_pfb(h5, request, provenance_index, h5py) -> None:
    rows = request.products.calibrator_raw_pfb
    if not rows:
        return
    group = h5.require_group("calibrator").create_group("raw_pfb")
    _write_common_rows(group, "calibrator_raw_pfb", rows, request, provenance_index)
    group.attrs["units"] = rows[0].units
    group.attrs["representation"] = rows[0].representation
    group.attrs["page_count"] = np.uint8(rows[0].page_count)
    group.attrs["page_clock_source"] = ClockSource.SPECTROMETER.value
    group.attrs["channel_labels"] = np.asarray(rows[0].channel_labels, dtype=np.uint8)
    data = np.stack([row.data for row in rows])
    _create_dataset(group, "data_real", data.real.astype(np.float64), request)
    _create_dataset(group, "data_imag", data.imag.astype(np.float64), request)
    page_raw = np.stack([row.page_raw_seconds for row in rows])
    _create_dataset(group, "page_raw_seconds", page_raw, request)
    page_mjd, page_valid = _page_mjd(page_raw, request)
    _create_dataset(group, "page_mjd_times", page_mjd, request)
    _create_dataset(group, "page_mjd_time_valid", page_valid, request)


def _write_calibrator_debug(h5, request, provenance_index, h5py) -> None:
    rows = request.products.calibrator_debug
    if not rows:
        return
    group = h5.require_group("calibrator").create_group("debug")
    _write_common_rows(group, "calibrator_debug", rows, request, provenance_index)
    group.attrs["page_count"] = np.uint8(rows[0].page_count)
    group.attrs["page_clock_source"] = ClockSource.SPECTROMETER.value
    page_raw = np.stack([row.page_raw_seconds for row in rows])
    _create_dataset(group, "page_raw_seconds", page_raw, request)
    page_mjd, page_valid = _page_mjd(page_raw, request)
    _create_dataset(group, "page_mjd_times", page_mjd, request)
    _create_dataset(group, "page_mjd_time_valid", page_valid, request)
    pages = group.create_group("pages")
    for page_index in range(rows[0].page_count):
        page_group = pages.create_group(f"page_{page_index}")
        page_group.attrs["page_index"] = np.uint8(page_index)
        _write_field_union(
            page_group,
            [row.pages[page_index].fields for row in rows],
            [row.pages[page_index].field_present for row in rows],
            request,
            h5py,
        )


def _write_telemetry(
    h5,
    request: WriteRequest,
    h5py,
) -> None:
    telemetry = request.telemetry
    if telemetry is None:
        return
    group = h5.create_group("telemetry")
    group.attrs["source_kind"] = telemetry.source_kind
    _write_strings(group, "field_names", telemetry.field_names, request, h5py)
    _write_strings(group, "units", telemetry.units, request, h5py)
    for name in (
        "source_indices",
        "mission_seconds",
        "lusee_subsecs",
        "mjd_times",
        "raw_counts",
        "values",
        "valid",
    ):
        _create_dataset(group, name, getattr(telemetry, name), request)


def _write_root_attrs(h5, request: WriteRequest) -> None:
    products = request.products
    h5.attrs["layout_version"] = np.uint16(HDF5_LAYOUT_VERSION)
    h5.attrs["quality_status"] = request.quality_status.value
    h5.attrs["execution_mode"] = products.execution_mode.value
    h5.attrs["issue_count"] = np.uint64(len(request.issues))
    severity_counts = Counter(issue.severity.value for issue in request.issues)
    for severity in ("info", "warning", "error"):
        h5.attrs[f"{severity}_issue_count"] = np.uint64(
            severity_counts.get(severity, 0)
        )
    h5.attrs["input_packet_count"] = np.uint64(products.validated_counts.input_packets)
    h5.attrs["valid_packet_count"] = np.uint64(products.validated_counts.valid_packets)
    for status in request.family_statuses:
        h5.attrs[f"decoded_{status.family}_rows"] = np.uint64(status.decoded_rows)
        h5.attrs[f"persisted_{status.family}_rows"] = np.uint64(
            status.decoded_rows if status.coverage.value == "persisted" else 0
        )


def _populate_layout_v4(
    h5,
    request: WriteRequest,
    h5py,
    *,
    destination_preexisted: bool,
) -> None:
    _write_root_attrs(h5, request)
    _write_session_invariants(h5, request)
    _write_constants(h5, request)
    _write_clock_reference(h5, request, h5py)
    _write_run_provenance(
        h5,
        request,
        destination_preexisted=destination_preexisted,
    )
    _write_decoder_provenance(h5, request, h5py)
    issue_index = _write_issues(h5, request, h5py)
    provenance_index = _write_product_provenance(h5, request, h5py, issue_index)
    _write_family_status(h5, request, h5py, issue_index)
    _write_telemetry(h5, request, h5py)
    _write_spectra(h5, request, provenance_index, h5py)
    _write_tr_spectra(h5, request, provenance_index, h5py)
    _write_zoom(h5, request, provenance_index, h5py)
    _write_waveforms(h5, request, provenance_index, h5py)
    _write_grimm(h5, request, provenance_index, h5py)
    _write_housekeeping(h5, request, provenance_index, h5py)
    _write_calibrator_metadata(h5, request, provenance_index, h5py)
    _write_calibrator_data(h5, request, provenance_index, h5py)
    _write_calibrator_raw_pfb(h5, request, provenance_index, h5py)
    _write_calibrator_debug(h5, request, provenance_index, h5py)


def _write_layout_v4(
    path: Path,
    request: WriteRequest,
    h5py,
    *,
    destination_preexisted: bool,
) -> None:
    with h5py.File(path, "w") as h5:
        _populate_layout_v4(
            h5,
            request,
            h5py,
            destination_preexisted=destination_preexisted,
        )
        h5.flush()


def _verify_layout_v4(
    path: Path,
    request: WriteRequest,
    h5py,
    *,
    destination_preexisted: bool,
) -> None:
    def group_at(h5, path_name: str):
        if path_name not in h5 or not isinstance(h5[path_name], h5py.Group):
            raise ValueError(f"HDF5 output is missing /{path_name}")
        return h5[path_name]

    def check_dataset(group, name: str, shape: tuple[int, ...], dtype):
        path_name = f"{group.name}/{name}"
        if name not in group or not isinstance(group[name], h5py.Dataset):
            raise ValueError(f"HDF5 output is missing {path_name}")
        dataset = group[name]
        string_dtype = dtype is None and h5py.check_string_dtype(dataset.dtype)
        if dataset.shape != shape or (
            dtype is None and string_dtype is None
        ) or (dtype is not None and dataset.dtype != np.dtype(dtype)):
            raise ValueError(f"HDF5 output contract failed for {path_name}")
        return dataset

    def check_attrs(obj, expected: Mapping[str, object]) -> None:
        for name, value in expected.items():
            if name not in obj.attrs:
                raise ValueError(
                    f"HDF5 output attribute {obj.name}@{name} disagrees"
                )
            observed = np.asarray(obj.attrs[name])
            expected_array = np.asarray(value)
            if (
                observed.shape != expected_array.shape
                or observed.dtype != expected_array.dtype
                or not np.array_equal(observed, expected_array)
            ):
                raise ValueError(
                    f"HDF5 output attribute {obj.name}@{name} disagrees"
                )

    def check_values(group, name: str, expected: object) -> None:
        dataset = group[name]
        observed = (
            dataset.asstr()[:]
            if h5py.check_string_dtype(dataset.dtype) is not None
            else dataset[:]
        )
        expected_array = np.asarray(expected)
        if observed.dtype.kind in "fc" and expected_array.dtype.kind in "fc":
            equal = np.array_equal(observed, expected_array, equal_nan=True)
        else:
            equal = np.array_equal(observed, expected_array)
        if not equal:
            raise ValueError(
                f"HDF5 output values disagree in {dataset.name}"
            )

    def optional_attrs(values: Mapping[str, object | None]) -> dict[str, object]:
        expected: dict[str, object] = {}
        for name, value in values.items():
            expected[f"{name}_valid"] = np.bool_(value is not None)
            if value is not None:
                expected[name] = value
        return expected

    def check_field_union(parent, count: int) -> None:
        fields = group_at(parent, "fields")
        presence = group_at(parent, "field_present")
        if set(fields) != set(presence):
            raise ValueError(
                f"HDF5 output field union disagrees in {parent.name}"
            )
        for name in fields:
            present = check_dataset(presence, name, (count,), np.bool_)
            field = group_at(fields, name)
            kind = field.attrs.get("kind")
            if kind == "untyped_absent":
                check_attrs(field, {"kind": "untyped_absent"})
                if np.any(presence[name][:]):
                    raise ValueError(
                        f"HDF5 output absent field is present in {field.name}"
                    )
            elif kind == "mapping":
                check_attrs(field, {"kind": "mapping"})
                check_field_union(field, count)
            elif kind == "array_variants":
                check_dataset(field, "variant_index", (count,), np.int32)
                variant_index = field["variant_index"][:]
                if (
                    "variant_count" not in field.attrs
                    or np.asarray(field.attrs["variant_count"]).dtype
                    != np.dtype(np.uint32)
                ):
                    raise ValueError(
                        f"HDF5 output variant count disagrees in {field.name}"
                    )
                variant_count = int(field.attrs["variant_count"])
                check_attrs(
                    field,
                    {
                        "kind": "array_variants",
                        "variant_count": np.uint32(variant_count),
                    },
                )
                if not np.array_equal(
                    variant_index >= 0,
                    present[:],
                ) or np.any(variant_index[present[:]] >= variant_count):
                    raise ValueError(
                        f"HDF5 output variant presence disagrees in {field.name}"
                    )
                names = {f"variant_{index:03d}" for index in range(variant_count)}
                observed = {
                    item for item in field if item.startswith("variant_")
                } - {"variant_index"}
                if variant_count < 0 or observed != names:
                    raise ValueError(
                        f"HDF5 output variants disagree in {field.name}"
                    )
                for variant_number in range(variant_count):
                    variant_name = f"variant_{variant_number:03d}"
                    variant = group_at(field, variant_name)
                    expected_rows = np.flatnonzero(
                        variant_index == variant_number
                    )
                    check_dataset(
                        variant,
                        "row_indices",
                        (len(expected_rows),),
                        np.uint64,
                    )
                    rows = variant["row_indices"]
                    data = variant.get("data")
                    if not np.array_equal(rows[:], expected_rows):
                        raise ValueError(
                            f"HDF5 output variant rows disagree in {variant.name}"
                        )
                    try:
                        value_shape = tuple(
                            json.loads(variant.attrs["value_shape_json"])
                        )
                        numpy_dtype = np.dtype(variant.attrs["numpy_dtype"])
                    except (KeyError, TypeError, ValueError) as exc:
                        raise ValueError(
                            f"HDF5 output variant is incomplete in {variant.name}"
                        ) from exc
                    string_dtype = (
                        numpy_dtype.kind == "U"
                        and isinstance(data, h5py.Dataset)
                        and h5py.check_string_dtype(data.dtype) is not None
                    )
                    if not isinstance(data, h5py.Dataset) or data.shape != (
                        len(rows),
                        *value_shape,
                    ) or (numpy_dtype.kind == "U" and not string_dtype) or (
                        numpy_dtype.kind != "U" and data.dtype != numpy_dtype
                    ):
                        raise ValueError(
                            f"HDF5 output variant data disagrees in {variant.name}"
                        )
            else:
                raise ValueError(
                    f"HDF5 output field kind is invalid in {field.name}"
                )

    products = request.products
    entries = list(_iter_product_rows(request))
    provenances = [row.provenance for _, _, row in entries]
    schema_rows = [
        (index, schema_id)
        for index, provenance in enumerate(provenances)
        for schema_id in provenance.reported_schema_ids
    ]
    packet_rows = [
        (provenance_index, source_order, packet)
        for provenance_index, provenance in enumerate(provenances)
        for source_order, packet in enumerate(provenance.source_packets)
    ]
    issue_id_to_index = {
        issue.issue_id: index for index, issue in enumerate(request.issues)
    }
    product_issue_rows = [
        (provenance_index, issue_id_to_index[issue_id])
        for provenance_index, provenance in enumerate(provenances)
        for issue_id in provenance.decoder_issue_ids
    ]
    family_issue_rows = [
        (family_index, issue_id_to_index[issue_id])
        for family_index, status in enumerate(request.family_statuses)
        for issue_id in status.issue_ids
    ]
    family_paths = {
        "spectra": "spectra",
        "tr_spectra": "tr_spectra",
        "zoom_spectra": "calibrator/zoom_spectra",
        "waveforms": "waveform",
        "housekeeping": "housekeeping",
        "grimm_spectra": "grimm_spectra",
        "calibrator_metadata": "calibrator/metadata",
        "calibrator_data": "calibrator/data",
        "calibrator_raw_pfb": "calibrator/raw_pfb",
        "calibrator_debug": "calibrator/debug",
    }
    total_rows = len(entries)
    issue_count = len(request.issues)
    decoder = products.decode_provenance
    source_count = len(packet_rows)
    schema_ref_count = len(schema_rows)
    product_issue_count = len(product_issue_rows)
    status_count = len(request.family_statuses)
    family_issue_count = len(family_issue_rows)

    specs = {
        "provenance/decoder": {
            "reported_schema_ids": (
                (len(decoder.reported_schema_ids),),
                np.uint16,
            ),
            "appids": ((len(decoder.appid_counts),), np.uint16),
            "appid_counts": ((len(decoder.appid_counts),), np.uint64),
            "issue_codes": ((len(decoder.issue_counts),), None),
            "issue_code_counts": ((len(decoder.issue_counts),), np.uint64),
        },
        "issues": {
            **{
                name: ((issue_count,), None)
                for name in (
                    "issue_id",
                    "code",
                    "stage",
                    "message",
                    "input_identity",
                    "bank",
                    "session",
                    "severity",
                    "action",
                    "details_json",
                )
            },
            **{
                name: ((issue_count,), dtype)
                for name, dtype in (
                    ("input_identity_valid", np.bool_),
                    ("bank_valid", np.bool_),
                    ("session_valid", np.bool_),
                    ("byte_offset", np.uint64),
                    ("byte_offset_valid", np.bool_),
                    ("frame_index", np.uint64),
                    ("frame_index_valid", np.bool_),
                    ("packet_index", np.uint64),
                    ("packet_index_valid", np.bool_),
                    ("appid", np.uint16),
                    ("appid_valid", np.bool_),
                    ("sequence_count", np.uint16),
                    ("sequence_count_valid", np.bool_),
                    ("uid", np.uint32),
                    ("uid_valid", np.bool_),
                )
            },
        },
        "provenance/product_rows": {
            **{
                name: ((total_rows,), None)
                for name in (
                    "family",
                    "uid_source",
                    "uid_source_role",
                    "time_source",
                    "time_source_role",
                    "clock_source",
                )
            },
            **{
                name: ((total_rows,), dtype)
                for name, dtype in (
                    ("row_index", np.uint64),
                    ("unique_ids", np.uint32),
                    ("uid_source_role_valid", np.bool_),
                    ("time_source_valid", np.bool_),
                    ("time_source_role_valid", np.bool_),
                    ("clock_source_valid", np.bool_),
                    ("time_valid", np.bool_),
                    ("selected_schema_ids", np.uint16),
                )
            },
        },
        "provenance/product_schema_refs": {
            "provenance_index": ((schema_ref_count,), np.uint64),
            "schema_id": ((schema_ref_count,), np.uint16),
        },
        "provenance/source_packets": {
            **{
                name: ((source_count,), dtype)
                for name, dtype in (
                    ("provenance_index", np.uint64),
                    ("source_order", np.uint16),
                    ("original_appid", np.uint16),
                    ("normalized_appid", np.uint16),
                    ("normalized_appid_valid", np.bool_),
                    ("packet_index", np.uint64),
                    ("packet_index_valid", np.bool_),
                    ("frame_start", np.uint64),
                    ("frame_start_valid", np.bool_),
                    ("frame_stop", np.uint64),
                    ("frame_stop_valid", np.bool_),
                    ("byte_offset_start", np.uint64),
                    ("byte_offset_start_valid", np.bool_),
                    ("byte_offset_stop", np.uint64),
                    ("byte_offset_stop_valid", np.bool_),
                    ("filename_valid", np.bool_),
                    ("bank_valid", np.bool_),
                )
            },
            "role": ((source_count,), None),
            "filename": ((source_count,), None),
            "bank": ((source_count,), None),
        },
        "provenance/product_issue_refs": {
            "provenance_index": ((product_issue_count,), np.uint64),
            "issue_index": ((product_issue_count,), np.uint64),
        },
        "status/families": {
            "family": ((status_count,), None),
            "supported": ((status_count,), np.bool_),
            "coverage": ((status_count,), None),
            "quality": ((status_count,), None),
            "decoded_rows": ((status_count,), np.uint64),
            "persisted_rows": ((status_count,), np.uint64),
            "reason": ((status_count,), None),
            "reason_valid": ((status_count,), np.bool_),
        },
        "status/family_issue_refs": {
            "family_index": ((family_issue_count,), np.uint64),
            "issue_index": ((family_issue_count,), np.uint64),
        },
    }
    session_fields = {
        "software_version": (
            None if products.sw_version is None else np.uint32(products.sw_version)
        ),
        "firmware_version": (
            None if products.fw_version is None else np.uint32(products.fw_version)
        ),
        "firmware_id": (
            None if products.fw_id is None else np.uint32(products.fw_id)
        ),
        "firmware_date": (
            None if products.fw_date is None else np.uint32(products.fw_date)
        ),
        "firmware_time": (
            None if products.fw_time is None else np.uint32(products.fw_time)
        ),
        "start_unique_packet_id": (
            None
            if products.start_unique_packet_id is None
            else np.uint32(products.start_unique_packet_id)
        ),
        "start_time_32": (
            None
            if products.start_time_32 is None
            else np.uint32(products.start_time_32)
        ),
        "start_time_16": (
            None
            if products.start_time_16 is None
            else np.uint16(products.start_time_16)
        ),
        "start_raw_seconds": (
            None
            if products.start_raw_seconds is None
            else np.float64(products.start_raw_seconds)
        ),
    }
    run_attrs = optional_attrs(request.run_provenance.as_record())
    run_attrs.update(
        {
            "overwrite_requested": np.bool_(request.overwrite),
            "destination_preexisted": np.bool_(destination_preexisted),
            **optional_attrs(
                {
                    "hdf5_compression": request.hdf5_compression,
                    "hdf5_compression_level": request.hdf5_compression_level,
                }
            ),
        }
    )
    decoder_optional_names = (
        "decoder_name",
        "distribution_version",
        "decoder_source_commit",
        "binding_key",
        "schema_variant",
        "binding_source_release",
        "binding_source_commit",
        "abi_fingerprint",
        "canonical_report_json",
    )
    decoder_attrs = optional_attrs(
        {name: getattr(decoder, name) for name in decoder_optional_names}
    )
    decoder_attrs.update(
        {
            "selected_schema_id": np.uint16(decoder.selected_schema_id),
            "schema_assumed": np.bool_(decoder.schema_assumed),
            "execution_mode": decoder.execution_mode.value,
            "input_packet_count": np.uint64(decoder.input_packet_count),
            "valid_packet_count": np.uint64(decoder.valid_packet_count),
        }
    )
    attrs: dict[str, dict[str, object]] = {
        "session_invariants": optional_attrs(session_fields),
        "constants": {
            "lun_lat_deg": np.float64(request.location.latitude_deg),
            "lun_long_deg": np.float64(request.location.longitude_deg),
            "lun_height_m": np.float64(request.location.height_m),
        },
        "clock_reference": {
            "available": np.bool_(request.clock_reference_set is not None)
        },
        "run_provenance": run_attrs,
        "provenance/decoder": decoder_attrs,
        "issues": {"count": np.uint64(issue_count)},
    }
    severity_counts = Counter(issue.severity.value for issue in request.issues)
    attrs[""] = {
        f"{severity}_issue_count": np.uint64(severity_counts.get(severity, 0))
        for severity in ("info", "warning", "error")
    }
    for status in request.family_statuses:
        attrs[""].update(
            {
                f"decoded_{status.family}_rows": np.uint64(status.decoded_rows),
                f"persisted_{status.family}_rows": np.uint64(
                    status.decoded_rows
                    if status.coverage.value == "persisted"
                    else 0
                ),
            }
        )
    reference_set = request.clock_reference_set
    if reference_set is None:
        attrs["clock_reference"]["unavailable_reason"] = (
            request.clock_reference_unavailable_reason
        )
    else:
        attrs["clock_reference"].update(
            {
                "format_version": np.uint16(reference_set.format_version),
                "reference_event": reference_set.reference_event,
                "clock_reference_isot": reference_set.clock_reference_isot,
                "time_scale": reference_set.time_scale,
                "source": reference_set.source,
                "assumed": np.bool_(reference_set.assumed),
                "source_sha256": reference_set.source_sha256,
                "canonical_record_json": _canonical_json(
                    reference_set.as_record()
                ),
            }
        )
        specs["clock_reference"] = {
            "clock_sources": ((len(reference_set.clocks),), None),
            "clock_reference_raw_seconds": (
                (len(reference_set.clocks),),
                np.float64,
            ),
        }

    values: dict[str, dict[str, object]] = {
        "provenance/decoder": {
            "reported_schema_ids": np.asarray(
                decoder.reported_schema_ids, dtype=np.uint16
            ),
            "appids": np.asarray(
                [item[0] for item in decoder.appid_counts], dtype=np.uint16
            ),
            "appid_counts": np.asarray(
                [item[1] for item in decoder.appid_counts], dtype=np.uint64
            ),
            "issue_codes": [item[0] for item in decoder.issue_counts],
            "issue_code_counts": np.asarray(
                [item[1] for item in decoder.issue_counts], dtype=np.uint64
            ),
        },
        "issues": {
            "issue_id": [issue.issue_id for issue in request.issues],
            "code": [issue.code for issue in request.issues],
            "stage": [issue.stage for issue in request.issues],
            "message": [issue.message for issue in request.issues],
            "severity": [issue.severity.value for issue in request.issues],
            "action": [issue.action.value for issue in request.issues],
            "details_json": [
                _canonical_json(issue.as_dict()["details"])
                for issue in request.issues
            ],
        },
        "provenance/product_rows": {
            "family": [family for family, _, _ in entries],
            "row_index": np.asarray(
                [row_index for _, row_index, _ in entries], dtype=np.uint64
            ),
            "unique_ids": np.asarray(
                [row.unique_packet_id for _, _, row in entries], dtype=np.uint32
            ),
            "uid_source": [item.uid_source for item in provenances],
            "time_valid": np.asarray(
                [item.time_valid for item in provenances], dtype=np.bool_
            ),
            "selected_schema_ids": np.asarray(
                [item.selected_schema_id for item in provenances], dtype=np.uint16
            ),
        },
        "provenance/product_schema_refs": {
            "provenance_index": np.asarray(
                [item[0] for item in schema_rows], dtype=np.uint64
            ),
            "schema_id": np.asarray(
                [item[1] for item in schema_rows], dtype=np.uint16
            ),
        },
        "provenance/source_packets": {
            "provenance_index": np.asarray(
                [item[0] for item in packet_rows], dtype=np.uint64
            ),
            "source_order": np.asarray(
                [item[1] for item in packet_rows], dtype=np.uint16
            ),
            "role": [item[2].role for item in packet_rows],
            "original_appid": np.asarray(
                [item[2].original_appid for item in packet_rows], dtype=np.uint16
            ),
        },
        "provenance/product_issue_refs": {
            "provenance_index": np.asarray(
                [item[0] for item in product_issue_rows], dtype=np.uint64
            ),
            "issue_index": np.asarray(
                [item[1] for item in product_issue_rows], dtype=np.uint64
            ),
        },
        "status/families": {
            "family": [status.family for status in request.family_statuses],
            "supported": np.asarray(
                [status.supported for status in request.family_statuses],
                dtype=np.bool_,
            ),
            "coverage": [
                status.coverage.value for status in request.family_statuses
            ],
            "quality": [status.quality.value for status in request.family_statuses],
            "decoded_rows": np.asarray(
                [status.decoded_rows for status in request.family_statuses],
                dtype=np.uint64,
            ),
            "persisted_rows": np.asarray(
                [
                    status.decoded_rows
                    if status.coverage.value == "persisted"
                    else 0
                    for status in request.family_statuses
                ],
                dtype=np.uint64,
            ),
        },
        "status/family_issue_refs": {
            "family_index": np.asarray(
                [item[0] for item in family_issue_rows], dtype=np.uint64
            ),
            "issue_index": np.asarray(
                [item[1] for item in family_issue_rows], dtype=np.uint64
            ),
        },
    }
    if reference_set is not None:
        values["clock_reference"] = {
            "clock_sources": [
                item.clock_source.value for item in reference_set.clocks
            ],
            "clock_reference_raw_seconds": np.asarray(
                [item.clock_reference_raw_seconds for item in reference_set.clocks],
                dtype=np.float64,
            ),
        }
    for name in ("input_identity", "bank", "session"):
        data, valid = _optional_string_column(
            [getattr(issue, name) for issue in request.issues]
        )
        values["issues"][name] = data
        values["issues"][f"{name}_valid"] = valid
    for name, dtype in (
        ("byte_offset", np.uint64),
        ("frame_index", np.uint64),
        ("packet_index", np.uint64),
        ("appid", np.uint16),
        ("sequence_count", np.uint16),
        ("uid", np.uint32),
    ):
        data, valid = _optional_integer_column(
            [getattr(issue, name) for issue in request.issues], dtype
        )
        values["issues"][name] = data
        values["issues"][f"{name}_valid"] = valid
    for name in (
        "uid_source_role",
        "time_source",
        "time_source_role",
        "clock_source",
    ):
        data, valid = _optional_string_column(
            [getattr(item, name) for item in provenances]
        )
        values["provenance/product_rows"][name] = data
        values["provenance/product_rows"][f"{name}_valid"] = valid
    for name, dtype in (
        ("normalized_appid", np.uint16),
        ("packet_index", np.uint64),
        ("frame_start", np.uint64),
        ("frame_stop", np.uint64),
        ("byte_offset_start", np.uint64),
        ("byte_offset_stop", np.uint64),
    ):
        data, valid = _optional_integer_column(
            [getattr(item[2], name) for item in packet_rows], dtype
        )
        values["provenance/source_packets"][name] = data
        values["provenance/source_packets"][f"{name}_valid"] = valid
    for name in ("filename", "bank"):
        data, valid = _optional_string_column(
            [getattr(item[2], name) for item in packet_rows]
        )
        values["provenance/source_packets"][name] = data
        values["provenance/source_packets"][f"{name}_valid"] = valid
    reasons, reason_valid = _optional_string_column(
        [status.reason for status in request.family_statuses]
    )
    values["status/families"]["reason"] = reasons
    values["status/families"]["reason_valid"] = reason_valid

    telemetry = request.telemetry
    if telemetry is not None:
        row_count = telemetry.row_count
        field_count = len(telemetry.field_names)
        attrs["telemetry"] = {"source_kind": telemetry.source_kind}
        specs["telemetry"] = {
            "field_names": ((field_count,), None),
            "units": ((field_count,), None),
            "source_indices": ((row_count,), np.int64),
            "mission_seconds": ((row_count,), np.uint32),
            "lusee_subsecs": ((row_count,), np.uint16),
            "mjd_times": ((row_count,), np.float64),
            "raw_counts": ((row_count, field_count), np.uint16),
            "values": ((row_count, field_count), np.float64),
            "valid": ((row_count, field_count), np.bool_),
        }
        values["telemetry"] = {
            "field_names": telemetry.field_names,
            "units": telemetry.units,
            "source_indices": telemetry.source_indices,
            "mission_seconds": telemetry.mission_seconds,
            "lusee_subsecs": telemetry.lusee_subsecs,
            "mjd_times": telemetry.mjd_times,
            "raw_counts": telemetry.raw_counts,
            "values": telemetry.values,
            "valid": telemetry.valid,
        }
    field_unions: list[tuple[str, int]] = []
    forbidden_paths: list[str] = []

    common = {
        "unique_ids": (None, np.uint32),
        "raw_seconds": (None, np.float64),
        "raw_time_valid": (None, np.bool_),
        "mjd_times": (None, np.float64),
        "mjd_time_valid": (None, np.bool_),
        "original_indices": (None, np.uint64),
        "provenance_index": (None, np.uint64),
    }
    for family, _ in FAMILY_TYPES:
        rows = getattr(products, family)
        path_name = family_paths[family]
        if rows:
            count = len(rows)
            specs[path_name] = {
                name: ((count,), dtype) for name, (_, dtype) in common.items()
            }
            attrs[path_name] = {"count": np.uint64(count)}
        status = next(
            item for item in request.family_statuses if item.family == family
        )

    if products.spectra:
        rows = products.spectra
        count = len(rows)
        specs["spectra"].update(
            {
                "data": ((count, NPRODUCTS, NCHANNELS), np.float32),
                "frequency_counts": ((count,), np.uint16),
                "navgf": ((count,), np.uint8),
                "frequency_window_index": ((count,), np.uint8),
            }
        )
        contracts = {row.navgf: row.frequency_contract for row in rows}
        contract_count = len(contracts)
        specs["spectra/frequency_windows"] = {
            "navgf": ((contract_count,), np.uint8),
            "native_count": ((contract_count,), np.uint16),
            "output_count": ((contract_count,), np.uint16),
            "stride": ((contract_count,), np.uint16),
            "divisor": ((contract_count,), np.uint16),
            "included_offsets": ((contract_count, 4), np.uint8),
            "included_offset_valid": ((contract_count, 4), np.bool_),
            "nominal_response_weights": ((contract_count, 4), np.float64),
        }
        first_contract = contracts[min(contracts)]
        attrs["spectra/data"] = {
            "units": rows[0].units,
            "representation": rows[0].representation,
            "bitslice_restored": np.bool_(True),
            "bitslice_reference": np.uint8(BITSLICE_REFERENCE),
            "normalization_version": np.uint16(
                SPECTRA_NORMALIZATION_VERSION
            ),
        }
        attrs["spectra/frequency_windows"] = {
            "contract_name": first_contract.contract_name,
            "contract_version": np.uint16(first_contract.contract_version),
            "source_commit": first_contract.source_commit,
            "frequency_coordinate_status": (
                first_contract.frequency_coordinate_status
            ),
            "integer_arithmetic": first_contract.as_record()[
                "integer_arithmetic"
            ],
        }
        field_unions.append(("spectra/metadata", count))
        forbidden_paths.extend(
            ("spectra/product_present", "spectra/data_valid")
        )

    if products.tr_spectra:
        rows = products.tr_spectra
        count = len(rows)
        first = rows[0]
        specs["tr_spectra"].update(
            {
                "data": (
                    (count, NPRODUCTS, first.navg2, first.tr_length),
                    np.float64,
                ),
                "navg2_per_sample": ((count,), np.uint32),
                "tr_length_per_sample": ((count,), np.uint32),
            }
        )
        attrs["tr_spectra"].update(
            {
                "navg2": np.uint32(first.navg2),
                "tr_length": np.uint32(first.tr_length),
                "native_dtype": "int32",
                "units": first.units,
                "representation": first.representation,
            }
        )
        field_unions.append(("tr_spectra/metadata", count))
        forbidden_paths.extend(
            ("tr_spectra/product_present", "tr_spectra/data_valid")
        )

    if products.zoom_spectra:
        rows = products.zoom_spectra
        count = len(rows)
        specs["calibrator/zoom_spectra"].update(
            {
                "data": ((count, *rows[0].data.shape), np.float32),
                "pfb_bins": ((count,), np.uint16),
            }
        )
        attrs["calibrator/zoom_spectra/data"] = {
            "units": rows[0].units,
            "representation": rows[0].representation,
            "component_labels": np.asarray(
                rows[0].component_labels,
                dtype="S3",
            ),
        }

    if products.waveforms:
        rows = products.waveforms
        count = len(rows)
        specs["waveform"].update(
            {
                "data": ((count, WAVEFORM_SAMPLES), np.int16),
                "channel": ((count,), np.uint8),
                "adc_timestamps": ((count,), np.uint64),
                "adc_timestamp_valid": ((count,), np.bool_),
            }
        )
        attrs["waveform"].update(
            {"adc_clock_source": ClockSource.ADC.value}
        )
        attrs["waveform/data"] = {
            "units": rows[0].units,
            "representation": rows[0].representation,
        }

    if products.grimm_spectra:
        rows = products.grimm_spectra
        count = len(rows)
        navg2_max = max(row.navg2 for row in rows)
        specs["grimm_spectra"].update(
            {
                "data": (
                    (count, navg2_max, NPRODUCTS, 4),
                    np.int32,
                ),
                "navg2_per_sample": ((count,), np.uint32),
                "average_valid": ((count, navg2_max), np.bool_),
            }
        )
        attrs["grimm_spectra/data"] = {
            "units": rows[0].units,
            "representation": rows[0].representation,
            "axis_labels": np.asarray(rows[0].axis_labels, dtype="S24"),
            "value_axis_labels": np.asarray(
                rows[0].value_axis_labels,
                dtype="S16",
            ),
        }

    if products.housekeeping:
        count = len(products.housekeeping)
        specs["housekeeping"].update(
            {
                "hk_type": ((count,), np.uint16),
                "version": ((count,), np.uint16),
                "firmware_errors": ((count,), np.uint32),
            }
        )
        field_unions.append(("housekeeping", count))

    if products.calibrator_metadata:
        count = len(products.calibrator_metadata)
        specs["calibrator/metadata"].update(
            {"from_debug": ((count,), np.bool_)}
        )
        field_unions.append(("calibrator/metadata", count))

    for family, path_name, include_gain in (
        ("calibrator_data", "calibrator/data", True),
        ("calibrator_raw_pfb", "calibrator/raw_pfb", False),
    ):
        rows = getattr(products, family)
        if not rows:
            continue
        count = len(rows)
        first = rows[0]
        specs[path_name].update(
            {
                "data_real": ((count, *first.data.shape), np.float64),
                "data_imag": ((count, *first.data.shape), np.float64),
                "page_raw_seconds": (
                    (count, first.page_count),
                    np.float64,
                ),
                "page_mjd_times": (
                    (count, first.page_count),
                    np.float64,
                ),
                "page_mjd_time_valid": (
                    (count, first.page_count),
                    np.bool_,
                ),
            }
        )
        if include_gain:
            specs[path_name].update(
                {
                    "g_nacc": ((count,), np.int32),
                    "gphase": ((count, *first.gphase.shape), np.int32),
                }
            )
        attrs[path_name].update(
            {
                "units": first.units,
                "representation": first.representation,
                "page_count": np.uint8(first.page_count),
                "page_clock_source": ClockSource.SPECTROMETER.value,
                "channel_labels": np.asarray(
                    first.channel_labels,
                    dtype=np.uint8,
                ),
            }
        )

    if products.calibrator_debug:
        rows = products.calibrator_debug
        count = len(rows)
        first = rows[0]
        page_shape = (count, first.page_count)
        specs["calibrator/debug"].update(
            {
                "page_raw_seconds": (page_shape, np.float64),
                "page_mjd_times": (page_shape, np.float64),
                "page_mjd_time_valid": (page_shape, np.bool_),
            }
        )
        attrs["calibrator/debug"].update(
            {
                "page_count": np.uint8(first.page_count),
                "page_clock_source": ClockSource.SPECTROMETER.value,
            }
        )
        pages = {
            f"calibrator/debug/pages/page_{index}": index
            for index in range(first.page_count)
        }
        attrs.update(
            {
                path_name: {"page_index": np.uint8(index)}
                for path_name, index in pages.items()
            }
        )
        field_unions.extend((path_name, count) for path_name in pages)

    with h5py.File(path, "r") as h5:
        check_attrs(
            h5,
            {
                "layout_version": np.uint16(HDF5_LAYOUT_VERSION),
                "quality_status": request.quality_status.value,
                "execution_mode": products.execution_mode.value,
                "issue_count": np.uint64(issue_count),
                "input_packet_count": np.uint64(
                    products.validated_counts.input_packets
                ),
                "valid_packet_count": np.uint64(
                    products.validated_counts.valid_packets
                ),
            },
        )
        for family, _ in FAMILY_TYPES:
            rows = getattr(products, family)
            path_name = family_paths[family]
            if not rows and path_name in h5:
                raise ValueError(
                    f"HDF5 output unexpectedly contains /{path_name}"
                )
        if telemetry is None:
            if "telemetry" in h5:
                raise ValueError("HDF5 output unexpectedly contains /telemetry")
        else:
            telemetry_group = group_at(h5, "telemetry")
            if set(telemetry_group) != set(specs["telemetry"]):
                raise ValueError("HDF5 output telemetry tree is not canonical")
        for path_name, datasets in specs.items():
            group = group_at(h5, path_name)
            for name, (shape, dtype) in datasets.items():
                check_dataset(group, name, shape, dtype)
        for path_name, expected in attrs.items():
            if path_name and path_name not in h5:
                raise ValueError(f"HDF5 output is missing /{path_name}")
            obj = h5 if path_name == "" else h5[path_name]
            check_attrs(obj, expected)
        for path_name, datasets in values.items():
            group = group_at(h5, path_name)
            for name, expected in datasets.items():
                check_values(group, name, expected)
        for path_name, count in field_unions:
            check_field_union(group_at(h5, path_name), count)
        for path_name in forbidden_paths:
            if path_name in h5:
                raise ValueError(
                    f"HDF5 output persisted forbidden mask /{path_name}"
                )


def write_hdf5(request: WriteRequest, dest: Path | str) -> Path:
    """Validate, write, and verify one layout-v4 file."""
    if not isinstance(request, WriteRequest):
        raise TypeError("write_hdf5 requires a validated WriteRequest")
    request.validate()
    destination = Path(dest)
    if destination.exists() and not request.overwrite:
        raise FileExistsError(destination)
    destination_preexisted = destination.exists()
    h5py = import_optional_dependency("h5py", "HDF5 ingest output")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_layout_v4(
        destination,
        request,
        h5py,
        destination_preexisted=destination_preexisted,
    )
    _verify_layout_v4(
        destination,
        request,
        h5py,
        destination_preexisted=destination_preexisted,
    )
    log.info("wrote layout-v4 HDF5 %s", destination)
    return destination


def _to_mjd(
    raw_seconds: np.ndarray,
    raw_subtract: float,
    mjd_offset: float,
) -> np.ndarray:
    """Retain layout-v3 time arithmetic for the legacy FITS writer."""
    return (raw_seconds - raw_subtract) / 86400.0 + mjd_offset


__all__ = ["write_hdf5"]
