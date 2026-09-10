"""Single compatibility and provenance boundary for the uncrater decoder."""

from __future__ import annotations

import importlib
import inspect
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from importlib import metadata
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import Any, Mapping

from .dependencies import import_optional_dependency
from .issues import (
    IngestIssue,
    IssueAction,
    IssueCollector,
    IssueSeverity,
)
from .products import (
    DecodeProvenance,
    ExecutionMode,
    SourcePacketProvenance,
)


class IncompatibleUncraterError(RuntimeError):
    """The installed uncrater lacks the reviewed public decoder contract."""


@dataclass(frozen=True)
class DecoderInfo:
    """Installed uncrater distribution provenance."""

    distribution_version: str | None
    source_commit: str | None


@dataclass(frozen=True)
class UncraterBindingInfo:
    """One Collection's selected schema and decode-summary provenance."""

    reported_schema_ids: tuple[int, ...]
    selected_schema_id: int
    binding_key: str
    variant: str | None
    schema_assumed: bool
    source_release: str
    source_commit: str
    abi_fingerprint: str
    appid_counts: tuple[tuple[int, int], ...]
    issue_counts: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class ImportedDecodeIssues:
    """Issues imported during one decoder-boundary call."""

    issue_ids_by_packet_index: Mapping[int, tuple[str, ...]]
    issues: tuple[IngestIssue, ...]

    def __post_init__(self) -> None:
        lookup = {
            packet_index: tuple(issue_ids)
            for packet_index, issue_ids in self.issue_ids_by_packet_index.items()
        }
        object.__setattr__(
            self,
            "issue_ids_by_packet_index",
            MappingProxyType(lookup),
        )
        object.__setattr__(self, "issues", tuple(self.issues))


_REQUIRED_TOP_LEVEL = (
    "Packet",
    "PacketBase",
    "Collection",
    "Packet_Cal_Data",
    "Packet_Cal_Debug",
    "Packet_Cal_Metadata",
    "Packet_Cal_RawPFB",
    "Packet_Cal_ZoomSpectra",
    "Packet_Grimm",
    "Packet_Hello",
    "Packet_Housekeep",
    "Packet_Metadata",
    "Packet_Spectrum",
    "Packet_TR_Spectrum",
    "Packet_Waveform",
    "Packet_Waveform_Meta",
    "NPRODUCTS",
    "NCHANNELS",
    "id",
    "normalize_dcb_appid",
    "appid_is_cal_data",
    "appid_is_cal_debug",
    "appid_is_cal_metadata",
    "appid_is_cal_raw_pfb",
    "appid_is_cal_segmented_payload",
    "appid_is_grimm_spectrum",
    "appid_is_heartbeat",
    "appid_is_hello",
    "appid_is_housekeeping",
    "appid_is_metadata",
    "appid_is_raw_adc",
    "appid_is_raw_adc_metadata",
    "appid_is_spectrum",
    "appid_is_tr_spectrum",
    "appid_is_watchdog",
    "appid_is_zoom_spectrum",
)

_REQUIRED_PACKET_CLASSES = (
    "Packet_Cal_Data",
    "Packet_Cal_Debug",
    "Packet_Cal_Metadata",
    "Packet_Cal_RawPFB",
    "Packet_Cal_ZoomSpectra",
    "Packet_Grimm",
    "Packet_Hello",
    "Packet_Housekeep",
    "Packet_Metadata",
    "Packet_Spectrum",
    "Packet_TR_Spectrum",
    "Packet_Waveform",
    "Packet_Waveform_Meta",
)

_REQUIRED_CONSTANTS = {
    "NPRODUCTS": 16,
    "NCHANNELS": 2048,
}


def _required_submodule(name: str) -> ModuleType:
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name != name:
            raise
        raise IncompatibleUncraterError(
            f"installed uncrater has no public {name!r} module"
        ) from exc
    except ImportError as exc:
        raise IncompatibleUncraterError(
            f"installed uncrater cannot import public {name!r}: {exc}"
        ) from exc


@lru_cache(maxsize=1)
def load_uncrater() -> ModuleType:
    """Load and validate the repaired decoder's public surface."""
    decoder = import_optional_dependency(
        "uncrater", "LuSEE packet identity assignment and decoding"
    )
    missing = [name for name in _REQUIRED_TOP_LEVEL if not hasattr(decoder, name)]
    if missing:
        raise IncompatibleUncraterError(
            "installed uncrater lacks required public attributes: "
            + ", ".join(missing)
        )
    for name in ("PacketBase", "Collection"):
        if not inspect.isclass(getattr(decoder, name)):
            raise IncompatibleUncraterError(
                f"installed uncrater {name} is not a public class"
            )
    if not callable(getattr(decoder.PacketBase, "read", None)):
        raise IncompatibleUncraterError(
            "installed uncrater PacketBase has no public read() method"
        )
    for name in _REQUIRED_PACKET_CLASSES:
        packet_class = getattr(decoder, name)
        if not inspect.isclass(packet_class):
            raise IncompatibleUncraterError(
                f"installed uncrater {name} is not a public packet class"
            )
        try:
            is_packet = issubclass(packet_class, decoder.PacketBase)
        except TypeError as exc:
            raise IncompatibleUncraterError(
                "installed uncrater PacketBase is not a public packet class"
            ) from exc
        if not is_packet:
            raise IncompatibleUncraterError(
                f"installed uncrater {name} is not a PacketBase subclass"
            )
    for name, expected in _REQUIRED_CONSTANTS.items():
        value = getattr(decoder, name)
        if type(value) is not int or value != expected:
            raise IncompatibleUncraterError(
                f"installed uncrater {name} must equal {expected}"
            )
    if not callable(decoder.normalize_dcb_appid):
        raise IncompatibleUncraterError(
            "installed uncrater normalize_dcb_appid is not callable"
        )
    parameters = inspect.signature(decoder.Collection).parameters
    missing_parameters = [
        name
        for name in ("strict", "diagnostic_override", "schema_variant", "waveform_packet_context")
        if name not in parameters
    ]
    if missing_parameters:
        raise IncompatibleUncraterError(
            "installed uncrater Collection lacks constructor parameters: "
            + ", ".join(missing_parameters)
        )
    if not callable(getattr(decoder.Collection, "canonical_report", None)):
        raise IncompatibleUncraterError(
            "installed uncrater Collection has no public canonical_report()"
        )

    registry = _required_submodule("uncrater.schema_registry")
    for name in ("LATEST_BINDING", "binding_for_key", "resolve_wire_version"):
        if not hasattr(registry, name):
            raise IncompatibleUncraterError(
                f"installed uncrater.schema_registry lacks {name}"
            )
    status = _required_submodule("uncrater.decode_status")
    for name in ("DecodeIssue", "DecodeStatus", "PacketDecodeError"):
        if not hasattr(status, name):
            raise IncompatibleUncraterError(
                f"installed uncrater.decode_status lacks {name}"
            )
        if not inspect.isclass(getattr(status, name)):
            raise IncompatibleUncraterError(
                f"installed uncrater.decode_status {name} is not a class"
            )
    return decoder


def _source_commit(distribution: metadata.Distribution) -> str | None:
    raw = distribution.read_text("direct_url.json")
    if not raw:
        return None
    try:
        direct_url = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return None
    vcs_info = direct_url.get("vcs_info")
    if not isinstance(vcs_info, dict):
        return None
    commit = vcs_info.get("commit_id")
    return commit if isinstance(commit, str) and commit else None


def decoder_info() -> DecoderInfo:
    """Return installed package version and optional PEP 610 Git commit."""
    decoder = load_uncrater()
    try:
        distribution = metadata.distribution("uncrater")
    except metadata.PackageNotFoundError:
        version = getattr(decoder, "__version__", None)
        return DecoderInfo(
            distribution_version=version if isinstance(version, str) else None,
            source_commit=None,
        )
    return DecoderInfo(
        distribution_version=distribution.version,
        source_commit=_source_commit(distribution),
    )


def make_collection(
    path: Path | str,
    *,
    strict: bool = False,
    diagnostic_override: bool = False,
    schema_variant: str | None = None,
    waveform_packet_context: Mapping[int, Mapping[str, object]] | None = None,
) -> Any:
    """Construct a Collection using only reviewed public options."""
    decoder = load_uncrater()
    return decoder.Collection(
        str(path),
        strict=strict,
        diagnostic_override=diagnostic_override,
        schema_variant=schema_variant,
        waveform_packet_context=waveform_packet_context,
    )


def read_packet(packet: Any) -> None:
    """Decode one packet through uncrater's public idempotent method."""
    read = getattr(packet, "read", None)
    if not callable(read):
        raise IncompatibleUncraterError(
            f"uncrater packet {type(packet).__name__} has no public read() method"
        )
    read()


def _required_public_attribute(value: Any, name: str) -> Any:
    if not hasattr(value, name):
        raise IncompatibleUncraterError(
            f"uncrater {type(value).__name__} has no public {name} field"
        )
    return getattr(value, name)


def _public_integer(
    value: Any,
    name: str,
    *,
    maximum: int | None = None,
) -> int:
    if type(value) is not int or value < 0:
        raise IncompatibleUncraterError(
            f"uncrater {name} must be a nonnegative integer"
        )
    if maximum is not None and value > maximum:
        raise IncompatibleUncraterError(
            f"uncrater {name} must be an integer in [0, {maximum}]"
        )
    return value


def _public_decode_issues(status: Any, status_module: ModuleType) -> tuple[Any, ...]:
    if not isinstance(status, status_module.DecodeStatus):
        raise IncompatibleUncraterError(
            "uncrater decode_status is not a public DecodeStatus"
        )
    issues = _required_public_attribute(status, "issues")
    if not isinstance(issues, tuple):
        raise IncompatibleUncraterError(
            "uncrater DecodeStatus.issues must be an immutable tuple"
        )
    for issue in issues:
        if not isinstance(issue, status_module.DecodeIssue):
            raise IncompatibleUncraterError(
                "uncrater DecodeStatus.issues contains a non-DecodeIssue"
            )
    return issues


def _validate_public_detail(value: Any) -> None:
    if value is None or isinstance(value, (bool, int, str)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise IncompatibleUncraterError(
                "uncrater DecodeIssue.details contains a nonfinite float"
            )
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise IncompatibleUncraterError(
                    "uncrater DecodeIssue.details mapping keys must be strings"
                )
            _validate_public_detail(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _validate_public_detail(item)
        return
    raise IncompatibleUncraterError(
        "uncrater DecodeIssue.details must be JSON-compatible"
    )


@dataclass(frozen=True)
class _PacketContract:
    packet_index: int
    original_appid: int
    normalized_appid: int
    schema_id: int
    issues: tuple[Any, ...]


def _packet_contract(
    packet: Any,
    decoder: ModuleType,
    status_module: ModuleType,
) -> _PacketContract:
    if not isinstance(packet, decoder.PacketBase):
        raise IncompatibleUncraterError(
            "uncrater Collection.cont contains a non-PacketBase value"
        )
    packet_index = _public_integer(
        _required_public_attribute(packet, "packet_index"),
        "packet_index",
    )
    original_appid = _public_integer(
        _required_public_attribute(packet, "original_appid"),
        "original_appid",
        maximum=0x7FF,
    )
    normalized_appid = _public_integer(
        _required_public_attribute(packet, "appid"),
        "appid",
        maximum=0x7FF,
    )
    expected_appid = decoder.normalize_dcb_appid(original_appid)
    if type(expected_appid) is not int or expected_appid != normalized_appid:
        raise IncompatibleUncraterError(
            "uncrater original and normalized AppID fields disagree"
        )
    schema_id = _public_integer(
        _required_public_attribute(packet, "schema_id"),
        "schema_id",
        maximum=0xFFFF,
    )
    issues = _public_decode_issues(
        _required_public_attribute(packet, "decode_status"),
        status_module,
    )
    return _PacketContract(
        packet_index=packet_index,
        original_appid=original_appid,
        normalized_appid=normalized_appid,
        schema_id=schema_id,
        issues=issues,
    )


def _packet_binding_matches(
    packet: Any,
    contract: _PacketContract,
    binding: UncraterBindingInfo,
) -> None:
    selected_schema_id = _public_integer(
        binding.selected_schema_id,
        "selected_schema_id",
        maximum=0xFFFF,
    )
    if contract.schema_id != selected_schema_id:
        raise IncompatibleUncraterError(
            "uncrater packet schema and Collection binding disagree"
        )
    binding_key = binding.binding_key
    if not isinstance(binding_key, str) or not binding_key:
        raise IncompatibleUncraterError(
            "uncrater Collection binding key must be a nonempty string"
        )
    schema = _required_public_attribute(packet, "schema")
    packet_binding_key = _required_public_attribute(schema, "binding_key")
    if not isinstance(packet_binding_key, str) or not packet_binding_key:
        raise IncompatibleUncraterError(
            "uncrater packet schema binding_key must be a nonempty string"
        )
    if packet_binding_key != binding_key:
        raise IncompatibleUncraterError(
            "uncrater packet and Collection binding keys disagree"
        )

    reported_ids = binding.reported_schema_ids
    if not isinstance(reported_ids, tuple) or len(reported_ids) > 1:
        raise IncompatibleUncraterError(
            "uncrater Collection must report at most one schema ID"
        )
    for reported_id in reported_ids:
        _public_integer(reported_id, "reported_schema_id", maximum=0xFFFF)
    reported_version = _required_public_attribute(packet, "reported_version")
    if reported_version is not None:
        reported_version = _public_integer(
            reported_version,
            "reported_version",
            maximum=0xFFFF,
        )
    expected_reported_version = reported_ids[0] if reported_ids else None
    if reported_version != expected_reported_version:
        raise IncompatibleUncraterError(
            "uncrater packet and Collection reported schema IDs disagree"
        )

    schema_assumed = _required_public_attribute(packet, "schema_assumed")
    if (
        type(schema_assumed) is not bool
        or type(binding.schema_assumed) is not bool
    ):
        raise IncompatibleUncraterError(
            "uncrater schema_assumed fields must be boolean"
        )
    if schema_assumed != binding.schema_assumed:
        raise IncompatibleUncraterError(
            "uncrater packet and Collection schema_assumed fields disagree"
        )

    provenance = _required_public_attribute(packet, "binding_provenance")
    if not isinstance(provenance, Mapping):
        raise IncompatibleUncraterError(
            "uncrater packet binding_provenance must be a mapping"
        )
    expected_fields = {
        "binding_key": binding_key,
        "canonical_schema_id": selected_schema_id,
        "variant": binding.variant,
        "source_release": binding.source_release,
        "source_commit": binding.source_commit,
    }
    for name, expected in expected_fields.items():
        if name not in provenance or provenance[name] != expected:
            raise IncompatibleUncraterError(
                f"uncrater packet binding_provenance disagrees on {name}"
            )
    provenance_schema_id = _public_integer(
        provenance["canonical_schema_id"],
        "binding_provenance.canonical_schema_id",
        maximum=0xFFFF,
    )
    if provenance_schema_id != contract.schema_id:
        raise IncompatibleUncraterError(
            "uncrater packet schema and binding_provenance disagree"
        )
    abi = provenance.get("abi")
    if not isinstance(abi, Mapping) or abi.get("sha256") != binding.abi_fingerprint:
        raise IncompatibleUncraterError(
            "uncrater packet binding_provenance ABI fingerprint disagrees"
        )


@dataclass(frozen=True)
class _IssueContract:
    code: str
    message: str
    fatal: bool
    appid: int | None
    source: str | None
    packet_index: int | None
    details: dict[str, object]

    def signature(self) -> tuple[object, ...]:
        """Return the public fields that identify one issue occurrence."""
        return (
            self.code,
            self.message,
            self.fatal,
            self.appid,
            self.source,
            json.dumps(
                self.details,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ),
        )


def _issue_contract(
    issue: Any,
    *,
    owner: _PacketContract | None,
) -> _IssueContract:
    code = _required_public_attribute(issue, "code")
    if (
        not isinstance(code, str)
        or not code
        or not code.replace("_", "a").isalnum()
        or not code[0].isalpha()
        or code.lower() != code
    ):
        raise IncompatibleUncraterError(
            "uncrater DecodeIssue.code must be a lowercase identifier"
        )
    message = _required_public_attribute(issue, "message")
    if not isinstance(message, str) or not message:
        raise IncompatibleUncraterError(
            "uncrater DecodeIssue.message must be a nonempty string"
        )
    fatal = _required_public_attribute(issue, "fatal")
    if type(fatal) is not bool:
        raise IncompatibleUncraterError(
            "uncrater DecodeIssue.fatal must be a boolean"
        )
    appid = _required_public_attribute(issue, "appid")
    if appid is not None:
        appid = _public_integer(appid, "DecodeIssue.appid", maximum=0x7FF)
    if owner is not None:
        if appid is None:
            raise IncompatibleUncraterError(
                "uncrater packet DecodeIssue has no public AppID"
            )
        if appid != owner.normalized_appid:
            raise IncompatibleUncraterError(
                "uncrater packet and DecodeIssue AppIDs disagree"
            )
    source = _required_public_attribute(issue, "source")
    if source is not None and (
        not isinstance(source, str)
        or not source
        or Path(source).name != source
    ):
        raise IncompatibleUncraterError(
            "uncrater DecodeIssue.source must be a basename or None"
        )
    raw_details = _required_public_attribute(issue, "details")
    if not isinstance(raw_details, tuple):
        raise IncompatibleUncraterError(
            "uncrater DecodeIssue.details must be an immutable tuple"
        )
    details: dict[str, object] = {}
    for item in raw_details:
        if not isinstance(item, tuple) or len(item) != 2:
            raise IncompatibleUncraterError(
                "uncrater DecodeIssue.details must contain key/value pairs"
            )
        key, value = item
        if not isinstance(key, str) or not key or key in details:
            raise IncompatibleUncraterError(
                "uncrater DecodeIssue.details has an invalid or duplicate key"
            )
        details[key] = value
        _validate_public_detail(value)

    packet_index = None if owner is None else owner.packet_index
    if "packet_index" in details:
        detail_packet_index = _public_integer(
            details["packet_index"],
            "DecodeIssue.details.packet_index",
        )
        if packet_index is not None and detail_packet_index != packet_index:
            raise IncompatibleUncraterError(
                "uncrater packet and DecodeIssue packet indices disagree"
            )
        packet_index = detail_packet_index
    return _IssueContract(
        code=code,
        message=message,
        fatal=fatal,
        appid=appid,
        source=source,
        packet_index=packet_index,
        details=details,
    )


def import_decode_issues(
    collection: Any,
    issue_collector: IssueCollector,
    selected_binding: UncraterBindingInfo,
) -> ImportedDecodeIssues:
    """Import every public uncrater issue exactly once into one collector."""
    if not isinstance(issue_collector, IssueCollector):
        raise TypeError("issue_collector must be an IssueCollector")
    if not isinstance(selected_binding, UncraterBindingInfo):
        raise TypeError("selected_binding must be an UncraterBindingInfo")
    decoder = load_uncrater()
    status_module = _required_submodule("uncrater.decode_status")
    if not isinstance(collection, decoder.Collection):
        raise IncompatibleUncraterError(
            "uncrater collection is not a public Collection"
        )
    packets = _required_public_attribute(collection, "cont")
    if not isinstance(packets, (list, tuple)):
        raise IncompatibleUncraterError(
            "uncrater Collection.cont must be a packet sequence"
        )
    packet_contracts = tuple(
        _packet_contract(packet, decoder, status_module) for packet in packets
    )
    for packet, contract in zip(packets, packet_contracts):
        _packet_binding_matches(packet, contract, selected_binding)
    collection_issues = _public_decode_issues(
        _required_public_attribute(collection, "decode_status"),
        status_module,
    )

    packet_items = [
        _issue_contract(issue, owner=packet)
        for packet in packet_contracts
        for issue in packet.issues
    ]
    collection_items = [
        _issue_contract(issue, owner=None) for issue in collection_issues
    ]
    packet_signatures = [item.signature() for item in packet_items]
    collection_signatures = [item.signature() for item in collection_items]
    aggregate_start = len(collection_items) - len(packet_items)
    collection_has_packet_aggregate = bool(packet_items) and (
        aggregate_start >= 0
        and collection_signatures[aggregate_start:] == packet_signatures
    )
    if collection_has_packet_aggregate:
        ordered = collection_items[:aggregate_start] + packet_items
    else:
        overlap = set(collection_signatures).intersection(packet_signatures)
        if overlap:
            raise IncompatibleUncraterError(
                "uncrater Collection and packet issue aggregation disagree"
            )
        ordered = collection_items + packet_items

    marker = issue_collector.mark()
    issue_ids: dict[int, list[str]] = {}
    for item in ordered:
        imported = issue_collector.record(
            code=f"decode.{item.code}",
            severity=(
                IssueSeverity.ERROR if item.fatal else IssueSeverity.WARNING
            ),
            stage="decode",
            message=item.message,
            action=(IssueAction.DROPPED if item.fatal else IssueAction.KEPT),
            input_identity=item.source,
            packet_index=item.packet_index,
            appid=item.appid,
            details=item.details,
        )
        if item.packet_index is not None:
            issue_ids.setdefault(item.packet_index, []).append(imported.issue_id)
    return ImportedDecodeIssues(
        issue_ids_by_packet_index={
            packet_index: tuple(ids)
            for packet_index, ids in sorted(issue_ids.items())
        },
        issues=issue_collector.since(marker),
    )


def source_packet_provenance(
    packet: Any,
    *,
    role: str,
    binding: UncraterBindingInfo,
) -> SourcePacketProvenance:
    """Build one concrete source record from reviewed public packet fields."""
    if not isinstance(binding, UncraterBindingInfo):
        raise TypeError("binding must be an UncraterBindingInfo")
    decoder = load_uncrater()
    status_module = _required_submodule("uncrater.decode_status")
    contract = _packet_contract(packet, decoder, status_module)
    for issue in contract.issues:
        _issue_contract(issue, owner=contract)
    _packet_binding_matches(packet, contract, binding)
    return SourcePacketProvenance(
        role=role,
        filename=None,
        packet_index=contract.packet_index,
        original_appid=contract.original_appid,
        normalized_appid=contract.normalized_appid,
    )


def binding_info(collection: Any) -> UncraterBindingInfo:
    """Extract immutable binding, AppID-count, and issue-count provenance."""
    load_uncrater()
    registry = _required_submodule("uncrater.schema_registry")
    binding_keys = tuple(getattr(collection, "selected_schema_bindings", ()))
    selected_ids = tuple(
        int(value) for value in getattr(collection, "selected_schema_ids", ())
    )
    if len(binding_keys) != 1 or len(selected_ids) != 1:
        raise IncompatibleUncraterError(
            "uncrater Collection must select exactly one schema binding"
        )
    binding = registry.binding_for_key(binding_keys[0])
    if int(binding.canonical_schema_id) != selected_ids[0]:
        raise IncompatibleUncraterError(
            "uncrater Collection binding key and selected schema ID disagree"
        )

    appid_counts = tuple(
        sorted(
            (int(appid), int(count))
            for appid, count in getattr(
                collection, "packet_counts_by_appid", {}
            ).items()
        )
    )
    status = getattr(collection, "decode_status", None)
    counts = status.counts() if status is not None else {}
    return UncraterBindingInfo(
        reported_schema_ids=tuple(
            int(value) for value in getattr(collection, "reported_schema_ids", ())
        ),
        selected_schema_id=selected_ids[0],
        binding_key=str(binding.binding_key),
        variant=binding.variant,
        schema_assumed=bool(getattr(collection, "schema_assumed", False)),
        source_release=str(binding.source_release),
        source_commit=str(binding.source_commit),
        abi_fingerprint=str(binding.abi_fingerprint),
        appid_counts=appid_counts,
        issue_counts=tuple(
            sorted((str(code), int(count)) for code, count in counts.items())
        ),
    )


def collection_provenance(
    collection: Any,
    *,
    strict: bool,
) -> DecodeProvenance:
    """Build immutable provenance from one Collection's public report APIs."""
    decoder = decoder_info()
    binding = binding_info(collection)
    packets = tuple(getattr(collection, "cont", ()))
    input_packet_count = sum(count for _, count in binding.appid_counts)
    if len(packets) != input_packet_count:
        raise IncompatibleUncraterError(
            "uncrater packet list and AppID counts disagree"
        )
    valid_packet_count = 0
    for packet in packets:
        status = getattr(packet, "decode_status", None)
        if status is None or not hasattr(status, "issues"):
            raise IncompatibleUncraterError(
                "uncrater packet has no public decode_status.issues"
            )
        issues = tuple(status.issues)
        for issue in issues:
            if type(getattr(issue, "fatal", None)) is not bool:
                raise IncompatibleUncraterError(
                    "uncrater decode issue has no boolean fatal field"
                )
        valid_packet_count += not any(issue.fatal for issue in issues)
    report = collection.canonical_report()
    if not isinstance(report, dict):
        raise IncompatibleUncraterError(
            "uncrater Collection canonical_report() must return a dictionary"
        )
    expected_report_fields = {
        "report_schema_version": 1,
        "reported_schema_ids": [
            f"0x{value:03X}" for value in binding.reported_schema_ids
        ],
        "selected_schema_ids": [f"0x{binding.selected_schema_id:03X}"],
        "selected_schema_bindings": [binding.binding_key],
        "schema_assumed": binding.schema_assumed,
        "packet_count": input_packet_count,
        "packet_counts_by_appid": {
            f"0x{appid:03X}": count for appid, count in binding.appid_counts
        },
        "invalid_counts_by_issue": dict(binding.issue_counts),
    }
    for name, expected in expected_report_fields.items():
        if report.get(name) != expected:
            raise IncompatibleUncraterError(
                f"uncrater canonical report disagrees on {name}"
            )
    return DecodeProvenance.from_report(
        distribution_version=decoder.distribution_version,
        decoder_source_commit=decoder.source_commit,
        reported_schema_ids=binding.reported_schema_ids,
        selected_schema_id=binding.selected_schema_id,
        binding_key=binding.binding_key,
        schema_variant=binding.variant,
        schema_assumed=binding.schema_assumed,
        binding_source_release=binding.source_release,
        binding_source_commit=binding.source_commit,
        abi_fingerprint=binding.abi_fingerprint,
        execution_mode=(
            ExecutionMode.STRICT if strict else ExecutionMode.COLLECT
        ),
        input_packet_count=input_packet_count,
        valid_packet_count=valid_packet_count,
        appid_counts=binding.appid_counts,
        issue_counts=binding.issue_counts,
        canonical_report=report,
    )
