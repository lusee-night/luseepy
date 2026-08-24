"""Single compatibility and provenance boundary for the uncrater decoder."""

from __future__ import annotations

import importlib
import inspect
import json
from dataclasses import dataclass
from functools import lru_cache
from importlib import metadata
from pathlib import Path
from types import ModuleType
from typing import Any

from .dependencies import import_optional_dependency
from .products import DecodeProvenance, ExecutionMode


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


_REQUIRED_TOP_LEVEL = (
    "Packet",
    "PacketBase",
    "Collection",
    "id",
    "normalize_dcb_appid",
    "appid_is_cal_metadata",
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
    if not callable(getattr(decoder.PacketBase, "read", None)):
        raise IncompatibleUncraterError(
            "installed uncrater PacketBase has no public read() method"
        )
    parameters = inspect.signature(decoder.Collection).parameters
    missing_parameters = [
        name
        for name in ("strict", "diagnostic_override", "schema_variant")
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
) -> Any:
    """Construct a Collection using only reviewed public options."""
    decoder = load_uncrater()
    return decoder.Collection(
        str(path),
        strict=strict,
        diagnostic_override=diagnostic_override,
        schema_variant=schema_variant,
    )


def read_packet(packet: Any) -> None:
    """Decode one packet through uncrater's public idempotent method."""
    read = getattr(packet, "read", None)
    if not callable(read):
        raise IncompatibleUncraterError(
            f"uncrater packet {type(packet).__name__} has no public read() method"
        )
    read()


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
