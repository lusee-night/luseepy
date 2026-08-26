"""Versioned provenance map for extracted uncrater packet files."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .reassembly import LogicalPacket


PACKET_MAP_FILENAME = "packet_map.json"
PACKET_MAP_FORMAT_VERSION = 1

_PACKET_FILENAME_RE = re.compile(
    r"^(?P<index>[0-9]+)_(?P<appid>[0-9a-f]{4})\.bin$"
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_TOP_LEVEL_KEYS = {
    "format_version",
    "packet_order",
    "packets",
    "provenance_limits",
    "reassembly_profile",
}
_ENTRY_KEYS = {
    "content_sha256",
    "last_sequence_count",
    "normalized_appid",
    "original_appid",
    "output_filename",
    "output_index",
    "source_bank",
    "start_sequence_count",
    "terminal_groupflag",
    "unavailable_fields",
    "unique_packet_id",
}
_PACKET_ORDER = MappingProxyType({
    "chronological": False,
    "key": ("unique_packet_id", "last_sequence_count"),
    "kind": "uid_sequence_heuristic",
})
_PROVENANCE_LIMITS = MappingProxyType({
    "contributing_frame_byte_offsets": (
        "not_retained_by_production_reassembly"
    ),
    "contributing_frame_flags": "only_terminal_groupflag_is_retained",
    "contributing_frame_ordinals": "not_retained_by_production_reassembly",
    "packet_issue_references": (
        "packet_linked_reassembly_issues_are_not_retained"
    ),
    "pre_sort_packet_ordinal": "not_retained_by_existing_pipeline",
    "uid_source": "identity_assignment_source_is_not_retained",
})
_OPTIONAL_FIELD_REASONS = {
    "source_bank": "logical_packet_bank_is_not_recorded",
    "unique_packet_id": "logical_packet_uid_is_not_recorded",
}


class PacketMapError(ValueError):
    """A present packet map is malformed or disagrees with its packet tree."""


@dataclass(frozen=True, slots=True)
class PacketMapEntry:
    """One output packet and the provenance retained at session persistence."""

    output_filename: str
    content_sha256: str
    output_index: int
    source_bank: str | None
    original_appid: int
    normalized_appid: int
    unique_packet_id: int | None
    start_sequence_count: int
    last_sequence_count: int
    terminal_groupflag: int
    unavailable_fields: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.output_filename, str):
            raise PacketMapError("output_filename must be a string")
        match = _PACKET_FILENAME_RE.fullmatch(self.output_filename)
        if match is None:
            raise PacketMapError("packet map has an invalid output filename")
        _require_int(self.output_index, "output_index", minimum=0)
        if int(match.group("index")) != self.output_index:
            raise PacketMapError("packet filename and output index disagree")
        _require_appid(self.original_appid, "original_appid")
        _require_appid(self.normalized_appid, "normalized_appid")
        if int(match.group("appid"), 16) != self.original_appid:
            raise PacketMapError("packet filename and original AppID disagree")
        if not isinstance(self.content_sha256, str) or not _SHA256_RE.fullmatch(
            self.content_sha256
        ):
            raise PacketMapError("content_sha256 must be lowercase 64-hex")
        if self.source_bank is not None and (
            not isinstance(self.source_bank, str) or not self.source_bank
        ):
            raise PacketMapError("source_bank must be a nonempty string or null")
        if self.unique_packet_id is not None:
            _require_int(
                self.unique_packet_id,
                "unique_packet_id",
                minimum=0,
                maximum=0xFFFFFFFF,
            )
        _require_int(
            self.start_sequence_count,
            "start_sequence_count",
            minimum=0,
            maximum=0x3FFF,
        )
        _require_int(
            self.last_sequence_count,
            "last_sequence_count",
            minimum=0,
            maximum=0x3FFF,
        )
        if type(self.terminal_groupflag) is not int or (
            self.terminal_groupflag not in (1, 3)
        ):
            raise PacketMapError("terminal_groupflag must be 1 or 3")
        unavailable = tuple(self.unavailable_fields)
        expected = tuple(
            (name, reason)
            for name, reason in sorted(_OPTIONAL_FIELD_REASONS.items())
            if getattr(self, name) is None
        )
        if unavailable != expected:
            raise PacketMapError("unavailable_fields disagree with null fields")
        object.__setattr__(self, "unavailable_fields", unavailable)

    def as_dict(self) -> dict[str, object]:
        return {
            "content_sha256": self.content_sha256,
            "last_sequence_count": self.last_sequence_count,
            "normalized_appid": self.normalized_appid,
            "original_appid": self.original_appid,
            "output_filename": self.output_filename,
            "output_index": self.output_index,
            "source_bank": self.source_bank,
            "start_sequence_count": self.start_sequence_count,
            "terminal_groupflag": self.terminal_groupflag,
            "unavailable_fields": dict(self.unavailable_fields),
            "unique_packet_id": self.unique_packet_id,
        }


@dataclass(frozen=True, slots=True)
class PacketMap:
    """Validated format-1 packet map in deterministic output order."""

    entries: tuple[PacketMapEntry, ...]

    def __post_init__(self) -> None:
        entries = tuple(self.entries)
        for index, entry in enumerate(entries):
            if not isinstance(entry, PacketMapEntry):
                raise TypeError("packet map entries must be PacketMapEntry records")
            if entry.output_index != index:
                raise PacketMapError("packet map entries are not in output order")
        object.__setattr__(self, "entries", entries)

    @property
    def by_output_index(self) -> Mapping[int, PacketMapEntry]:
        return MappingProxyType({entry.output_index: entry for entry in self.entries})

    def as_dict(self) -> dict[str, object]:
        return {
            "format_version": PACKET_MAP_FORMAT_VERSION,
            "packet_order": {
                "chronological": _PACKET_ORDER["chronological"],
                "key": list(_PACKET_ORDER["key"]),
                "kind": _PACKET_ORDER["kind"],
            },
            "packets": [entry.as_dict() for entry in self.entries],
            "provenance_limits": dict(_PROVENANCE_LIMITS),
            "reassembly_profile": "legacy",
        }


def _require_int(
    value: object,
    name: str,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    if type(value) is not int or value < minimum:
        raise PacketMapError(f"{name} must be an integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise PacketMapError(f"{name} must be <= {maximum}")
    return value


def _require_appid(value: object, name: str) -> int:
    return _require_int(value, name, minimum=0, maximum=0x7FF)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _object_without_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    value = {}
    for key, item in pairs:
        if key in value:
            raise PacketMapError(f"packet map contains duplicate key {key!r}")
        value[key] = item
    return value


def build_packet_map(
    packets: Sequence[LogicalPacket],
    filenames: Sequence[str],
    *,
    normalize_appid: Callable[[int], int],
) -> PacketMap:
    """Build a map without changing packet order or identity assignment."""
    if len(packets) != len(filenames):
        raise ValueError("packets and filenames must have the same length")
    entries = []
    for index, (packet, filename) in enumerate(zip(packets, filenames)):
        original_appid = _require_appid(packet.appid, "original_appid")
        normalized_appid = normalize_appid(original_appid)
        _require_appid(normalized_appid, "normalized_appid")
        unavailable = tuple(
            (name, reason)
            for name, reason in sorted(_OPTIONAL_FIELD_REASONS.items())
            if getattr(packet, "bank" if name == "source_bank" else name) is None
        )
        entries.append(PacketMapEntry(
            output_filename=filename,
            content_sha256=_sha256_bytes(packet.blob),
            output_index=index,
            source_bank=packet.bank,
            original_appid=original_appid,
            normalized_appid=normalized_appid,
            unique_packet_id=packet.unique_packet_id,
            start_sequence_count=packet.start_seq,
            last_sequence_count=packet.seq,
            terminal_groupflag=3 if packet.single_packet else 1,
            unavailable_fields=unavailable,
        ))
    return PacketMap(tuple(entries))


def write_packet_map(packet_map: PacketMap, path: Path | str) -> Path:
    """Write deterministic ASCII JSON beside an extracted session tree."""
    if not isinstance(packet_map, PacketMap):
        raise TypeError("packet_map must be a PacketMap")
    destination = Path(path)
    body = json.dumps(
        packet_map.as_dict(),
        allow_nan=False,
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    ) + "\n"
    with destination.open("w", encoding="ascii", newline="\n") as output:
        output.write(body)
    return destination


def read_packet_map(
    session_dir: Path | str,
    cdi_dir: Path | str,
    *,
    normalize_appid: Callable[[int], int],
) -> PacketMap | None:
    """Load and verify a present map; return None for a legacy session."""
    path = Path(session_dir) / PACKET_MAP_FILENAME
    if path.is_symlink():
        raise PacketMapError("packet_map.json must be a regular file")
    if not path.exists():
        return None
    if not path.is_file():
        raise PacketMapError("packet_map.json must be a regular file")
    try:
        with path.open("r", encoding="ascii") as source:
            document = json.load(
                source,
                object_pairs_hook=_object_without_duplicate_keys,
            )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PacketMapError(f"cannot read packet_map.json: {exc}") from exc
    packet_map = _parse_packet_map(document, normalize_appid=normalize_appid)
    _validate_packet_inventory(packet_map, Path(cdi_dir))
    return packet_map


def _parse_packet_map(
    document: object,
    *,
    normalize_appid: Callable[[int], int],
) -> PacketMap:
    if not isinstance(document, dict) or set(document) != _TOP_LEVEL_KEYS:
        raise PacketMapError("packet map has an invalid top-level schema")
    if type(document["format_version"]) is not int or (
        document["format_version"] != PACKET_MAP_FORMAT_VERSION
    ):
        raise PacketMapError("unsupported packet map format_version")
    if document["reassembly_profile"] != "legacy":
        raise PacketMapError("packet map reassembly_profile must be legacy")
    expected_order = {
        "chronological": _PACKET_ORDER["chronological"],
        "key": list(_PACKET_ORDER["key"]),
        "kind": _PACKET_ORDER["kind"],
    }
    if document["packet_order"] != expected_order:
        raise PacketMapError("packet map has an invalid packet_order contract")
    if document["provenance_limits"] != dict(_PROVENANCE_LIMITS):
        raise PacketMapError("packet map has an invalid provenance_limits contract")
    raw_entries = document["packets"]
    if not isinstance(raw_entries, list):
        raise PacketMapError("packet map packets must be a list")
    entries = []
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, dict) or set(raw_entry) != _ENTRY_KEYS:
            raise PacketMapError("packet map entry has an invalid schema")
        unavailable = raw_entry["unavailable_fields"]
        if not isinstance(unavailable, dict) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in unavailable.items()
        ):
            raise PacketMapError("unavailable_fields must be a string mapping")
        entry = PacketMapEntry(
            output_filename=raw_entry["output_filename"],
            content_sha256=raw_entry["content_sha256"],
            output_index=raw_entry["output_index"],
            source_bank=raw_entry["source_bank"],
            original_appid=raw_entry["original_appid"],
            normalized_appid=raw_entry["normalized_appid"],
            unique_packet_id=raw_entry["unique_packet_id"],
            start_sequence_count=raw_entry["start_sequence_count"],
            last_sequence_count=raw_entry["last_sequence_count"],
            terminal_groupflag=raw_entry["terminal_groupflag"],
            unavailable_fields=tuple(sorted(unavailable.items())),
        )
        expected_appid = normalize_appid(entry.original_appid)
        if type(expected_appid) is not int or expected_appid != entry.normalized_appid:
            raise PacketMapError("packet map normalized AppID disagrees with uncrater")
        entries.append(entry)
    return PacketMap(tuple(entries))


def _validate_packet_inventory(packet_map: PacketMap, cdi_dir: Path) -> None:
    actual = sorted(
        path for path in cdi_dir.iterdir()
        if path.suffix == ".bin"
    )
    if any(path.is_symlink() or not path.is_file() for path in actual):
        raise PacketMapError("cdi_output .bin entries must be regular files")
    expected_names = [entry.output_filename for entry in packet_map.entries]
    if [path.name for path in actual] != expected_names:
        raise PacketMapError("packet map and cdi_output .bin inventory disagree")
    for path, entry in zip(actual, packet_map.entries):
        if _sha256_file(path) != entry.content_sha256:
            raise PacketMapError(
                f"packet content hash mismatch for {entry.output_filename}"
            )
