"""Strict readers and schema primitives for ingest layout v4 files."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from .clock_reference import ClockSource, clock_reference_set_from_record
from .constants import INGEST_LAYOUT_VERSION
from .dependencies import import_optional_dependency
from .layout_v4_tree import LayoutDataset, LayoutGroup
from .telemetry import TELEMETRY_FIELD_COUNT, TelemetryData
from .write_request import LunarLocation


class LayoutV4ValidationError(ValueError):
    """A layout-v4 file or logical tree violates its stored contract."""


class LayoutV4Validator:
    """Exact path, dtype, shape, attribute, and row-alignment checks."""

    def __init__(self, root: LayoutGroup):
        if not isinstance(root, LayoutGroup) or root.path != "":
            raise LayoutV4ValidationError("layout-v4 root group is invalid")
        self.root = root

    def node(self, path: str) -> LayoutGroup | LayoutDataset:
        """Return one absolute-path node or raise a validation error."""
        parts = self._path_parts(path)
        current: LayoutGroup | LayoutDataset = self.root
        for part in parts:
            if not isinstance(current, LayoutGroup) or part not in current.children:
                raise LayoutV4ValidationError(
                    f"layout-v4 path is missing: {path}"
                )
            current = current.children[part]
        return current

    def has(self, path: str) -> bool:
        """Return whether one absolute path exists in the logical tree."""
        try:
            self.node(path)
        except LayoutV4ValidationError:
            return False
        return True

    def group(self, path: str) -> LayoutGroup:
        """Require and return one group."""
        node = self.node(path)
        if not isinstance(node, LayoutGroup):
            raise LayoutV4ValidationError(
                f"layout-v4 path is not a group: {path}"
            )
        return node

    def dataset(
        self,
        path: str,
        *,
        dtype: object | None = None,
        utf8: bool | None = None,
        shape: tuple[int, ...] | None = None,
        ndim: int | None = None,
        row_count: int | None = None,
        tail_shape: tuple[int, ...] | None = None,
    ) -> LayoutDataset:
        """Require one dataset and any supplied exact logical contract."""
        node = self.node(path)
        if not isinstance(node, LayoutDataset):
            raise LayoutV4ValidationError(
                f"layout-v4 path is not a dataset: {path}"
            )
        if dtype is not None and node.dtype != np.dtype(dtype):
            raise LayoutV4ValidationError(
                f"layout-v4 dataset {path} has dtype {node.dtype}, "
                f"expected {np.dtype(dtype)}"
            )
        if utf8 is not None and node.is_utf8 is not utf8:
            raise LayoutV4ValidationError(
                f"layout-v4 dataset {path} UTF-8 contract disagrees"
            )
        if shape is not None and node.shape != shape:
            raise LayoutV4ValidationError(
                f"layout-v4 dataset {path} has shape {node.shape}, "
                f"expected {shape}"
            )
        if ndim is not None and node.data.ndim != ndim:
            raise LayoutV4ValidationError(
                f"layout-v4 dataset {path} has rank {node.data.ndim}, "
                f"expected {ndim}"
            )
        if row_count is not None and (
            node.data.ndim == 0 or node.shape[0] != row_count
        ):
            raise LayoutV4ValidationError(
                f"layout-v4 dataset {path} is not aligned to {row_count} rows"
            )
        if tail_shape is not None and (
            node.data.ndim == 0 or node.shape[1:] != tail_shape
        ):
            raise LayoutV4ValidationError(
                f"layout-v4 dataset {path} has tail shape "
                f"{node.shape[1:] if node.data.ndim else ()}, expected {tail_shape}"
            )
        return node

    def attribute(
        self,
        path: str,
        name: str,
        *,
        dtype: object | None = None,
    ) -> object:
        """Require one attribute and optionally its exact NumPy dtype."""
        node = self.node(path)
        attrs = node.attrs
        if name not in attrs:
            raise LayoutV4ValidationError(
                f"layout-v4 attribute is missing: {path}@{name}"
            )
        value = attrs[name]
        if dtype is not None:
            array = np.asarray(value)
            if array.shape != () or array.dtype != np.dtype(dtype):
                raise LayoutV4ValidationError(
                    f"layout-v4 attribute {path}@{name} has contract "
                    f"{array.shape}/{array.dtype}, expected scalar/{np.dtype(dtype)}"
                )
        return value

    def require_row_aligned(
        self,
        group_path: str,
        names: Iterable[str],
        *,
        row_count: int | None = None,
    ) -> int:
        """Require named direct datasets to share one first-axis length."""
        names = tuple(names)
        if not names:
            raise ValueError("row-alignment validation needs at least one dataset")
        group = self.group(group_path)
        observed = row_count
        for name in names:
            if "/" in name or not name:
                raise ValueError("row-aligned dataset names must be direct children")
            child = group.children.get(name)
            if not isinstance(child, LayoutDataset) or child.data.ndim == 0:
                raise LayoutV4ValidationError(
                    f"layout-v4 row dataset is missing: {group_path}/{name}"
                )
            if observed is None:
                observed = child.shape[0]
            elif child.shape[0] != observed:
                raise LayoutV4ValidationError(
                    f"layout-v4 datasets in {group_path} do not share a row count"
                )
        assert observed is not None
        return observed

    @staticmethod
    def _path_parts(path: str) -> tuple[str, ...]:
        if path == "/":
            return ()
        if not isinstance(path, str) or not path.startswith("/"):
            raise ValueError("layout-v4 paths must be absolute")
        parts = tuple(path[1:].split("/"))
        if any(not part or part in (".", "..") for part in parts):
            raise ValueError(f"invalid layout-v4 path: {path!r}")
        return parts


def _scalar_attr(
    validator: LayoutV4Validator,
    path: str,
    name: str,
    dtype: object,
) -> object:
    return validator.attribute(path, name, dtype=dtype)


def _string_attr(validator: LayoutV4Validator, path: str, name: str) -> str:
    value = validator.attribute(path, name)
    if type(value) is not str or not value:
        raise LayoutV4ValidationError(
            f"layout-v4 attribute {path}@{name} must be a nonempty string"
        )
    return value


def _bool_attr(validator: LayoutV4Validator, path: str, name: str) -> bool:
    return bool(_scalar_attr(validator, path, name, np.bool_))


def _uint64_attr(validator: LayoutV4Validator, path: str, name: str) -> int:
    return int(_scalar_attr(validator, path, name, np.uint64))


def _validate_tree_structure(root: LayoutGroup) -> None:
    seen: set[int] = set()

    def visit(group: LayoutGroup, expected_path: str) -> None:
        identity = id(group)
        if identity in seen:
            raise LayoutV4ValidationError("layout-v4 tree contains a cycle or alias")
        seen.add(identity)
        if group.path != expected_path:
            raise LayoutV4ValidationError(
                f"layout-v4 group path {group.path!r} disagrees with its tree path"
            )
        _validate_attributes(group.attrs, expected_path or "/")
        for name, child in group.children.items():
            if type(name) is not str or not name or "/" in name:
                raise LayoutV4ValidationError(
                    f"layout-v4 child name is invalid in {expected_path or '/'}"
                )
            child_path = f"{expected_path}/{name}" if expected_path else f"/{name}"
            child_identity = id(child)
            if isinstance(child, LayoutGroup):
                visit(child, child_path)
            elif isinstance(child, LayoutDataset):
                if child_identity in seen:
                    raise LayoutV4ValidationError(
                        "layout-v4 tree contains a cycle or alias"
                    )
                seen.add(child_identity)
                _validate_dataset(child, child_path)
            else:
                raise LayoutV4ValidationError(
                    f"layout-v4 child has an unsupported type at {child_path}"
                )

    visit(root, "")


def _validate_attributes(attrs: dict[str, object], path: str) -> None:
    for name, value in attrs.items():
        if type(name) is not str or not name:
            raise LayoutV4ValidationError(
                f"layout-v4 attribute name is invalid at {path}"
            )
        if type(value) is str:
            if "\x00" in value:
                raise LayoutV4ValidationError(
                    f"layout-v4 attribute {path}@{name} contains a null"
                )
            continue
        array = np.asarray(value)
        if array.dtype.hasobject or array.dtype.metadata is not None:
            raise LayoutV4ValidationError(
                f"layout-v4 attribute {path}@{name} has an unsupported dtype"
            )
        if not array.dtype.isnative or not _portable_dtype(array.dtype, allow_unicode=True):
            raise LayoutV4ValidationError(
                f"layout-v4 attribute {path}@{name} has nonportable dtype "
                f"{array.dtype}"
            )
        if array.dtype.kind == "U" and any(
            "\x00" in item for item in array.reshape(-1).tolist()
        ):
            raise LayoutV4ValidationError(
                f"layout-v4 attribute {path}@{name} contains a null"
            )


def _validate_dataset(dataset: LayoutDataset, path: str) -> None:
    if type(dataset.data) is not np.ndarray:
        raise LayoutV4ValidationError(
            f"layout-v4 dataset {path} is not a NumPy array"
        )
    if dataset.data.ndim == 0 or dataset.data.ndim > 32:
        raise LayoutV4ValidationError(
            f"layout-v4 dataset {path} must have a row axis and rank at most 32"
        )
    if dataset.is_utf8:
        if dataset.dtype != np.dtype(object) or any(
            type(item) is not str or "\x00" in item
            for item in dataset.data.reshape(-1).tolist()
        ):
            raise LayoutV4ValidationError(
                f"layout-v4 UTF-8 dataset {path} is invalid"
            )
    elif (
        dataset.dtype.hasobject
        or dataset.dtype.metadata is not None
        or not dataset.dtype.isnative
        or not _portable_dtype(dataset.dtype, allow_unicode=False)
    ):
        raise LayoutV4ValidationError(
            f"layout-v4 dataset {path} has nonportable dtype {dataset.dtype}"
        )
    _validate_attributes(dataset.attrs, path)


def _portable_dtype(dtype: np.dtype, *, allow_unicode: bool) -> bool:
    return bool(
        dtype.kind == "b"
        or (dtype.kind in "iu" and dtype.itemsize in (1, 2, 4, 8))
        or (dtype.kind == "f" and dtype.itemsize in (4, 8))
        or (dtype.kind == "c" and dtype.itemsize in (8, 16))
        or dtype.kind == "S"
        or (allow_unicode and dtype.kind == "U")
    )


def _validate_optional_text_pair(
    validator: LayoutV4Validator,
    path: str,
    name: str,
) -> bool:
    valid = _bool_attr(validator, path, f"{name}_valid")
    if valid:
        _string_attr(validator, path, name)
    elif name in validator.group(path).attrs:
        raise LayoutV4ValidationError(
            f"layout-v4 optional attribute {path}@{name} is present but invalid"
        )
    return valid


def _validate_common_provenance(validator: LayoutV4Validator) -> None:
    version = int(_scalar_attr(validator, "/", "layout_version", np.uint16))
    if version != INGEST_LAYOUT_VERSION:
        raise LayoutV4ValidationError(
            f"unsupported ingest layout version {version}; expected "
            f"{INGEST_LAYOUT_VERSION}"
        )
    quality = _string_attr(validator, "/", "quality_status")
    if quality not in ("clean", "partial"):
        raise LayoutV4ValidationError("layout-v4 root quality status is invalid")
    execution_mode = _string_attr(validator, "/", "execution_mode")
    if execution_mode not in ("collect", "strict"):
        raise LayoutV4ValidationError("layout-v4 execution mode is invalid")

    issue_count = _uint64_attr(validator, "/", "issue_count")
    severity_total = sum(
        _uint64_attr(validator, "/", f"{severity}_issue_count")
        for severity in ("info", "warning", "error")
    )
    if severity_total != issue_count:
        raise LayoutV4ValidationError("layout-v4 root issue counts disagree")
    input_count = _uint64_attr(validator, "/", "input_packet_count")
    valid_count = _uint64_attr(validator, "/", "valid_packet_count")
    if valid_count > input_count:
        raise LayoutV4ValidationError("layout-v4 packet counts disagree")

    try:
        LunarLocation(
            latitude_deg=float(
                _scalar_attr(
                    validator, "/constants", "lun_lat_deg", np.float64
                )
            ),
            longitude_deg=float(
                _scalar_attr(
                    validator, "/constants", "lun_long_deg", np.float64
                )
            ),
            height_m=float(
                _scalar_attr(
                    validator, "/constants", "lun_height_m", np.float64
                )
            ),
        )
    except (TypeError, ValueError) as exc:
        raise LayoutV4ValidationError(
            f"layout-v4 location contract is invalid: {exc}"
        ) from exc

    run_path = "/run_provenance"
    identity_valid = _validate_optional_text_pair(
        validator, run_path, "input_identity"
    )
    identity_kind_valid = _validate_optional_text_pair(
        validator, run_path, "input_identity_kind"
    )
    unavailable_valid = _validate_optional_text_pair(
        validator, run_path, "input_identity_unavailable_reason"
    )
    if identity_valid == unavailable_valid or identity_valid != identity_kind_valid:
        raise LayoutV4ValidationError(
            "layout-v4 portable input identity provenance is inconsistent"
        )
    if not _validate_optional_text_pair(validator, run_path, "source_kind"):
        raise LayoutV4ValidationError("layout-v4 source kind is unavailable")
    for name in ("source_path", "pipeline_version"):
        _validate_optional_text_pair(validator, run_path, name)
    compression_valid = _validate_optional_text_pair(
        validator, run_path, "hdf5_compression"
    )
    compression_level_valid = _bool_attr(
        validator, run_path, "hdf5_compression_level_valid"
    )
    if compression_valid:
        if (
            _string_attr(validator, run_path, "hdf5_compression") != "gzip"
            or not compression_level_valid
        ):
            raise LayoutV4ValidationError(
                "layout-v4 HDF5 compression contract is invalid"
            )
        compression_level = int(
            _scalar_attr(
                validator,
                run_path,
                "hdf5_compression_level",
                np.int64,
            )
        )
        if not 0 <= compression_level <= 9:
            raise LayoutV4ValidationError(
                "layout-v4 HDF5 compression level is invalid"
            )
    elif (
        compression_level_valid
        or "hdf5_compression_level" in validator.group(run_path).attrs
    ):
        raise LayoutV4ValidationError(
            "layout-v4 HDF5 compression level presence disagrees"
        )
    _bool_attr(validator, run_path, "overwrite_requested")
    _bool_attr(validator, run_path, "destination_preexisted")

    decoder_path = "/provenance/decoder"
    _scalar_attr(validator, decoder_path, "selected_schema_id", np.uint16)
    _bool_attr(validator, decoder_path, "schema_assumed")
    decoder_mode = _string_attr(validator, decoder_path, "execution_mode")
    if decoder_mode != execution_mode:
        raise LayoutV4ValidationError(
            "layout-v4 decoder and root execution modes disagree"
        )
    if (
        _uint64_attr(validator, decoder_path, "input_packet_count") != input_count
        or _uint64_attr(validator, decoder_path, "valid_packet_count")
        != valid_count
    ):
        raise LayoutV4ValidationError(
            "layout-v4 decoder and root packet counts disagree"
        )
    validator.dataset(
        f"{decoder_path}/reported_schema_ids", dtype=np.uint16, utf8=False
    )
    appid_rows = validator.require_row_aligned(
        decoder_path, ("appids", "appid_counts")
    )
    validator.dataset(f"{decoder_path}/appids", dtype=np.uint16, row_count=appid_rows)
    validator.dataset(
        f"{decoder_path}/appid_counts", dtype=np.uint64, row_count=appid_rows
    )
    issue_code_rows = validator.require_row_aligned(
        decoder_path, ("issue_codes", "issue_code_counts")
    )
    validator.dataset(
        f"{decoder_path}/issue_codes",
        utf8=True,
        row_count=issue_code_rows,
    )
    validator.dataset(
        f"{decoder_path}/issue_code_counts",
        dtype=np.uint64,
        row_count=issue_code_rows,
    )

    issue_path = "/issues"
    stored_issue_count = _uint64_attr(validator, issue_path, "count")
    if stored_issue_count != issue_count:
        raise LayoutV4ValidationError("layout-v4 issue row count disagrees")
    issue_group = validator.group(issue_path)
    issue_names = tuple(
        name
        for name, child in issue_group.children.items()
        if isinstance(child, LayoutDataset)
    )
    if not issue_names:
        raise LayoutV4ValidationError("layout-v4 issue schema is empty")
    validator.require_row_aligned(
        issue_path, issue_names, row_count=stored_issue_count
    )

    family_path = "/status/families"
    family_names = (
        "family",
        "supported",
        "coverage",
        "quality",
        "decoded_rows",
        "persisted_rows",
        "reason",
        "reason_valid",
    )
    family_count = validator.require_row_aligned(family_path, family_names)
    for name in ("family", "coverage", "quality", "reason"):
        validator.dataset(
            f"{family_path}/{name}", utf8=True, row_count=family_count
        )
    validator.dataset(
        f"{family_path}/supported", dtype=np.bool_, row_count=family_count
    )
    validator.dataset(
        f"{family_path}/reason_valid", dtype=np.bool_, row_count=family_count
    )
    for name in ("decoded_rows", "persisted_rows"):
        validator.dataset(
            f"{family_path}/{name}", dtype=np.uint64, row_count=family_count
        )


def _validate_telemetry(validator: LayoutV4Validator) -> None:
    path = "/telemetry"
    if not validator.has(path):
        return
    group = validator.group(path)
    expected_children = {
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
    if (
        set(group.attrs) != {"source_kind"}
        or set(group.children) != expected_children
    ):
        raise LayoutV4ValidationError(
            "layout-v4 telemetry tree is not canonical"
        )

    field_names = validator.dataset(
        f"{path}/field_names",
        utf8=True,
        shape=(TELEMETRY_FIELD_COUNT,),
    ).data
    units = validator.dataset(
        f"{path}/units",
        utf8=True,
        shape=(TELEMETRY_FIELD_COUNT,),
    ).data
    mission_seconds = validator.dataset(
        f"{path}/mission_seconds",
        dtype=np.uint32,
        ndim=1,
    ).data
    row_count = mission_seconds.size
    arrays = {
        "source_indices": validator.dataset(
            f"{path}/source_indices",
            dtype=np.int64,
            shape=(row_count,),
        ).data,
        "lusee_subsecs": validator.dataset(
            f"{path}/lusee_subsecs",
            dtype=np.uint16,
            shape=(row_count,),
        ).data,
        "mjd_times": validator.dataset(
            f"{path}/mjd_times",
            dtype=np.float64,
            shape=(row_count,),
        ).data,
        "raw_counts": validator.dataset(
            f"{path}/raw_counts",
            dtype=np.uint16,
            shape=(row_count, TELEMETRY_FIELD_COUNT),
        ).data,
        "values": validator.dataset(
            f"{path}/values",
            dtype=np.float64,
            shape=(row_count, TELEMETRY_FIELD_COUNT),
        ).data,
        "valid": validator.dataset(
            f"{path}/valid",
            dtype=np.bool_,
            shape=(row_count, TELEMETRY_FIELD_COUNT),
        ).data,
    }
    try:
        telemetry = TelemetryData(
            source_kind=_string_attr(validator, path, "source_kind"),
            field_names=tuple(field_names.tolist()),
            units=tuple(units.tolist()),
            source_indices=arrays["source_indices"],
            mission_seconds=mission_seconds,
            lusee_subsecs=arrays["lusee_subsecs"],
            mjd_times=arrays["mjd_times"],
            raw_counts=arrays["raw_counts"],
            values=arrays["values"],
            valid=arrays["valid"],
        )
    except (TypeError, ValueError) as exc:
        raise LayoutV4ValidationError(
            f"layout-v4 telemetry table is invalid: {exc}"
        ) from exc

    clock_path = "/clock_reference"
    reference_set = None
    if _bool_attr(validator, clock_path, "available"):
        try:
            record = json.loads(
                _string_attr(
                    validator,
                    clock_path,
                    "canonical_record_json",
                )
            )
            reference_set = clock_reference_set_from_record(record)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise LayoutV4ValidationError(
                f"layout-v4 clock reference is invalid: {exc}"
            ) from exc
        sources = validator.dataset(
            f"{clock_path}/clock_sources",
            utf8=True,
            ndim=1,
        ).data
        raw_seconds = validator.dataset(
            f"{clock_path}/clock_reference_raw_seconds",
            dtype=np.float64,
            shape=(sources.size,),
        ).data
        if tuple(sources.tolist()) != tuple(
            item.clock_source.value for item in reference_set.clocks
        ) or not np.array_equal(
            raw_seconds,
            np.asarray(
                [
                    item.clock_reference_raw_seconds
                    for item in reference_set.clocks
                ],
                dtype=np.float64,
            ),
        ):
            raise LayoutV4ValidationError(
                "layout-v4 clock-reference datasets disagree"
            )

    dcb_reference = (
        reference_set.reference_for(ClockSource.DCB)
        if reference_set is not None
        else None
    )
    if dcb_reference is None:
        if not np.isnan(telemetry.mjd_times).all():
            raise LayoutV4ValidationError(
                "layout-v4 telemetry MJD requires a DCB clock reference"
            )
        return
    if not np.isfinite(telemetry.mjd_times).all():
        raise LayoutV4ValidationError(
            "layout-v4 DCB-referenced telemetry has missing MJD times"
        )
    expected = np.asarray(
        reference_set.to_mjd(
            telemetry.raw_seconds,
            clock_source=ClockSource.DCB,
        ),
        dtype=np.float64,
    )
    if not np.array_equal(telemetry.mjd_times, expected):
        raise LayoutV4ValidationError(
            "layout-v4 telemetry MJD contradicts the DCB clock reference"
        )


def _validate_counted_groups(validator: LayoutV4Validator) -> None:
    def visit(group: LayoutGroup) -> None:
        if "count" in group.attrs and group.path != "/issues":
            count = _uint64_attr(validator, group.path or "/", "count")
            direct = tuple(
                name
                for name, child in group.children.items()
                if isinstance(child, LayoutDataset)
            )
            if direct:
                validator.require_row_aligned(group.path, direct, row_count=count)
        for child in group.children.values():
            if isinstance(child, LayoutGroup):
                visit(child)

    visit(validator.root)


def validate_layout_v4_tree(root: LayoutGroup) -> LayoutV4Validator:
    """Validate generic v4 structure and foundational provenance contracts."""
    _validate_tree_structure(root)
    validator = LayoutV4Validator(root)
    _validate_common_provenance(validator)
    _validate_telemetry(validator)
    _validate_counted_groups(validator)
    return validator


def decode_field_union_rows(
    validator: LayoutV4Validator,
    path: str,
    row_count: int,
    *,
    parent_present: np.ndarray | None = None,
) -> tuple[tuple[dict[str, object], ...], tuple[dict[str, bool], ...]]:
    """Validate and decode one recursive layout-v4 field union by row."""
    fields = validator.group(f"{path}/fields")
    presence = validator.group(f"{path}/field_present")
    if set(fields.children) != set(presence.children):
        raise LayoutV4ValidationError(
            f"layout-v4 field and presence keys disagree at {path}"
        )
    if parent_present is None:
        parent_present = np.ones(row_count, dtype=np.bool_)
    else:
        parent_present = np.asarray(parent_present)
        if parent_present.dtype != np.dtype(np.bool_) or parent_present.shape != (
            row_count,
        ):
            raise LayoutV4ValidationError(
                f"layout-v4 parent presence is invalid at {path}"
            )

    rows = [{} for _ in range(row_count)]
    masks = [{} for _ in range(row_count)]
    for name in sorted(fields.children):
        present = validator.dataset(
            f"{path}/field_present/{name}",
            dtype=np.bool_,
            utf8=False,
            shape=(row_count,),
        ).data
        if np.any(present & ~parent_present):
            raise LayoutV4ValidationError(
                f"layout-v4 nested field {path}/{name} outlives its parent"
            )
        values = _decode_field_values(
            validator,
            f"{path}/fields/{name}",
            present,
        )
        for index in range(row_count):
            is_present = bool(present[index])
            masks[index][name] = is_present
            rows[index][name] = values[index] if is_present else None
    return tuple(rows), tuple(masks)


def _decode_field_values(
    validator: LayoutV4Validator,
    path: str,
    present: np.ndarray,
) -> list[object | None]:
    group = validator.group(path)
    kind = _string_attr(validator, path, "kind")
    row_count = present.size
    values: list[object | None] = [None] * row_count
    if kind == "untyped_absent":
        if np.any(present) or group.children:
            raise LayoutV4ValidationError(
                f"layout-v4 untyped field is not absent at {path}"
            )
        return values
    if kind == "mapping":
        if set(group.children) != {"fields", "field_present"}:
            raise LayoutV4ValidationError(
                f"layout-v4 mapping field has unexpected children at {path}"
            )
        nested_rows, nested_masks = decode_field_union_rows(
            validator,
            path,
            row_count,
            parent_present=present,
        )
        for index in np.flatnonzero(present):
            values[index] = {
                name: value
                for name, value in nested_rows[index].items()
                if nested_masks[index][name]
            }
        return values
    if kind != "array_variants":
        raise LayoutV4ValidationError(
            f"layout-v4 field kind {kind!r} is unsupported at {path}"
        )

    variant_count = int(
        _scalar_attr(validator, path, "variant_count", np.uint32)
    )
    if variant_count < 1:
        raise LayoutV4ValidationError(
            f"layout-v4 array field has no variants at {path}"
        )
    expected_children = {"variant_index"} | {
        f"variant_{index:03d}" for index in range(variant_count)
    }
    if set(group.children) != expected_children:
        raise LayoutV4ValidationError(
            f"layout-v4 array variants disagree at {path}"
        )
    variant_index = validator.dataset(
        f"{path}/variant_index",
        dtype=np.int32,
        utf8=False,
        shape=(row_count,),
    ).data
    if np.any(variant_index[~present] != -1) or np.any(
        (variant_index[present] < 0) | (variant_index[present] >= variant_count)
    ):
        raise LayoutV4ValidationError(
            f"layout-v4 variant index disagrees with presence at {path}"
        )

    for number in range(variant_count):
        variant_path = f"{path}/variant_{number:03d}"
        variant = validator.group(variant_path)
        if set(variant.children) != {"row_indices", "data"}:
            raise LayoutV4ValidationError(
                f"layout-v4 variant has unexpected children at {variant_path}"
            )
        dtype_text = _string_attr(validator, variant_path, "numpy_dtype")
        shape_text = _string_attr(validator, variant_path, "value_shape_json")
        try:
            declared_dtype = np.dtype(dtype_text)
            shape_value = json.loads(shape_text)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise LayoutV4ValidationError(
                f"layout-v4 variant contract is invalid at {variant_path}"
            ) from exc
        if (
            declared_dtype.hasobject
            or declared_dtype.metadata is not None
            or not declared_dtype.isnative
            or not _portable_dtype(declared_dtype, allow_unicode=True)
            or not isinstance(shape_value, list)
            or any(type(item) is not int or item < 0 for item in shape_value)
        ):
            raise LayoutV4ValidationError(
                f"layout-v4 variant contract is invalid at {variant_path}"
            )
        value_shape = tuple(shape_value)
        expected_rows = np.flatnonzero(variant_index == number).astype(
            np.uint64
        )
        row_indices = validator.dataset(
            f"{variant_path}/row_indices",
            dtype=np.uint64,
            utf8=False,
            shape=(expected_rows.size,),
        ).data
        if not np.array_equal(row_indices, expected_rows):
            raise LayoutV4ValidationError(
                f"layout-v4 variant row indices disagree at {variant_path}"
            )
        data = validator.dataset(
            f"{variant_path}/data",
            shape=(expected_rows.size, *value_shape),
        )
        if declared_dtype.kind == "U":
            if not data.is_utf8:
                raise LayoutV4ValidationError(
                    f"layout-v4 Unicode variant is not UTF-8 at {variant_path}"
                )
        elif data.is_utf8 or data.dtype != declared_dtype:
            raise LayoutV4ValidationError(
                f"layout-v4 variant dtype disagrees at {variant_path}"
            )
        for position, row_index in enumerate(expected_rows):
            value = data.data[position]
            if declared_dtype.kind == "U":
                value = np.asarray(value, dtype=declared_dtype)
                if value_shape == ():
                    value = value[()]
            elif value_shape != ():
                value = np.asarray(value).copy()
            values[int(row_index)] = value

    if any(values[index] is None for index in np.flatnonzero(present)):
        raise LayoutV4ValidationError(
            f"layout-v4 field variants do not cover every present row at {path}"
        )
    return values


def _read_hdf5_group(group, h5py, path: str, seen: set[int]) -> LayoutGroup:
    identity = hash(group.id)
    if identity in seen:
        raise LayoutV4ValidationError("layout-v4 HDF5 contains a hard-link alias")
    seen.add(identity)
    result = LayoutGroup(
        path=path,
        attrs=_read_hdf5_attrs(group.attrs),
    )
    for name in group:
        link = group.get(name, getlink=True)
        if not isinstance(link, h5py.HardLink):
            raise LayoutV4ValidationError(
                f"layout-v4 HDF5 link is not a hard link at "
                f"{path or '/'}/{name}"
            )
        child = group[name]
        child_path = f"{path}/{name}" if path else f"/{name}"
        if isinstance(child, h5py.Group):
            result.children[name] = _read_hdf5_group(
                child, h5py, child_path, seen
            )
            continue
        if not isinstance(child, h5py.Dataset):
            raise LayoutV4ValidationError(
                f"layout-v4 HDF5 object type is unsupported at {child_path}"
            )
        child_identity = hash(child.id)
        if child_identity in seen:
            raise LayoutV4ValidationError(
                "layout-v4 HDF5 contains a hard-link alias"
            )
        seen.add(child_identity)
        string_info = h5py.check_string_dtype(child.dtype)
        if string_info is None:
            if child.dtype.hasobject:
                raise LayoutV4ValidationError(
                    f"layout-v4 HDF5 object dataset is unsupported at {child_path}"
                )
            data = np.asarray(child[...])
            is_utf8 = False
        elif (
            string_info.encoding == "utf-8"
            and string_info.length is None
        ):
            data = np.asarray(
                child.asstr(encoding="utf-8", errors="strict")[...],
                dtype=object,
            )
            is_utf8 = True
        elif child.dtype.kind == "S" and string_info.length is not None:
            data = np.asarray(
                child[...], dtype=np.dtype(f"S{child.dtype.itemsize}")
            )
            is_utf8 = False
        else:
            raise LayoutV4ValidationError(
                f"layout-v4 HDF5 string encoding is unsupported at {child_path}"
            )
        result.children[name] = LayoutDataset(
            data=data,
            is_utf8=is_utf8,
            attrs=_read_hdf5_attrs(child.attrs),
        )
    return result


def _read_hdf5_attrs(attrs) -> dict[str, object]:
    result = {}
    for name in attrs:
        value = attrs[name]
        array = np.asarray(value)
        if array.dtype.kind == "S" and array.dtype.metadata is not None:
            array = np.asarray(array, dtype=np.dtype(f"S{array.dtype.itemsize}"))
            value = array[()] if array.shape == () else array
        result[name] = value
    return result


def read_layout_v4_hdf5(path: Path | str) -> LayoutGroup:
    """Read and validate one HDF5 layout-v4 file with lazy h5py import."""
    h5py = import_optional_dependency("h5py", "HDF5 ingest input")
    with h5py.File(Path(path), "r") as h5:
        version = h5.attrs.get("layout_version")
        version_array = np.asarray(version) if version is not None else None
        if (
            version_array is None
            or version_array.shape != ()
            or version_array.dtype != np.dtype(np.uint16)
            or int(version_array) != INGEST_LAYOUT_VERSION
        ):
            raise LayoutV4ValidationError(
                "HDF5 input is not ingest layout version 4"
            )
        root = _read_hdf5_group(h5, h5py, "", set())
    validate_layout_v4_tree(root)
    return root


def read_layout_v4_fits(path: Path | str) -> LayoutGroup:
    """Read and validate one checksummed canonical FITS layout-v4 file."""
    from . import fits_writer

    fits = fits_writer._fits_module()
    with fits.open(Path(path), mode="readonly", uint=True, memmap=False) as hdul:
        hdul.verify("exception")
        if not hdul:
            raise LayoutV4ValidationError("FITS input has no HDUs")
        for index, hdu in enumerate(hdul):
            if (
                fits_writer._header_count(hdu.header, "CHECKSUM") != 1
                or fits_writer._header_count(hdu.header, "DATASUM") != 1
                or hdu.verify_checksum() != 1
                or hdu.verify_datasum() != 1
            ):
                raise LayoutV4ValidationError(
                    f"FITS checksum contract failed at HDU {index}"
                )
        root = fits_writer._read_layout_tree(hdul, fits)
        validate_layout_v4_tree(root)
        fits_writer._verify_transport_headers(hdul, root, fits)
    return root


def read_layout_v4(path: Path | str) -> LayoutGroup:
    """Detect HDF5 or FITS by file signature, then read strict layout v4."""
    source = Path(path)
    with source.open("rb") as stream:
        signature = stream.read(16)
    if signature.startswith(b"\x89HDF\r\n\x1a\n"):
        return read_layout_v4_hdf5(source)
    if signature.startswith(b"SIMPLE  ="):
        return read_layout_v4_fits(source)
    raise LayoutV4ValidationError("input is neither HDF5 nor FITS")


__all__ = [
    "LayoutV4ValidationError",
    "LayoutV4Validator",
    "decode_field_union_rows",
    "read_layout_v4",
    "read_layout_v4_fits",
    "read_layout_v4_hdf5",
    "validate_layout_v4_tree",
]
