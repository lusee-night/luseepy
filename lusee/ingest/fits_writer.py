"""Atomic FITS transport for the validated ingest layout v4 tree."""

from __future__ import annotations

import base64
import json
import logging
import os
import tempfile
from importlib import import_module
from pathlib import Path

import numpy as np

from .constants import INGEST_LAYOUT_VERSION
from .dependencies import import_optional_dependency
from .layout_v4_tree import (
    LayoutDataset,
    LayoutGroup,
    assert_layout_trees_equal,
    build_layout_v4_tree,
    group_datasets,
    iter_layout_groups,
)
from .write_request import WriteRequest

log = logging.getLogger(__name__)

FITS_TRANSPORT_VERSION = 1
_PATH_KEY = "LUSEEPTH"
_KIND_KEY = "LUSEEKND"
_ATTRS_KEY = "ATTRJSON"
_COLUMNS_KEY = "COLJSON"
_PART_KEY = "LUSEEPRT"
_COLUMN_NAME_MAX = 60
_MAX_TABLE_COLUMNS = 999


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _header_count(header, name: str) -> int:
    return header.count(name) if name in header else 0


def _semantic_header_cards(header) -> list[tuple[str, object, str]]:
    return [
        (card.keyword, card.value, card.comment)
        for card in header.cards
        if card.keyword not in ("CHECKSUM", "DATASUM")
    ]


def _fits_module():
    import_optional_dependency("astropy", "FITS ingest output")
    return import_module("astropy.io.fits")


def _encode_attr(value: object) -> dict[str, object]:
    if type(value) is str:
        return {"encoding": "utf8", "value": value}
    array = np.asarray(value)
    if array.dtype.kind == "U":
        return {
            "encoding": "utf8_array",
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "values": array.reshape(-1).tolist(),
        }
    if array.dtype.hasobject:
        raise TypeError("layout-v4 FITS attributes must not have object dtype")
    return {
        "encoding": "raw",
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "data": base64.b64encode(array.tobytes(order="C")).decode("ascii"),
    }


def _decode_attr(record: object) -> object:
    if not isinstance(record, dict):
        raise TypeError("FITS attribute record must be an object")
    encoding = record.get("encoding")
    if encoding == "utf8":
        value = record.get("value")
        if type(value) is not str:
            raise ValueError("FITS UTF-8 attribute is invalid")
        return value
    if encoding == "utf8_array":
        dtype = np.dtype(record["dtype"])
        shape = _json_shape(record["shape"])
        values = record.get("values")
        if not isinstance(values, list):
            raise ValueError("FITS UTF-8 attribute array is invalid")
        return np.asarray(values, dtype=dtype).reshape(shape)
    if encoding != "raw":
        raise ValueError("FITS attribute encoding is unsupported")
    dtype = np.dtype(record["dtype"])
    shape = _json_shape(record["shape"])
    encoded = record.get("data")
    if type(encoded) is not str:
        raise ValueError("FITS raw attribute data is invalid")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError("FITS raw attribute data is invalid") from exc
    expected_size = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    if not shape:
        expected_size = dtype.itemsize
    if len(raw) != expected_size:
        raise ValueError("FITS raw attribute byte count disagrees")
    array = np.frombuffer(raw, dtype=dtype).copy().reshape(shape)
    return array[()] if not shape else array


def _encode_attrs(attrs: dict[str, object]) -> str:
    return _canonical_json(
        {name: _encode_attr(value) for name, value in sorted(attrs.items())}
    )


def _decode_attrs(value: object) -> dict[str, object]:
    if type(value) is not str:
        raise ValueError("FITS layout attributes are missing")
    try:
        records = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("FITS layout attribute JSON is invalid") from exc
    if not isinstance(records, dict):
        raise TypeError("FITS layout attribute JSON must encode an object")
    return {name: _decode_attr(record) for name, record in records.items()}


def _json_shape(value: object) -> tuple[int, ...]:
    if not isinstance(value, list) or any(
        type(item) is not int or item < 0 for item in value
    ):
        raise ValueError("FITS logical array shape is invalid")
    return tuple(value)


def _column_for_dataset(name: str, dataset: LayoutDataset, fits):
    array = dataset.data
    if array.ndim == 0:
        raise ValueError(f"layout-v4 FITS dataset {name!r} has no row axis")
    logical_shape = array.shape
    cell_shape = logical_shape[1:]
    schema: dict[str, object] = {
        "logical_dtype": "utf8" if dataset.is_utf8 else array.dtype.str,
        "logical_shape": list(logical_shape),
        "attrs": {
            attr_name: _encode_attr(value)
            for attr_name, value in sorted(dataset.attrs.items())
        },
    }
    if cell_shape and int(np.prod(cell_shape, dtype=np.int64)) == 0:
        storage = np.zeros((logical_shape[0], 1), dtype=np.uint8)
        column = fits.Column(name=name, format="1B", array=storage)
        schema.update({"encoding": "empty", "tform": "1B"})
        return column, schema
    if dataset.is_utf8:
        column, transport = _utf8_column(name, array, fits)
        schema.update(transport)
        return column, schema
    if array.dtype.kind == "S":
        itemsize = array.dtype.itemsize
        storage = array.view(np.uint8).reshape((*array.shape, itemsize))
        column = _numeric_column(name, storage, "B", fits)
        schema.update(
            {
                "encoding": "bytes",
                "itemsize": itemsize,
                "tform": column.format,
            }
        )
        return column, schema
    if array.dtype.kind == "b":
        column = _numeric_column(name, array, "L", fits)
        schema.update({"encoding": "native", "tform": column.format})
        return column, schema
    if array.dtype.kind == "i":
        if array.dtype.itemsize == 1:
            storage = array.astype(np.int16)
            column = _numeric_column(name, storage, "I", fits)
            schema.update({"encoding": "int8", "tform": column.format})
            return column, schema
        base = {2: "I", 4: "J", 8: "K"}.get(array.dtype.itemsize)
        if base is None:
            raise TypeError(f"FITS cannot store dtype {array.dtype}")
        column = _numeric_column(name, array, base, fits)
        schema.update({"encoding": "native", "tform": column.format})
        return column, schema
    if array.dtype.kind == "u":
        base = {1: "B", 2: "I", 4: "J", 8: "K"}.get(array.dtype.itemsize)
        if base is None:
            raise TypeError(f"FITS cannot store dtype {array.dtype}")
        bzero = None if array.dtype.itemsize == 1 else 1 << (
            8 * array.dtype.itemsize - 1
        )
        column = _numeric_column(name, array, base, fits, bzero=bzero)
        schema.update(
            {
                "encoding": "unsigned",
                "tform": column.format,
                "bzero": bzero,
            }
        )
        return column, schema
    if array.dtype.kind == "f":
        base = {4: "E", 8: "D"}.get(array.dtype.itemsize)
    elif array.dtype.kind == "c":
        base = {8: "C", 16: "M"}.get(array.dtype.itemsize)
    else:
        base = None
    if base is None:
        raise TypeError(f"FITS cannot store dtype {array.dtype}")
    column = _numeric_column(name, array, base, fits)
    schema.update({"encoding": "native", "tform": column.format})
    return column, schema


def _utf8_column(name: str, array: np.ndarray, fits):
    row_count = array.shape[0]
    cell_shape = array.shape[1:]
    element_count = int(np.prod(cell_shape, dtype=np.int64)) if cell_shape else 1
    flattened = array.reshape(row_count, element_count)
    encoded = [
        item.encode("utf-8")
        for item in flattened.reshape(-1).tolist()
    ]
    width = max((len(item) for item in encoded), default=0)
    width = max(width, 1)
    storage = np.zeros((row_count, element_count, width), dtype=np.uint8)
    storage_flat = storage.reshape(-1, width)
    for index, item in enumerate(encoded):
        storage_flat[index, : len(item)] = np.frombuffer(item, dtype=np.uint8)
    if not cell_shape:
        storage = storage.reshape(row_count, width)
    else:
        storage = storage.reshape(row_count, *cell_shape, width)
    column = _numeric_column(name, storage, "B", fits)
    return column, {
        "encoding": "utf8",
        "utf8_width": width,
        "tform": column.format,
    }


def _numeric_column(name: str, array: np.ndarray, base: str, fits, *, bzero=None):
    inner_shape = array.shape[1:]
    repeat = int(np.prod(inner_shape, dtype=np.int64)) if inner_shape else 1
    format_name = base if repeat == 1 else f"{repeat}{base}"
    dim = None
    if inner_shape:
        dim = "(" + ",".join(str(item) for item in reversed(inner_shape)) + ")"
    return fits.Column(
        name=name,
        format=format_name,
        array=array,
        dim=dim,
        bzero=bzero,
    )


def _table_hdu(
    group: LayoutGroup,
    datasets: dict[str, LayoutDataset],
    index: int,
    fits,
    *,
    kind: str,
    include_attrs: bool,
    part: int | None = None,
):
    extname = f"L4G{index:04d}"
    columns = []
    schema = {}
    for column_index, (name, dataset) in enumerate(datasets.items(), start=1):
        transport_name = (
            name
            if len(name.encode("ascii")) <= _COLUMN_NAME_MAX
            else f"__L4C{column_index:04d}"
        )
        column, record = _column_for_dataset(transport_name, dataset, fits)
        if transport_name != name:
            record["transport_name"] = transport_name
        columns.append(column)
        schema[name] = record
    hdu = fits.BinTableHDU.from_columns(columns, name=extname)
    hdu.header[_KIND_KEY] = kind
    hdu.header[_COLUMNS_KEY] = _canonical_json(schema)
    hdu.header[_PATH_KEY] = group.path
    if include_attrs:
        hdu.header[_ATTRS_KEY] = _encode_attrs(group.attrs)
    if part is not None:
        hdu.header[_PART_KEY] = part
    return hdu


def _group_hdus(group: LayoutGroup, start_index: int, fits):
    datasets = group_datasets(group)
    if not datasets:
        hdu = fits.ImageHDU(data=None, name=f"L4G{start_index:04d}")
        hdu.header[_KIND_KEY] = "group"
        hdu.header[_PATH_KEY] = group.path
        hdu.header[_ATTRS_KEY] = _encode_attrs(group.attrs)
        return [hdu]
    by_row_count: dict[int, dict[str, LayoutDataset]] = {}
    for name, dataset in datasets.items():
        by_row_count.setdefault(dataset.shape[0], {})[name] = dataset
    partitions = []
    for row_count in sorted(by_row_count):
        items = list(by_row_count[row_count].items())
        for start in range(0, len(items), _MAX_TABLE_COLUMNS):
            partitions.append(dict(items[start : start + _MAX_TABLE_COLUMNS]))
    if len(partitions) == 1:
        return [
            _table_hdu(
                group,
                partitions[0],
                start_index,
                fits,
                kind="table",
                include_attrs=True,
            )
        ]
    header_hdu = fits.ImageHDU(data=None, name=f"L4G{start_index:04d}")
    header_hdu.header[_KIND_KEY] = "group"
    header_hdu.header[_PATH_KEY] = group.path
    header_hdu.header[_ATTRS_KEY] = _encode_attrs(group.attrs)
    hdus = [header_hdu]
    for part, partition in enumerate(partitions):
        hdus.append(
            _table_hdu(
                group,
                partition,
                start_index + part + 1,
                fits,
                kind="table-part",
                include_attrs=False,
                part=part,
            )
        )
    return hdus


def _tree_to_hdul(root: LayoutGroup, fits):
    primary = fits.PrimaryHDU()
    primary.header["ORIGIN"] = "lusee.ingest.fits_writer"
    primary.header["LAYOUTV"] = INGEST_LAYOUT_VERSION
    primary.header["FITSFMT"] = FITS_TRANSPORT_VERSION
    primary.header["QUALITY"] = root.attrs["quality_status"]
    primary.header["EXECMODE"] = root.attrs["execution_mode"]
    primary.header[_PATH_KEY] = "/"
    primary.header[_KIND_KEY] = "primary"
    primary.header[_ATTRS_KEY] = _encode_attrs(root.attrs)
    hdus = [primary]
    next_index = 1
    for group in iter_layout_groups(root):
        group_hdus = _group_hdus(group, next_index, fits)
        hdus.extend(group_hdus)
        next_index += len(group_hdus)
    return fits.HDUList(hdus)


def _write_fits_temp(path: Path, root: LayoutGroup, fits) -> None:
    hdul = _tree_to_hdul(root, fits)
    try:
        hdul.writeto(
            path,
            overwrite=True,
            checksum=True,
            output_verify="exception",
        )
    finally:
        hdul.close()


def _read_layout_tree(hdul, fits) -> LayoutGroup:
    primary = hdul[0]
    if not isinstance(primary, fits.PrimaryHDU):
        raise TypeError("temporary FITS has no primary HDU")
    if (
        primary.header.get("LAYOUTV") != INGEST_LAYOUT_VERSION
        or primary.header.get("FITSFMT") != FITS_TRANSPORT_VERSION
        or primary.header.get(_PATH_KEY) != "/"
        or primary.header.get(_KIND_KEY) != "primary"
    ):
        raise ValueError("temporary FITS primary contract disagrees")
    root = LayoutGroup(path="", attrs=_decode_attrs(primary.header.get(_ATTRS_KEY)))
    if (
        primary.header.get("QUALITY") != root.attrs.get("quality_status")
        or primary.header.get("EXECMODE") != root.attrs.get("execution_mode")
    ):
        raise ValueError("temporary FITS public provenance cards disagree")
    declared_paths = set()
    next_part: dict[str, int] = {}
    for hdu_index, hdu in enumerate(hdul[1:], start=1):
        if hdu.name != f"L4G{hdu_index:04d}":
            raise ValueError("temporary FITS extension order disagrees")
        path = hdu.header.get(_PATH_KEY)
        kind = hdu.header.get(_KIND_KEY)
        if (
            type(path) is not str
            or not path.startswith("/")
            or path == "/"
        ):
            raise ValueError("temporary FITS group path is invalid")
        group = _require_tree_group(root, path)
        if kind in ("group", "table"):
            if path in declared_paths:
                raise ValueError("temporary FITS group path is duplicated")
            declared_paths.add(path)
            group.attrs.update(_decode_attrs(hdu.header.get(_ATTRS_KEY)))
        elif kind == "table-part":
            if path not in declared_paths or _ATTRS_KEY in hdu.header:
                raise ValueError("temporary FITS table partition is invalid")
            part = hdu.header.get(_PART_KEY)
            if type(part) is not int or part != next_part.get(path, 0):
                raise ValueError("temporary FITS table partition order disagrees")
            next_part[path] = part + 1
        else:
            raise ValueError(f"temporary FITS HDU kind disagrees at {path}")
        if kind == "group":
            if not isinstance(hdu, fits.ImageHDU) or hdu.data is not None:
                raise ValueError(f"temporary FITS group HDU disagrees at {path}")
            continue
        if not isinstance(hdu, fits.BinTableHDU):
            raise TypeError(f"temporary FITS table HDU disagrees at {path}")
        _read_group_datasets(group, hdu)
    return root


def _require_tree_group(root: LayoutGroup, path: str) -> LayoutGroup:
    group = root
    for part in path.strip("/").split("/"):
        child = group.children.get(part)
        if child is None:
            child = group._new_group(part)
            group.children[part] = child
        if not isinstance(child, LayoutGroup):
            raise TypeError(f"temporary FITS path crosses a dataset: {path}")
        group = child
    return group


def _read_group_datasets(group: LayoutGroup, hdu) -> None:
    raw_schema = hdu.header.get(_COLUMNS_KEY)
    if type(raw_schema) is not str:
        raise ValueError(f"temporary FITS column schema is missing at {group.path}")
    try:
        schema = json.loads(raw_schema)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"temporary FITS column schema is invalid at {group.path}"
        ) from exc
    if not isinstance(schema, dict) or not schema:
        raise ValueError(f"temporary FITS columns disagree at {group.path}")
    transport_names = []
    for name, record in schema.items():
        if not isinstance(record, dict):
            raise TypeError(
                f"temporary FITS column record is invalid at {group.path}"
            )
        transport_name = record.get("transport_name", name)
        if type(transport_name) is not str:
            raise TypeError(
                f"temporary FITS transport name is invalid at {group.path}"
            )
        transport_names.append(transport_name)
    if len(set(transport_names)) != len(transport_names) or (
        list(hdu.columns.names) != transport_names
    ):
        raise ValueError(f"temporary FITS columns disagree at {group.path}")
    for column_index, (name, record) in enumerate(schema.items(), start=1):
        transport_name = transport_names[column_index - 1]
        expected_tform = record.get("tform")
        observed_tform = hdu.header.get(f"TFORM{column_index}")
        if (
            type(expected_tform) is not str
            or observed_tform.strip() != expected_tform
        ):
            raise ValueError(
                f"temporary FITS TFORM disagrees for {group.path}/{name}"
            )
        bzero = record.get("bzero")
        observed_bzero = hdu.header.get(f"TZERO{column_index}")
        observed_scale = hdu.header.get(f"TSCAL{column_index}")
        if bzero is None:
            if observed_bzero is not None:
                raise ValueError(
                    f"temporary FITS unsigned scaling is unexpected for "
                    f"{group.path}/{name}"
                )
        elif observed_bzero != bzero or observed_scale not in (None, 1):
            raise ValueError(
                f"temporary FITS unsigned scaling disagrees for "
                f"{group.path}/{name}"
            )
        dataset = _decode_column(hdu.data[transport_name], record)
        raw_attrs = record.get("attrs")
        if not isinstance(raw_attrs, dict):
            raise TypeError(
                f"temporary FITS dataset attributes are invalid at "
                f"{group.path}/{name}"
            )
        dataset.attrs = {
            attr_name: _decode_attr(value)
            for attr_name, value in raw_attrs.items()
        }
        if name in group.children:
            raise ValueError(
                f"temporary FITS duplicates dataset {group.path}/{name}"
            )
        group.children[name] = dataset


def _decode_column(raw: object, record: dict[str, object]) -> LayoutDataset:
    logical_shape = _json_shape(record.get("logical_shape"))
    logical_dtype = record.get("logical_dtype")
    encoding = record.get("encoding")
    if encoding == "empty":
        if logical_dtype == "utf8":
            return LayoutDataset(
                np.empty(logical_shape, dtype=object),
                is_utf8=True,
            )
        dtype = np.dtype(logical_dtype)
        return LayoutDataset(np.empty(logical_shape, dtype=dtype))
    if encoding == "utf8":
        width = record.get("utf8_width")
        if type(width) is not int or width <= 0:
            raise ValueError("temporary FITS UTF-8 width is invalid")
        row_count = logical_shape[0]
        cell_shape = logical_shape[1:]
        element_count = (
            int(np.prod(cell_shape, dtype=np.int64)) if cell_shape else 1
        )
        storage = np.asarray(raw, dtype=np.uint8).reshape(
            row_count, element_count, width
        )
        values = []
        for item in storage.reshape(-1, width):
            encoded = item.tobytes().rstrip(b"\x00")
            try:
                values.append(encoded.decode("utf-8"))
            except UnicodeDecodeError as exc:
                raise ValueError("temporary FITS UTF-8 value is invalid") from exc
        data = np.asarray(values, dtype=object).reshape(logical_shape)
        return LayoutDataset(data=data, is_utf8=True)
    dtype = np.dtype(logical_dtype)
    if encoding == "bytes":
        itemsize = record.get("itemsize")
        if type(itemsize) is not int or itemsize != dtype.itemsize:
            raise ValueError("temporary FITS byte width disagrees")
        storage = np.asarray(raw, dtype=np.uint8).reshape(
            *logical_shape, itemsize
        )
        data = np.frombuffer(storage.tobytes(), dtype=dtype).copy().reshape(
            logical_shape
        )
        return LayoutDataset(data=data)
    if encoding not in ("native", "unsigned", "int8"):
        raise ValueError("temporary FITS column encoding is unsupported")
    data = np.asarray(raw).astype(dtype, copy=False).reshape(logical_shape)
    return LayoutDataset(data=data)


def _verify_fits(path: Path, expected: LayoutGroup, fits) -> None:
    with fits.open(path, mode="readonly", uint=True, memmap=False) as hdul:
        hdul.verify("exception")
        for hdu in hdul:
            if (
                _header_count(hdu.header, "CHECKSUM") != 1
                or _header_count(hdu.header, "DATASUM") != 1
                or hdu.verify_checksum() != 1
                or hdu.verify_datasum() != 1
            ):
                raise ValueError("temporary FITS checksum contract failed")
        _verify_transport_headers(hdul, expected, fits)
        observed = _read_layout_tree(hdul, fits)
    assert_layout_trees_equal(
        expected,
        observed,
        context="temporary FITS",
    )


def _verify_transport_headers(observed, expected_root: LayoutGroup, fits) -> None:
    expected = _tree_to_hdul(expected_root, fits)
    try:
        if len(observed) != len(expected):
            raise ValueError("temporary FITS HDU count disagrees")
        for hdu_index, (observed_hdu, expected_hdu) in enumerate(
            zip(observed, expected, strict=True)
        ):
            if type(observed_hdu) is not type(expected_hdu):
                raise TypeError(
                    f"temporary FITS HDU type disagrees at index {hdu_index}"
                )
            for name in (
                "ORIGIN",
                "LAYOUTV",
                "FITSFMT",
                "QUALITY",
                "EXECMODE",
                "EXTNAME",
                _PATH_KEY,
                _KIND_KEY,
                _ATTRS_KEY,
                _COLUMNS_KEY,
                _PART_KEY,
                "XTENSION",
                "BITPIX",
                "NAXIS",
                "NAXIS1",
                "NAXIS2",
                "PCOUNT",
                "GCOUNT",
                "TFIELDS",
            ):
                if (
                    _header_count(observed_hdu.header, name)
                    != _header_count(expected_hdu.header, name)
                    or observed_hdu.header.get(name)
                    != expected_hdu.header.get(name)
                ):
                    raise ValueError(
                        f"temporary FITS header {name} disagrees at "
                        f"index {hdu_index}"
                    )
            if not isinstance(expected_hdu, fits.BinTableHDU):
                if _semantic_header_cards(observed_hdu.header) != (
                    _semantic_header_cards(expected_hdu.header)
                ):
                    raise ValueError(
                        f"temporary FITS complete header disagrees at "
                        f"index {hdu_index}"
                    )
                continue
            if observed_hdu.columns.names != expected_hdu.columns.names:
                raise ValueError(
                    f"temporary FITS column names disagree at index {hdu_index}"
                )
            for column_index in range(1, len(expected_hdu.columns) + 1):
                for prefix in (
                    "TTYPE",
                    "TFORM",
                    "TDIM",
                    "TZERO",
                    "TSCAL",
                    "TNULL",
                    "TUNIT",
                ):
                    key = f"{prefix}{column_index}"
                    if (
                        _header_count(observed_hdu.header, key)
                        != _header_count(expected_hdu.header, key)
                        or observed_hdu.header.get(key)
                        != expected_hdu.header.get(key)
                    ):
                        raise ValueError(
                            f"temporary FITS header {key} disagrees at "
                            f"index {hdu_index}"
                        )
            if _semantic_header_cards(observed_hdu.header) != (
                _semantic_header_cards(expected_hdu.header)
            ):
                raise ValueError(
                    f"temporary FITS complete header disagrees at "
                    f"index {hdu_index}"
                )
    finally:
        expected.close()


def _install_atomic(temp_path: Path, destination: Path, *, overwrite: bool) -> None:
    if overwrite:
        os.replace(temp_path, destination)
        return
    os.link(temp_path, destination)
    temp_path.unlink()


def write_fits(request: WriteRequest, dest: Path | str) -> Path:
    """Validate, write, verify, and atomically install one layout-v4 FITS file."""
    if not isinstance(request, WriteRequest):
        raise TypeError("write_fits requires a validated WriteRequest")
    request.validate()
    destination = Path(dest)
    if destination.exists() and not request.overwrite:
        raise FileExistsError(destination)
    destination_preexisted = destination.exists()
    fits = _fits_module()
    expected = build_layout_v4_tree(
        request,
        destination_preexisted=destination_preexisted,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        _write_fits_temp(temporary, expected, fits)
        _verify_fits(temporary, expected, fits)
        _install_atomic(temporary, destination, overwrite=request.overwrite)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    log.info("wrote layout-v4 FITS %s", destination)
    return destination


__all__ = ["write_fits"]
