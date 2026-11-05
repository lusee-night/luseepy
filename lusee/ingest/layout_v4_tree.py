"""In-memory layout-v4 tree shared by non-HDF5 serializers."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field

import numpy as np

from .write_request import WriteRequest


@dataclass(slots=True)
class LayoutDataset:
    """One array plus its layout attributes."""

    data: np.ndarray
    is_utf8: bool = False
    attrs: dict[str, object] = field(default_factory=dict)

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape


@dataclass(slots=True)
class LayoutGroup:
    """Minimal h5py-like group used to assemble format-neutral v4 content."""

    path: str
    attrs: dict[str, object] = field(default_factory=dict)
    children: dict[str, LayoutGroup | LayoutDataset] = field(default_factory=dict)

    def _parts(self, name: str) -> tuple[str, ...]:
        return tuple(part for part in name.strip("/").split("/") if part)

    def _group_for_parent(self, name: str) -> tuple[LayoutGroup, str]:
        parts = self._parts(name)
        if not parts:
            raise ValueError("layout child name must not be empty")
        parent = self
        for part in parts[:-1]:
            child = parent.children.get(part)
            if child is None:
                child = parent._new_group(part)
                parent.children[part] = child
            if not isinstance(child, LayoutGroup):
                raise TypeError(f"layout path {name!r} crosses a dataset")
            parent = child
        return parent, parts[-1]

    def _new_group(self, name: str) -> LayoutGroup:
        path = f"{self.path}/{name}" if self.path else f"/{name}"
        return LayoutGroup(path=path)

    def create_group(self, name: str) -> LayoutGroup:
        parent, leaf = self._group_for_parent(name)
        if leaf in parent.children:
            raise ValueError(f"layout path already exists: {name}")
        group = parent._new_group(leaf)
        parent.children[leaf] = group
        return group

    def require_group(self, name: str) -> LayoutGroup:
        parent, leaf = self._group_for_parent(name)
        child = parent.children.get(leaf)
        if child is None:
            child = parent._new_group(leaf)
            parent.children[leaf] = child
        if not isinstance(child, LayoutGroup):
            raise TypeError(f"layout path is not a group: {name}")
        return child

    def create_dataset(
        self,
        name: str,
        *,
        data: object,
        dtype: object | None = None,
        **kwargs,
    ) -> LayoutDataset:
        del kwargs
        parent, leaf = self._group_for_parent(name)
        if leaf in parent.children:
            raise ValueError(f"layout path already exists: {name}")
        requested_dtype = None if dtype is None else np.dtype(dtype)
        metadata = None if requested_dtype is None else requested_dtype.metadata
        is_utf8 = bool(metadata and metadata.get("vlen") is str)
        array = np.asarray(data, dtype=object if is_utf8 else requested_dtype)
        dataset = LayoutDataset(data=array, is_utf8=is_utf8)
        parent.children[leaf] = dataset
        return dataset

    def __getitem__(self, name: str) -> LayoutGroup | LayoutDataset:
        current: LayoutGroup | LayoutDataset = self
        for part in self._parts(name):
            if not isinstance(current, LayoutGroup):
                raise KeyError(name)
            current = current.children[part]
        return current


class LayoutStorage:
    """h5py surface used by the canonical layout population functions."""

    @staticmethod
    def string_dtype(*, encoding: str) -> np.dtype:
        if encoding != "utf-8":
            raise ValueError("layout-v4 strings must use UTF-8")
        return np.dtype("O", metadata={"vlen": str})


def build_layout_v4_tree(
    request: WriteRequest,
    *,
    destination_preexisted: bool,
) -> LayoutGroup:
    """Populate one in-memory tree through the canonical v4 layout code."""
    from .hdf5_writer import _populate_layout_v4

    root = LayoutGroup(path="")
    _populate_layout_v4(
        root,
        request,
        LayoutStorage,
        destination_preexisted=destination_preexisted,
    )
    return root


def iter_layout_groups(root: LayoutGroup) -> Iterator[LayoutGroup]:
    """Yield non-root groups in deterministic path order."""
    for name in sorted(root.children):
        child = root.children[name]
        if isinstance(child, LayoutGroup):
            yield child
            yield from iter_layout_groups(child)


def group_datasets(group: LayoutGroup) -> Mapping[str, LayoutDataset]:
    """Return direct datasets in deterministic name order."""
    return {
        name: child
        for name, child in sorted(group.children.items())
        if isinstance(child, LayoutDataset)
    }


def assert_layout_trees_equal(
    expected: LayoutGroup,
    observed: LayoutGroup,
    *,
    context: str,
) -> None:
    """Require exact paths, attributes, logical dtypes, shapes, and values."""
    expected_groups = {"": expected}
    expected_groups.update({group.path: group for group in iter_layout_groups(expected)})
    observed_groups = {"": observed}
    observed_groups.update({group.path: group for group in iter_layout_groups(observed)})
    if set(observed_groups) != set(expected_groups):
        raise ValueError(f"{context} group paths disagree")
    for path, expected_group in expected_groups.items():
        observed_group = observed_groups[path]
        _assert_attrs_equal(
            expected_group.attrs,
            observed_group.attrs,
            context=f"{context} group {path or '/'}",
        )
        expected_datasets = group_datasets(expected_group)
        observed_datasets = group_datasets(observed_group)
        if set(observed_datasets) != set(expected_datasets):
            raise ValueError(f"{context} datasets disagree in {path or '/'}")
        for name, expected_dataset in expected_datasets.items():
            observed_dataset = observed_datasets[name]
            if (
                observed_dataset.is_utf8 != expected_dataset.is_utf8
                or observed_dataset.shape != expected_dataset.shape
                or observed_dataset.dtype != expected_dataset.dtype
            ):
                raise ValueError(
                    f"{context} dataset contract disagrees at "
                    f"{path or '/'}/{name}"
                )
            _assert_attrs_equal(
                expected_dataset.attrs,
                observed_dataset.attrs,
                context=f"{context} dataset {path or '/'}/{name}",
            )
            if not _array_equal(expected_dataset.data, observed_dataset.data):
                raise ValueError(
                    f"{context} dataset values disagree at "
                    f"{path or '/'}/{name}"
                )


def _assert_attrs_equal(
    expected: Mapping[str, object],
    observed: Mapping[str, object],
    *,
    context: str,
) -> None:
    if set(observed) != set(expected):
        raise ValueError(f"{context} attributes disagree")
    for name, expected_value in expected.items():
        observed_value = observed[name]
        expected_array = np.asarray(expected_value)
        observed_array = np.asarray(observed_value)
        if (
            observed_array.shape != expected_array.shape
            or observed_array.dtype != expected_array.dtype
            or not _array_equal(expected_array, observed_array)
        ):
            raise ValueError(f"{context} attribute {name!r} disagrees")


def _array_equal(left: np.ndarray, right: np.ndarray) -> bool:
    if left.dtype.kind in "fc" or right.dtype.kind in "fc":
        return bool(np.array_equal(left, right, equal_nan=True))
    return bool(np.array_equal(left, right))


__all__ = [
    "LayoutDataset",
    "LayoutGroup",
    "assert_layout_trees_equal",
    "build_layout_v4_tree",
    "group_datasets",
    "iter_layout_groups",
]
