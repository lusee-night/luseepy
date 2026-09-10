"""Optional dependency loading for :mod:`lusee.ingest`."""

from __future__ import annotations

import importlib
from types import ModuleType


class MissingIngestExtraError(ModuleNotFoundError):
    """An ingest entry point needs a package from the ``ingest`` extra."""

    def __init__(self, dependency: str, feature: str):
        self.dependency = dependency
        self.feature = feature
        super().__init__(
            f"{dependency!r} is required for {feature}; install the ingest "
            'extras with pip install "lusee[ingest]"',
            name=dependency,
        )


def import_optional_dependency(dependency: str, feature: str) -> ModuleType:
    """Import one optional package without hiding broken transitive imports."""
    try:
        return importlib.import_module(dependency)
    except ModuleNotFoundError as exc:
        if exc.name != dependency:
            raise
        raise MissingIngestExtraError(dependency, feature) from exc
