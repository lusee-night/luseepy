"""Smoke test: lusee.ingest imports cleanly even without uncrater installed."""

from __future__ import annotations

import importlib


def test_lusee_imports():
    lusee = importlib.import_module("lusee")
    # ingest may be None if h5py is missing; both states are acceptable here.
    assert hasattr(lusee, "ingest")


def test_lusee_ingest_imports():
    li = importlib.import_module("lusee.ingest")
    expected = [
        "process_flash", "process_session", "parse_flash",
        "parse_stream", "parse_bank_file",
        "reassemble_logical_packets", "assign_identities", "split_sessions",
        "write_uncrater_session", "read_uncrater_session", "write_hdf5",
        "SessionResult", "write_manifest",
        "Session", "LogicalPacket", "Products",
    ]
    for name in expected:
        assert getattr(li, name, None) is not None, f"missing: {name}"
