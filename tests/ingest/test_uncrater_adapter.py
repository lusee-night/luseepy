from __future__ import annotations

import sys
from collections import Counter
from dataclasses import FrozenInstanceError
from types import ModuleType, SimpleNamespace

import pytest

from lusee.ingest import uncrater_adapter as adapter


class FakePacketBase:
    def read(self):
        return None


class FakeCollection:
    calls = []

    def __init__(
        self,
        directory,
        *,
        strict=False,
        diagnostic_override=False,
        schema_variant=None,
    ):
        self.calls.append(
            (directory, strict, diagnostic_override, schema_variant)
        )

    def canonical_report(self):
        return {"public": True}


class FakeBinding:
    binding_key = "307"
    canonical_schema_id = 0x307
    variant = None
    source_release = "3r09"
    source_commit = "38770b94bd20dfdbf52d1d8b872a716537b851f3"
    abi_fingerprint = (
        "23509698f27914dc881af7da335997aed308a45710ca8a9e6c5ace224a949f7e"
    )


@pytest.fixture
def fake_decoder(monkeypatch):
    package = ModuleType("uncrater")
    package.__path__ = []
    package.__version__ = "1.0.0"
    package.PacketBase = FakePacketBase
    package.Collection = FakeCollection
    package.Packet = lambda *args, **kwargs: None
    package.id = SimpleNamespace()
    for name in adapter._REQUIRED_TOP_LEVEL:
        if name in {"PacketBase", "Collection", "Packet", "id"}:
            continue
        setattr(package, name, lambda appid: False)

    registry = ModuleType("uncrater.schema_registry")
    registry.LATEST_BINDING = FakeBinding()
    registry.binding_for_key = lambda key: FakeBinding()
    registry.resolve_wire_version = lambda version: None

    status = ModuleType("uncrater.decode_status")
    status.DecodeIssue = type("DecodeIssue", (), {})
    status.DecodeStatus = type("DecodeStatus", (), {})
    status.PacketDecodeError = type("PacketDecodeError", (ValueError,), {})

    monkeypatch.setitem(sys.modules, "uncrater", package)
    monkeypatch.setitem(sys.modules, "uncrater.schema_registry", registry)
    monkeypatch.setitem(sys.modules, "uncrater.decode_status", status)
    FakeCollection.calls.clear()
    adapter.load_uncrater.cache_clear()
    yield package
    adapter.load_uncrater.cache_clear()


def test_public_only_decoder_and_collection_policy_forwarding(fake_decoder):
    assert adapter.load_uncrater() is fake_decoder

    class Packet:
        def __init__(self):
            self.read_calls = 0

        def read(self):
            self.read_calls += 1

    packet = Packet()
    adapter.read_packet(packet)
    assert packet.read_calls == 1
    assert not hasattr(packet, "_read")

    first = adapter.make_collection("capture")
    second = adapter.make_collection(
        "capture",
        strict=True,
        diagnostic_override=True,
        schema_variant="early",
    )
    assert first.canonical_report() == {"public": True}
    assert second.canonical_report() == {"public": True}
    assert FakeCollection.calls == [
        ("capture", False, False, None),
        ("capture", True, True, "early"),
    ]


def test_decoder_and_binding_provenance_is_immutable_and_sorted(
    fake_decoder,
    monkeypatch,
):
    direct_url = (
        '{"vcs_info":{"vcs":"git","commit_id":'
        '"00bed15f18d62530e3f706c9ce5be61255ce4740"}}'
    )
    distribution = SimpleNamespace(
        version="1.0.0",
        read_text=lambda name: direct_url if name == "direct_url.json" else None,
    )
    monkeypatch.setattr(adapter.metadata, "distribution", lambda name: distribution)

    decoder = adapter.decoder_info()
    assert decoder.distribution_version == "1.0.0"
    assert decoder.source_commit == "00bed15f18d62530e3f706c9ce5be61255ce4740"

    collection = SimpleNamespace(
        reported_schema_ids=(0x305,),
        selected_schema_ids=(0x307,),
        selected_schema_bindings=("307",),
        schema_assumed=True,
        packet_counts_by_appid=Counter({0x2A0: 1, 0x209: 2}),
        decode_status=SimpleNamespace(
            counts=lambda: {"z_issue": 1, "a_issue": 2}
        ),
    )
    provenance = adapter.binding_info(collection)
    assert provenance.reported_schema_ids == (0x305,)
    assert provenance.selected_schema_id == 0x307
    assert provenance.binding_key == "307"
    assert provenance.schema_assumed is True
    assert provenance.source_release == "3r09"
    assert provenance.source_commit == FakeBinding.source_commit
    assert provenance.abi_fingerprint == FakeBinding.abi_fingerprint
    assert provenance.appid_counts == ((0x209, 2), (0x2A0, 1))
    assert provenance.issue_counts == (("a_issue", 2), ("z_issue", 1))
    with pytest.raises(FrozenInstanceError):
        provenance.binding_key = "changed"


def test_boundary_rejects_an_incompatible_decoder(fake_decoder):
    fake_decoder.PacketBase = type("PacketBaseWithoutRead", (), {})
    adapter.load_uncrater.cache_clear()
    with pytest.raises(adapter.IncompatibleUncraterError, match="read"):
        adapter.load_uncrater()


def test_binding_provenance_rejects_inconsistent_selection(fake_decoder):
    collection = SimpleNamespace(
        reported_schema_ids=(),
        selected_schema_ids=(0x305,),
        selected_schema_bindings=("307",),
        schema_assumed=False,
        packet_counts_by_appid={},
        decode_status=SimpleNamespace(counts=lambda: {}),
    )
    with pytest.raises(adapter.IncompatibleUncraterError, match="disagree"):
        adapter.binding_info(collection)


def test_real_empty_collection_matches_pinned_decoder(tmp_path):
    adapter.load_uncrater.cache_clear()
    try:
        decoder = adapter.load_uncrater()
    except (ModuleNotFoundError, adapter.IncompatibleUncraterError):
        pytest.skip("repaired uncrater is not installed")
    if not getattr(decoder, "__file__", None):
        pytest.skip("repaired uncrater is not installed")

    collection = adapter.make_collection(tmp_path)
    provenance = adapter.binding_info(collection)
    report = collection.canonical_report()

    assert decoder.__version__ == "1.0.0"
    decoder_provenance = adapter.decoder_info()
    assert decoder_provenance.distribution_version == "1.0.0"
    assert decoder_provenance.source_commit in (
        None,
        "00bed15f18d62530e3f706c9ce5be61255ce4740",
    )
    assert collection.reported_schema_ids == ()
    assert collection.selected_schema_ids == (0x307,)
    assert collection.selected_schema_bindings == ("307",)
    assert collection.schema_assumed is True
    assert dict(collection.packet_counts_by_appid) == {}
    assert dict(collection.invalid_counts_by_issue) == {}
    assert collection.decode_status.ok is True
    assert provenance.binding_key == "307"
    assert provenance.selected_schema_id == 0x307
    assert provenance.source_release == "3r09"
    assert provenance.source_commit == FakeBinding.source_commit
    assert provenance.abi_fingerprint == FakeBinding.abi_fingerprint
    assert provenance.appid_counts == ()
    assert provenance.issue_counts == ()
    assert report["report_schema_version"] == 1
    assert report["reported_schema_ids"] == []
    assert report["selected_schema_ids"] == ["0x307"]
    assert report["selected_schema_bindings"] == ["307"]
    assert report["schema_assumed"] is True
    assert report["packet_count"] == 0

    from lusee.ingest import collation

    assert collation.is_uid_typed(0x280)
    assert not collation.is_uid_prefixed(0x280)
    assert collation.is_uid_prefixed(0x281)
