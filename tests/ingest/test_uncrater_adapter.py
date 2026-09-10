from __future__ import annotations

import sys
from collections import Counter
from dataclasses import FrozenInstanceError, dataclass, replace
from types import ModuleType, SimpleNamespace

import pytest

from lusee.ingest import uncrater_adapter as adapter
from lusee.ingest.issues import (
    IngestIssueError,
    IssueAction,
    IssueCollector,
    IssuePolicy,
    IssueSeverity,
)


class FakePacketBase:
    def read(self):
        return None


class FakeMetadata(FakePacketBase):
    pass


class FakeSpectrum(FakePacketBase):
    pass


class FakeTRSpectrum(FakePacketBase):
    pass


class FakeAuxPacket(FakePacketBase):
    pass


@dataclass(frozen=True)
class FakeDecodeIssue:
    code: str
    message: str
    appid: int | None = None
    source: str | None = None
    fatal: bool = False
    details: tuple[tuple[str, object], ...] = ()


class FakeDecodeStatus:
    def __init__(self, issues=()):
        self._issues = tuple(issues)

    @property
    def issues(self):
        return self._issues

    def counts(self):
        return dict(Counter(issue.code for issue in self._issues))


class FakeCollection:
    calls = []

    def __init__(
        self,
        directory,
        *,
        strict=False,
        diagnostic_override=False,
        schema_variant=None,
        waveform_packet_context=None,
        schema_resolution=None,
    ):
        self.calls.append(
            (directory, strict, diagnostic_override, schema_variant)
        )
        self.cont = []
        self.decode_status = FakeDecodeStatus()

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
    package.Packet_Metadata = FakeMetadata
    package.Packet_Spectrum = FakeSpectrum
    package.Packet_TR_Spectrum = FakeTRSpectrum
    for name in adapter._REQUIRED_PACKET_CLASSES:
        if not hasattr(package, name):
            setattr(package, name, FakeAuxPacket)
    package.NPRODUCTS = 16
    package.NCHANNELS = 2048
    package.normalize_dcb_appid = lambda appid: (
        0x2F0 if appid == 0x4F0 else appid
    )
    package.id = SimpleNamespace()
    for name in adapter._REQUIRED_TOP_LEVEL:
        if hasattr(package, name):
            continue
        setattr(package, name, lambda appid: False)

    registry = ModuleType("uncrater.schema_registry")
    registry.LATEST_BINDING = FakeBinding()
    registry.binding_for_key = lambda key: FakeBinding()
    registry.resolve_wire_version = lambda version: None
    registry.resolve_packet_stream = lambda *args, **kwargs: None
    registry.schema_resolution_record = lambda value: {}
    registry.schema_resolution_from_record = lambda value, **kwargs: None

    status = ModuleType("uncrater.decode_status")
    status.DecodeIssue = FakeDecodeIssue
    status.DecodeStatus = FakeDecodeStatus
    status.PacketDecodeError = type("PacketDecodeError", (ValueError,), {})

    monkeypatch.setitem(sys.modules, "uncrater", package)
    monkeypatch.setitem(sys.modules, "uncrater.schema_registry", registry)
    monkeypatch.setitem(sys.modules, "uncrater.decode_status", status)
    FakeCollection.calls.clear()
    adapter.load_uncrater.cache_clear()
    yield package
    adapter.load_uncrater.cache_clear()


def fake_packet(
    packet_class=FakeSpectrum,
    *,
    packet_index=7,
    original_appid=0x210,
    appid=0x210,
    schema_id=0x307,
    binding_key="307",
    reported_version=0x307,
    binding_variant=None,
    schema_assumed=False,
    source_release=FakeBinding.source_release,
    source_commit=FakeBinding.source_commit,
    abi_fingerprint=FakeBinding.abi_fingerprint,
    issues=(),
):
    packet = packet_class()
    packet.packet_index = packet_index
    packet.original_appid = original_appid
    packet.appid = appid
    packet.schema_id = schema_id
    packet.schema = SimpleNamespace(binding_key=binding_key)
    packet.reported_version = reported_version
    packet.binding_provenance = {
        "binding_key": binding_key,
        "canonical_schema_id": schema_id,
        "variant": binding_variant,
        "source_release": source_release,
        "source_commit": source_commit,
        "abi": {"sha256": abi_fingerprint},
    }
    packet.schema_assumed = schema_assumed
    packet.decode_status = FakeDecodeStatus(issues)
    return packet


def fake_binding(
    *,
    reported_schema_ids=(0x307,),
    selected_schema_id=0x307,
    binding_key="307",
    variant=None,
    schema_assumed=False,
) -> adapter.UncraterBindingInfo:
    return adapter.UncraterBindingInfo(
        reported_schema_ids=reported_schema_ids,
        selected_schema_id=selected_schema_id,
        binding_key=binding_key,
        variant=variant,
        schema_assumed=schema_assumed,
        source_release=FakeBinding.source_release,
        source_commit=FakeBinding.source_commit,
        abi_fingerprint=FakeBinding.abi_fingerprint,
        appid_counts=(),
        issue_counts=(),
    )


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


def test_required_packet_classes_and_dimensions_fail_closed(fake_decoder):
    fake_decoder.NPRODUCTS = 15
    adapter.load_uncrater.cache_clear()
    with pytest.raises(adapter.IncompatibleUncraterError, match="NPRODUCTS"):
        adapter.load_uncrater()

    fake_decoder.NPRODUCTS = 16
    fake_decoder.Packet_Metadata = object()
    adapter.load_uncrater.cache_clear()
    with pytest.raises(adapter.IncompatibleUncraterError, match="Packet_Metadata"):
        adapter.load_uncrater()

    fake_decoder.Packet_Metadata = FakeMetadata
    fake_decoder.Packet_Grimm = object()
    adapter.load_uncrater.cache_clear()
    with pytest.raises(adapter.IncompatibleUncraterError, match="Packet_Grimm"):
        adapter.load_uncrater()


@pytest.mark.parametrize("copy_packet_issues", [False, True])
def test_import_decode_issues_deduplicates_collection_and_packet_statuses(
    fake_decoder,
    copy_packet_issues,
):
    diagnostic = FakeDecodeIssue(
        code="crc_mismatch",
        message="CRC differs",
        appid=0x210,
        source="00007_0210.bin",
        fatal=False,
        details=(("expected", 1), ("observed", 2)),
    )
    fatal = FakeDecodeIssue(
        code="payload_decode_failed",
        message="payload is truncated",
        appid=0x210,
        source="00007_0210.bin",
        fatal=True,
        details=(("available", 3),),
    )
    assembly = FakeDecodeIssue(
        code="duplicate_numeric_index",
        message="packet index is duplicated",
        appid=0x210,
        source="00007_0210.bin",
        fatal=False,
        details=(("packet_index", 7), ("filenames", ["a.bin", "b.bin"])),
    )
    packet = fake_packet(issues=(diagnostic, fatal))
    collection = FakeCollection("capture")
    collection.cont = [packet]
    # Real Collection.decode_status retains the same immutable packet issues
    aggregate = (
        (replace(diagnostic), replace(fatal))
        if copy_packet_issues
        else (diagnostic, fatal)
    )
    collection.decode_status = FakeDecodeStatus((assembly, *aggregate))
    collector = IssueCollector()
    collector.record(
        code="framing.test",
        severity="info",
        stage="framing",
        message="preexisting",
        action="kept",
    )

    imported = adapter.import_decode_issues(
        collection,
        collector,
        fake_binding(),
    )

    assert [issue.issue_id for issue in imported.issues] == [
        "issue-00000002",
        "issue-00000003",
        "issue-00000004",
    ]
    assert [issue.code for issue in imported.issues] == [
        "decode.duplicate_numeric_index",
        "decode.crc_mismatch",
        "decode.payload_decode_failed",
    ]
    assert imported.issues[0].severity is IssueSeverity.WARNING
    assert imported.issues[0].action is IssueAction.KEPT
    assert imported.issues[1].packet_index == 7
    assert imported.issues[1].appid == 0x210
    assert imported.issues[1].input_identity == "00007_0210.bin"
    assert imported.issues[1].as_dict()["details"] == {
        "expected": 1,
        "observed": 2,
    }
    assert imported.issues[2].severity is IssueSeverity.ERROR
    assert imported.issues[2].action is IssueAction.DROPPED
    assert imported.issue_ids_by_packet_index == {
        7: (
            "issue-00000002",
            "issue-00000003",
            "issue-00000004",
        ),
    }
    assert len(collector.issues) == 4
    with pytest.raises(TypeError):
        imported.issue_ids_by_packet_index[7] = ()


def test_import_decode_issues_preserves_strict_collector_policy(fake_decoder):
    issue = FakeDecodeIssue(
        code="crc_mismatch",
        message="CRC differs",
        appid=0x210,
        source="00007_0210.bin",
    )
    collection = FakeCollection("capture")
    collection.cont = [fake_packet(issues=(issue,))]
    collection.decode_status = FakeDecodeStatus((issue,))
    collector = IssueCollector(IssuePolicy.STRICT)

    with pytest.raises(IngestIssueError) as caught:
        adapter.import_decode_issues(collection, collector, fake_binding())

    assert caught.value.issue.code == "decode.crc_mismatch"
    assert collector.issues == (caught.value.issue,)


def test_source_packet_provenance_uses_only_concrete_public_fields(fake_decoder):
    packet = fake_packet()

    provenance = adapter.source_packet_provenance(
        packet,
        role="normal_product_0",
        binding=fake_binding(),
    )

    assert provenance.role == "normal_product_0"
    assert provenance.filename is None
    assert provenance.packet_index == 7
    assert provenance.original_appid == 0x210
    assert provenance.normalized_appid == 0x210

    normalized = adapter.source_packet_provenance(
        fake_packet(original_appid=0x4F0, appid=0x2F0),
        role="waveform",
        binding=fake_binding(),
    )
    assert normalized.original_appid == 0x4F0
    assert normalized.normalized_appid == 0x2F0


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("packet_index", "packet_index"),
        ("original_appid", "original_appid"),
        ("appid", "appid"),
        ("schema_id", "schema_id"),
        ("decode_status", "decode_status"),
        ("schema", "schema"),
        ("reported_version", "reported_version"),
        ("binding_provenance", "binding_provenance"),
        ("schema_assumed", "schema_assumed"),
    ],
)
def test_source_packet_provenance_rejects_missing_public_fields(
    fake_decoder,
    field,
    message,
):
    packet = fake_packet()
    delattr(packet, field)

    with pytest.raises(adapter.IncompatibleUncraterError, match=message):
        adapter.source_packet_provenance(
            packet,
            role="normal_product_0",
            binding=fake_binding(),
        )


def test_source_packet_provenance_rejects_contract_disagreement(fake_decoder):
    wrong_appid = fake_packet(appid=0x211)
    with pytest.raises(adapter.IncompatibleUncraterError, match="AppID"):
        adapter.source_packet_provenance(
            wrong_appid,
            role="normal_product_0",
            binding=fake_binding(),
        )

    wrong_schema = fake_packet(schema_id=0x306)
    with pytest.raises(adapter.IncompatibleUncraterError, match="schema"):
        adapter.source_packet_provenance(
            wrong_schema,
            role="normal_product_0",
            binding=fake_binding(),
        )

    malformed_status = fake_packet()
    malformed_status.decode_status = SimpleNamespace(issues=())
    with pytest.raises(adapter.IncompatibleUncraterError, match="DecodeStatus"):
        adapter.source_packet_provenance(
            malformed_status,
            role="normal_product_0",
            binding=fake_binding(),
        )


def test_source_packet_provenance_rejects_wrong_306_binding(fake_decoder):
    binding = fake_binding(
        reported_schema_ids=(0x306,),
        selected_schema_id=0x306,
        binding_key="306-early",
        variant="early",
    )
    matching = fake_packet(
        schema_id=0x306,
        binding_key="306-early",
        reported_version=0x306,
        binding_variant="early",
    )
    assert adapter.source_packet_provenance(
        matching,
        role="normal_product_0",
        binding=binding,
    ).packet_index == 7

    packet = fake_packet(
        schema_id=0x306,
        binding_key="306-final",
        reported_version=0x306,
        binding_variant="final",
    )

    with pytest.raises(adapter.IncompatibleUncraterError, match="binding keys"):
        adapter.source_packet_provenance(
            packet,
            role="normal_product_0",
            binding=binding,
        )


def test_source_packet_provenance_rejects_binding_provenance_drift(fake_decoder):
    missing_binding_key = fake_packet()
    missing_binding_key.schema = SimpleNamespace()
    with pytest.raises(adapter.IncompatibleUncraterError, match="binding_key"):
        adapter.source_packet_provenance(
            missing_binding_key,
            role="normal_product_0",
            binding=fake_binding(),
        )

    wrong_reported_version = fake_packet(reported_version=0x306)
    with pytest.raises(adapter.IncompatibleUncraterError, match="reported"):
        adapter.source_packet_provenance(
            wrong_reported_version,
            role="normal_product_0",
            binding=fake_binding(),
        )

    wrong_assumption = fake_packet(schema_assumed=True)
    with pytest.raises(adapter.IncompatibleUncraterError, match="schema_assumed"):
        adapter.source_packet_provenance(
            wrong_assumption,
            role="normal_product_0",
            binding=fake_binding(),
        )

    wrong_provenance = fake_packet(source_commit="wrong")
    with pytest.raises(
        adapter.IncompatibleUncraterError,
        match="binding_provenance disagrees on source_commit",
    ):
        adapter.source_packet_provenance(
            wrong_provenance,
            role="normal_product_0",
            binding=fake_binding(),
        )

    wrong_abi = fake_packet(abi_fingerprint="wrong")
    with pytest.raises(adapter.IncompatibleUncraterError, match="ABI fingerprint"):
        adapter.source_packet_provenance(
            wrong_abi,
            role="normal_product_0",
            binding=fake_binding(),
        )


def test_import_decode_issues_rejects_wrong_306_binding_before_mutation(
    fake_decoder,
):
    binding = fake_binding(
        reported_schema_ids=(0x306,),
        selected_schema_id=0x306,
        binding_key="306-early",
        variant="early",
    )
    collection = FakeCollection("capture")
    collection.cont = [
        fake_packet(
            schema_id=0x306,
            binding_key="306-final",
            reported_version=0x306,
            binding_variant="final",
        )
    ]
    collector = IssueCollector()
    preexisting = collector.record(
        code="framing.test",
        severity="info",
        stage="framing",
        message="preexisting",
        action="kept",
    )

    with pytest.raises(adapter.IncompatibleUncraterError, match="binding keys"):
        adapter.import_decode_issues(collection, collector, binding)

    assert collector.issues == (preexisting,)


def test_import_decode_issues_validates_before_mutating_collector(fake_decoder):
    valid = FakeDecodeIssue(
        code="crc_mismatch",
        message="CRC differs",
        appid=0x210,
    )
    malformed = FakeDecodeIssue(
        code="payload_decode_failed",
        message="bad fatal field",
        appid=0x210,
        fatal=1,
    )
    packet = fake_packet(issues=(valid, malformed))
    collection = FakeCollection("capture")
    collection.cont = [packet]
    collection.decode_status = FakeDecodeStatus((valid, malformed))
    collector = IssueCollector()

    with pytest.raises(adapter.IncompatibleUncraterError, match="fatal"):
        adapter.import_decode_issues(collection, collector, fake_binding())

    assert collector.issues == ()


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


def test_collection_provenance_uses_public_report_and_packet_status(
    fake_decoder,
    monkeypatch,
):
    monkeypatch.setattr(
        adapter,
        "decoder_info",
        lambda: adapter.DecoderInfo(
            distribution_version="1.0.0",
            source_commit="00bed15f18d62530e3f706c9ce5be61255ce4740",
        ),
    )
    warning_issue = SimpleNamespace(fatal=False)
    fatal_issue = SimpleNamespace(fatal=True)
    collection = SimpleNamespace(
        cont=(
            SimpleNamespace(decode_status=SimpleNamespace(issues=())),
            SimpleNamespace(decode_status=SimpleNamespace(issues=(warning_issue,))),
            SimpleNamespace(decode_status=SimpleNamespace(issues=(fatal_issue,))),
        ),
        reported_schema_ids=(0x307,),
        selected_schema_ids=(0x307,),
        selected_schema_bindings=("307",),
        schema_assumed=False,
        packet_counts_by_appid=Counter({0x20F: 1, 0x210: 2}),
        decode_status=SimpleNamespace(counts=lambda: {"crc_mismatch": 1}),
        canonical_report=lambda: {
            "report_schema_version": 1,
            "reported_schema_ids": ["0x307"],
            "selected_schema_ids": ["0x307"],
            "selected_schema_bindings": ["307"],
            "schema_assumed": False,
            "packet_count": 3,
            "packet_counts_by_appid": {"0x20F": 1, "0x210": 2},
            "invalid_counts_by_issue": {"crc_mismatch": 1},
        },
    )

    provenance = adapter.collection_provenance(collection, strict=True)

    assert provenance.input_packet_count == 3
    assert provenance.valid_packet_count == 2
    assert provenance.execution_mode.value == "strict"
    assert provenance.canonical_report()["packet_count"] == 3
    from lusee.ingest.decode import _decode_quality

    assert _decode_quality(provenance, usable_product_count=1).value == "partial"
    assert _decode_quality(provenance, usable_product_count=0).value == "failed"


def test_collection_provenance_rejects_missing_status_and_report_drift(
    fake_decoder,
    monkeypatch,
):
    monkeypatch.setattr(
        adapter,
        "decoder_info",
        lambda: adapter.DecoderInfo(None, None),
    )
    report = {
        "report_schema_version": 1,
        "reported_schema_ids": [],
        "selected_schema_ids": ["0x307"],
        "selected_schema_bindings": ["307"],
        "schema_assumed": True,
        "packet_count": 1,
        "packet_counts_by_appid": {"0x20F": 1},
        "invalid_counts_by_issue": {},
    }
    collection = SimpleNamespace(
        cont=(SimpleNamespace(),),
        reported_schema_ids=(),
        selected_schema_ids=(0x307,),
        selected_schema_bindings=("307",),
        schema_assumed=True,
        packet_counts_by_appid=Counter({0x20F: 1}),
        decode_status=SimpleNamespace(counts=lambda: {}),
        canonical_report=lambda: report,
    )

    with pytest.raises(adapter.IncompatibleUncraterError, match="decode_status"):
        adapter.collection_provenance(collection, strict=False)

    collection.cont = (
        SimpleNamespace(decode_status=SimpleNamespace(issues=())),
    )
    report.pop("report_schema_version")
    with pytest.raises(
        adapter.IncompatibleUncraterError,
        match="report_schema_version",
    ):
        adapter.collection_provenance(collection, strict=False)

    report["report_schema_version"] = 2
    with pytest.raises(
        adapter.IncompatibleUncraterError,
        match="report_schema_version",
    ):
        adapter.collection_provenance(collection, strict=False)

    report["report_schema_version"] = 1
    report["packet_count"] = 2
    with pytest.raises(adapter.IncompatibleUncraterError, match="packet_count"):
        adapter.collection_provenance(collection, strict=False)


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
    decode_provenance = adapter.collection_provenance(
        collection, strict=False
    )
    imported = adapter.import_decode_issues(
        collection,
        IssueCollector(),
        provenance,
    )

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
    assert decode_provenance.input_packet_count == 0
    assert decode_provenance.valid_packet_count == 0
    assert decode_provenance.execution_mode.value == "collect"
    assert decode_provenance.canonical_report() == report
    assert dict(imported.issue_ids_by_packet_index) == {}
    assert imported.issues == ()

    from lusee.ingest.decode import read_uncrater_session

    products = read_uncrater_session(tmp_path, strict=False)
    assert products.decode_provenance.unavailable_reason is None
    assert products.packet_map_status == "unavailable"
    assert products.packet_map_format_version is None
    assert (
        products.raw_flash_provenance_unavailable_reason
        == "packet_map_missing_legacy_session"
    )
    assert products.decode_provenance.canonical_report()["packet_count"] == 0
    assert products.validated_counts.input_packets == 0
    assert products.validated_counts.valid_packets == 0
    assert products.validated_counts.product_rows == ()
    assert products.validated_counts.persisted_rows is None

    from lusee.ingest import collation

    assert collation.is_uid_typed(0x280)
    assert not collation.is_uid_prefixed(0x280)
    assert collation.is_uid_prefixed(0x281)
