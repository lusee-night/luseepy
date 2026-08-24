from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from lusee.ingest import dependencies


def test_light_imports_and_stage2_work_without_ingest_extras():
    script = textwrap.dedent(
        r'''
        import importlib.abc
        import importlib.util
        import sys

        blocked = {"h5py", "uncrater"}
        original_find_spec = importlib.util.find_spec

        class BlockLoader(importlib.abc.Loader):
            def __init__(self, root):
                self.root = root

            def create_module(self, spec):
                return None

            def exec_module(self, module):
                raise ModuleNotFoundError(
                    f"blocked optional dependency {self.root}",
                    name=self.root,
                )

        class BlockFinder(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                root = fullname.partition(".")[0]
                if root not in blocked:
                    return None
                return importlib.util.spec_from_loader(
                    fullname,
                    BlockLoader(root),
                    is_package=(fullname == root),
                )

        def find_spec(name, package=None):
            if name.partition(".")[0] in blocked:
                return None
            return original_find_spec(name, package)

        importlib.util.find_spec = find_spec
        sys.meta_path.insert(0, BlockFinder())

        import lusee.ingest as ingest

        assert not blocked.intersection(sys.modules)
        payload = b"stage2"
        appid = 0x234
        sequence = 17
        header = (
            appid.to_bytes(2, "big")
            + ((3 << 14) | sequence).to_bytes(2, "big")
            + (len(payload) - 1).to_bytes(2, "big")
        )
        crc = ingest.crc16_ccitt(header + payload).to_bytes(2, "big")
        frames = list(ingest.parse_stream(b"\xec\xa0" + header + payload + crc))
        packets = list(
            ingest.reassemble_logical_packets(frames, byteswap_pairs=False)
        )
        assert len(packets) == 1
        assert packets[0].appid == appid
        assert packets[0].seq == sequence
        assert packets[0].blob == payload

        assert "Products" not in ingest.__dict__
        products = ingest.Products
        assert products is ingest.Products
        assert products.__module__ == "lusee.ingest.decode"
        assert products is __import__(
            "lusee.ingest.decode", fromlist=["Products"]
        ).Products
        assert callable(ingest.write_fits)
        for name in ingest.__all__:
            assert getattr(ingest, name) is not None
        assert not blocked.intersection(sys.modules)

        try:
            ingest.read_uncrater_session("missing")
        except ingest.MissingIngestExtraError as exc:
            assert exc.name == "uncrater"
            assert exc.dependency == "uncrater"
            assert "packet identity assignment and decoding" in str(exc)
            assert 'pip install "lusee[ingest]"' in str(exc)
        else:
            raise AssertionError("missing uncrater was not reported")

        try:
            ingest.write_hdf5(products(), "unused.h5")
        except ingest.MissingIngestExtraError as exc:
            assert exc.name == "h5py"
            assert exc.dependency == "h5py"
            assert "HDF5 ingest output" in str(exc)
            assert 'pip install "lusee[ingest]"' in str(exc)
        else:
            raise AssertionError("missing h5py was not reported")

        try:
            ingest.does_not_exist
        except AttributeError:
            pass
        else:
            raise AssertionError("unknown lazy attribute did not fail")
        '''
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("dependency", ["h5py", "uncrater"])
def test_dependency_helper_classifies_only_the_requested_package(
    dependency,
    monkeypatch,
):
    def missing(name):
        raise ModuleNotFoundError("absent", name=name)

    monkeypatch.setattr(
        dependencies,
        "importlib",
        type("FakeImportlib", (), {"import_module": staticmethod(missing)}),
    )
    with pytest.raises(dependencies.MissingIngestExtraError) as caught:
        dependencies.import_optional_dependency(dependency, "test feature")
    assert caught.value.name == dependency
    assert caught.value.dependency == dependency


def test_dependency_helper_does_not_hide_broken_transitive_import(monkeypatch):
    def broken(name):
        raise ModuleNotFoundError("broken leaf", name="broken_leaf")

    monkeypatch.setattr(
        dependencies,
        "importlib",
        type("FakeImportlib", (), {"import_module": staticmethod(broken)}),
    )
    with pytest.raises(ModuleNotFoundError) as caught:
        dependencies.import_optional_dependency("uncrater", "packet decoding")
    assert type(caught.value) is ModuleNotFoundError
    assert caught.value.name == "broken_leaf"
