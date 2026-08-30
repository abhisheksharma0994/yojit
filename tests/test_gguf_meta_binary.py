"""Exercises gguf_meta's actual binary parsing (read_metadata/_read_value/_read_string)
against real GGUF-format bytes, not just the already-parsed dicts test_gguf_meta.py uses."""
import struct

from yojit import gguf_meta


def _pack_string(s: str) -> bytes:
    encoded = s.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _pack_kv(key: str, value_type: int, value_bytes: bytes) -> bytes:
    return _pack_string(key) + struct.pack("<I", value_type) + value_bytes


def _build_gguf(kv_pairs: list[tuple[str, int, bytes]], magic=b"GGUF") -> bytes:
    header = magic + struct.pack("<I", 3)  # version 3
    header += struct.pack("<Q", 0)  # tensor_count
    header += struct.pack("<Q", len(kv_pairs))  # kv_count
    body = b"".join(_pack_kv(k, t, v) for k, t, v in kv_pairs)
    return header + body


def test_read_metadata_returns_empty_dict_for_wrong_magic(tmp_path):
    path = tmp_path / "not-gguf.bin"
    path.write_bytes(b"NOPE" + b"\x00" * 20)
    assert gguf_meta.read_metadata(path) == {}


def test_read_metadata_returns_empty_dict_for_missing_file(tmp_path):
    assert gguf_meta.read_metadata(tmp_path / "does-not-exist.gguf") == {}


def test_read_metadata_parses_string_value(tmp_path):
    data = _build_gguf([("general.architecture", gguf_meta._TYPE_STRING, _pack_string("llama"))])
    path = tmp_path / "model.gguf"
    path.write_bytes(data)
    assert gguf_meta.read_metadata(path) == {"general.architecture": "llama"}


def test_read_metadata_parses_scalar_types(tmp_path):
    data = _build_gguf([
        ("llama.block_count", gguf_meta._TYPE_UINT32, struct.pack("<I", 22)),
        ("llama.context_length", gguf_meta._TYPE_UINT64, struct.pack("<Q", 2048)),
        ("some.float", gguf_meta._TYPE_FLOAT32, struct.pack("<f", 1.5)),
        ("some.bool", gguf_meta._TYPE_BOOL, struct.pack("<B", 1)),
    ])
    path = tmp_path / "model.gguf"
    path.write_bytes(data)
    meta = gguf_meta.read_metadata(path)
    assert meta["llama.block_count"] == 22
    assert meta["llama.context_length"] == 2048
    assert meta["some.float"] == 1.5
    assert meta["some.bool"] == 1


def test_read_metadata_parses_array_value(tmp_path):
    array_bytes = struct.pack("<I", gguf_meta._TYPE_UINT32) + struct.pack("<Q", 3) + \
        struct.pack("<I", 1) + struct.pack("<I", 2) + struct.pack("<I", 3)
    data = _build_gguf([("some.array", gguf_meta._TYPE_ARRAY, array_bytes)])
    path = tmp_path / "model.gguf"
    path.write_bytes(data)
    assert gguf_meta.read_metadata(path) == {"some.array": [1, 2, 3]}


def test_read_metadata_returns_empty_dict_on_truncated_file(tmp_path):
    path = tmp_path / "truncated.gguf"
    path.write_bytes(b"GGUF" + struct.pack("<I", 3))  # cut off mid-header
    assert gguf_meta.read_metadata(path) == {}
