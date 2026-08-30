"""Minimal GGUF header metadata reader for classify.py's KV-cache math.
Spec: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
"""
import struct

_TYPE_UINT8, _TYPE_INT8, _TYPE_UINT16, _TYPE_INT16 = 0, 1, 2, 3
_TYPE_UINT32, _TYPE_INT32, _TYPE_FLOAT32, _TYPE_BOOL = 4, 5, 6, 7
_TYPE_STRING, _TYPE_ARRAY, _TYPE_UINT64, _TYPE_INT64, _TYPE_FLOAT64 = 8, 9, 10, 11, 12

_SCALAR_FORMATS = {
    _TYPE_UINT8: ("<B", 1), _TYPE_INT8: ("<b", 1),
    _TYPE_UINT16: ("<H", 2), _TYPE_INT16: ("<h", 2),
    _TYPE_UINT32: ("<I", 4), _TYPE_INT32: ("<i", 4),
    _TYPE_FLOAT32: ("<f", 4), _TYPE_BOOL: ("<B", 1),
    _TYPE_UINT64: ("<Q", 8), _TYPE_INT64: ("<q", 8),
    _TYPE_FLOAT64: ("<d", 8),
}


def _read_string(f) -> str:
    (length,) = struct.unpack("<Q", f.read(8))
    return f.read(length).decode("utf-8", errors="replace")


def _read_value(f, value_type: int):
    if value_type == _TYPE_STRING:
        return _read_string(f)
    if value_type == _TYPE_ARRAY:
        (elem_type,) = struct.unpack("<I", f.read(4))
        (count,) = struct.unpack("<Q", f.read(8))
        return [_read_value(f, elem_type) for _ in range(count)]
    fmt, size = _SCALAR_FORMATS[value_type]
    return struct.unpack(fmt, f.read(size))[0]


def read_metadata(path) -> dict:
    """Returns the raw GGUF metadata dict (dotted keys -> values), or {} on any parse failure."""
    try:
        with open(path, "rb") as f:
            magic = f.read(4)
            if magic != b"GGUF":
                return {}
            (version,) = struct.unpack("<I", f.read(4))
            (tensor_count,) = struct.unpack("<Q", f.read(8))
            (kv_count,) = struct.unpack("<Q", f.read(8))
            meta = {}
            for _ in range(kv_count):
                key = _read_string(f)
                (value_type,) = struct.unpack("<I", f.read(4))
                meta[key] = _read_value(f, value_type)
            return meta
    except Exception:
        return {}


def to_hf_style_config(meta: dict) -> dict:
    """Translates GGUF's `{arch}.*` keys into the HF config.json field names classify.py expects.
    Hybrid architectures store `{arch}.full_attention_interval` instead of a per-layer layer_types list."""
    arch = meta.get("general.architecture")
    if not arch:
        return {}

    num_layers = meta.get(f"{arch}.block_count")
    full_attention_interval = meta.get(f"{arch}.full_attention_interval")
    layer_types = None
    if num_layers and full_attention_interval:
        layer_types = [
            "full_attention" if (i + 1) % full_attention_interval == 0 else "linear_attention"
            for i in range(num_layers)
        ]

    return {
        "max_position_embeddings": meta.get(f"{arch}.context_length"),
        "num_hidden_layers": num_layers,
        "layer_types": layer_types,
        "hidden_size": meta.get(f"{arch}.embedding_length"),
        "num_attention_heads": meta.get(f"{arch}.attention.head_count"),
        "num_key_value_heads": meta.get(f"{arch}.attention.head_count_kv"),
        "head_dim": meta.get(f"{arch}.attention.key_length"),  # more accurate than hidden_size / heads
    }
