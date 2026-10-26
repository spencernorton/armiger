"""Explicit R13 tensor ABI: 40 S2 features plus raw title byte length.

The last scalar is a mask boundary, never a learned feature. Both views share
the transformer; the title head cannot attend to appended file names.
"""

from __future__ import annotations

import numpy as np

try:
    from . import r13_feature_extractor as s2
except ImportError:
    try:
        import r13_feature_extractor as s2
    except ImportError:
        from r13_master_program.s2_neural_features import r13_feature_extractor as s2

BASE_STATS_DIM, SIDE_DIM, STATS_DIM = 12, 28, 41
INPUTS = ("name_bytes", "name_len", "ext", "size", "depth", "words", "stats")


def encode_torrent_features(row, max_name_len=256, max_files=64):
    encoded = s2.encode_torrent_features(row, max_name_len, max_files)
    title = encoded["title_name_bytes"]
    # S2's enriched representation starts with the supplied title. Empty titles
    # have a stable space token rather than consuming any file-derived byte.
    if not str(
        s2.inference_input(row).get("name")
        or s2.inference_input(row).get("title")
        or ""
    ):
        encoded["name_bytes"] = [32]
    if encoded["name_bytes"][: len(title)] != title:
        raise ValueError("Enriched input no longer preserves the raw title prefix")
    encoded["stats"] = list(encoded["stats"]) + [float(len(title))]
    return encoded


def build_onnx_batch_inputs(rows, max_name_len=256, max_files=64, stats_dim=STATS_DIM):
    if isinstance(stats_dim, bool) or not isinstance(stats_dim, (int, np.integer)) or stats_dim != STATS_DIM:
        raise ValueError("R13 requires exactly 41 stats components; padding or truncation changes its feature ABI")
    if not rows:
        raise ValueError("Cannot collate an empty batch")
    enc = [encode_torrent_features(r, max_name_len, max_files) for r in rows]
    b, length, files = (
        len(enc),
        max(len(e["name_bytes"]) for e in enc),
        max(1, max(len(e["ext"]) for e in enc)),
    )
    out = {
        "name_bytes": np.zeros((b, length), np.int64),
        "name_len": np.zeros(b, np.int64),
        "ext": np.zeros((b, files), np.int32),
        "size": np.zeros((b, files), np.int64),
        "depth": np.zeros((b, files), np.int64),
        "words": np.zeros((b, files, 4), np.int32),
        "stats": np.zeros((b, stats_dim), np.float32),
    }
    for i, e in enumerate(enc):
        n, f = len(e["name_bytes"]), len(e["ext"])
        out["name_bytes"][i, :n], out["name_len"][i] = e["name_bytes"], n
        for k in ("ext", "size", "depth", "words"):
            out[k][i, :f] = e[k]
        out["stats"][i] = e["stats"]
    return out
