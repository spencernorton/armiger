"""Frozen-Trunk 9-Head Delta Bundler & Reconstructor (`armiger/dht_head_delta.py`).

Implements Session 3's Frozen-Trunk 9-Head Delta architecture for decentralized DHT model updates:
  - Armiger v3.0 total parameters: `8,112,098`
  - Shared frozen backbone trunk: `7,281,344` params (`89.76%`)
  - All 9 task heads combined (`category`, `adult`, `junk`, `anime`, `english_subs`, `lang`,
    `owner_scope`, `title_language`, `span_bio`): `830,754` params (`10.24%`)
  - 9th `span_bio` token-classification head alone: `68,103` params (`0.84%`)

DHT Payload Sizing vs `32.54 MB` (`32,537,454 B`) Full FP32 ONNX:
  - 9-Head Delta (`FP16`): `1,661,508` weight bytes (`1.66 MB`, `-94.89%`)
  - 9-Head Delta (`INT8`): `830,754` weight bytes (`0.83 MB`, `-97.45%`)
  - Single-Head `span_bio` Delta (`INT8`): `68,103` weight bytes (`68.1 KB`, `-99.79%`)
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import dataclasses
import hashlib
import json
import lzma
import math
from pathlib import Path
import struct
from typing import Any
import zlib

import numpy as np

HDLT_MAGIC = b"HDLT"
HDLT_HEADER_PREFIX = struct.Struct("<4sI")
MAX_HDLT_DECOMPRESSED_BYTES = 32 * 1024 * 1024

ALL_9_HEAD_NAMES: tuple[str, ...] = (
    "category",
    "adult",
    "junk",
    "anime",
    "english_subs",
    "lang",
    "owner_scope",
    "title_language",
    "span_bio",
)

EXPECTED_TOTAL_PARAMS = 8_112_098
EXPECTED_TRUNK_PARAMS = 7_281_344
EXPECTED_ALL_9_HEADS_PARAMS = 830_754
EXPECTED_ALL_HEADS_PARAMS = EXPECTED_ALL_9_HEADS_PARAMS
EXPECTED_SPAN_BIO_PARAMS = 68_103


def _to_numpy_fp32(tensor_or_arr: Any) -> np.ndarray:
    """Convert a PyTorch Tensor or NumPy array into a contiguous little-endian `float32` ndarray."""
    if hasattr(tensor_or_arr, "detach"):
        arr = tensor_or_arr.detach().cpu().numpy()
    else:
        arr = np.asarray(tensor_or_arr)
    return np.ascontiguousarray(arr, dtype=np.float32)


def is_head_tensor_key(key: str) -> bool:
    """Return True iff a state_dict/initializer key belongs to an anchored `heads.<head>.<param>` task head."""
    k = canonical_head_key(str(key).strip())
    if not k.startswith("heads."):
        return False
    parts = k.split(".", 2)
    return len(parts) == 3 and parts[1] in ALL_9_HEAD_NAMES and bool(parts[2])


def canonical_head_key(key: str) -> str:
    """Normalize `m.heads.<head>...` to `heads.<head>...`."""
    return key.removeprefix("m.")


def _extract_onnx_tensors(model_proto: Any) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, tuple[str, bool]]]:
    """Extract `(trunk_tensors, head_tensors, onnx_head_mapping)` from an `onnx.ModelProto`.

    In exported Armiger v3 ONNX graphs, `m.heads.<h>.*` are named directly except for the two
    3D `span_bio` `MatMul` weight tensors (`heads.span_bio.0.weight.T` of shape `(256, 256)` and
    `heads.span_bio.3.weight.T` of shape `(256, 7)`), which are traced directly from the
    `span_bio` output subgraph.
    """
    from onnx import numpy_helper

    init_by_name: dict[str, np.ndarray] = {
        init.name: _to_numpy_fp32(numpy_helper.to_array(init))
        for init in model_proto.graph.initializer
    }

    # Trace the `span_bio` subgraph to identify its two MatMul weight initializers
    span_matmul_map: dict[str, tuple[str, bool]] = {}
    node_by_out: dict[str, Any] = {}
    for node in model_proto.graph.node:
        for out_name in node.output:
            node_by_out[out_name] = node

    if any(o.name == "span_bio" for o in model_proto.graph.output):
        # Walk backwards from output "span_bio"
        visited: set[str] = set()
        queue: list[str] = ["span_bio"]
        matmul_inits: list[str] = []
        while queue:
            curr = queue.pop(0)
            if curr in visited:
                continue
            visited.add(curr)
            nd = node_by_out.get(curr)
            if nd is None:
                continue
            if nd.op_type == "MatMul":
                for inp in nd.input:
                    if inp in init_by_name:
                        matmul_inits.append(inp)
            # Do not walk past trunk LayerNormalization (`m.name_encoder.ln`) or non-head `/m/` backbone nodes
            for inp in nd.input:
                if inp in init_by_name:
                    continue
                nd_inp = node_by_out.get(inp)
                if nd_inp is not None and nd_inp.op_type == "LayerNormalization":
                    if not any(is_head_tensor_key(i) for i in nd_inp.input if i in init_by_name):
                        continue
                if inp.startswith("/m/") and not inp.startswith(("/m/heads/", "/m/head_", "/m/span_bio")):
                    continue
                queue.append(inp)

        assigned_targets: set[str] = set()
        for init_name in matmul_inits:
            shape = init_by_name[init_name].shape
            if len(shape) == 2 and shape[1] == 7 and shape[0] != 7 and "heads.span_bio.3.weight" not in assigned_targets:
                span_matmul_map[init_name] = ("heads.span_bio.3.weight", True)
                assigned_targets.add("heads.span_bio.3.weight")
            elif len(shape) == 2 and shape[0] == shape[1] and "heads.span_bio.0.weight" not in assigned_targets:
                span_matmul_map[init_name] = ("heads.span_bio.0.weight", True)
                assigned_targets.add("heads.span_bio.0.weight")

    trunk: dict[str, np.ndarray] = {}
    heads: dict[str, np.ndarray] = {}
    onnx_map: dict[str, tuple[str, bool]] = {}

    for name, arr in init_by_name.items():
        if name in span_matmul_map:
            canon_name, transpose_flag = span_matmul_map[name]
            heads[canon_name] = np.ascontiguousarray(arr.T if transpose_flag else arr, dtype=np.float32)
            onnx_map[canon_name] = (name, transpose_flag)
        elif is_head_tensor_key(name):
            canon_name = canonical_head_key(name)
            heads[canon_name] = arr
            onnx_map[canon_name] = (name, False)
        else:
            trunk[name] = arr

    return trunk, heads, onnx_map


def load_state_tensors(source: Any) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Split a checkpoint path, PyTorch `state_dict`, or `onnx.ModelProto` into `(trunk_tensors, head_tensors)`."""
    if hasattr(source, "graph") and hasattr(source.graph, "initializer"):
        trunk, heads, _ = _extract_onnx_tensors(source)
        return trunk, heads

    if isinstance(source, (str, Path)):
        p = Path(source)
        if p.suffix.lower() == ".onnx":
            import onnx

            m = onnx.load(str(p))
            trunk, heads, _ = _extract_onnx_tensors(m)
            return trunk, heads
        import torch

        ckpt = torch.load(str(p), map_location="cpu", weights_only=True)
        sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, Mapping) else ckpt
        return load_state_tensors(sd)

    if isinstance(source, Mapping):
        sd = source.get("model_state_dict", source) if "model_state_dict" in source and isinstance(source["model_state_dict"], Mapping) else source
        trunk: dict[str, np.ndarray] = {}
        heads: dict[str, np.ndarray] = {}
        for k, v in sd.items():
            if not hasattr(v, "shape") and not isinstance(v, (np.ndarray, list, tuple)):
                continue
            arr = _to_numpy_fp32(v)
            if is_head_tensor_key(k):
                heads[canonical_head_key(k)] = arr
            else:
                trunk[k] = arr
        return trunk, heads

    raise TypeError(f"Unsupported model/checkpoint source type: {type(source)!r}")


def compute_trunk_sha256(trunk_tensors: Mapping[str, Any]) -> str:
    """Compute a deterministic SHA-256 digest over all sorted frozen-trunk tensors."""
    hasher = hashlib.sha256()
    hasher.update(b"armiger-frozen-trunk-v3:")
    for k in sorted(trunk_tensors.keys()):
        arr = _to_numpy_fp32(trunk_tensors[k])
        shape_str = ",".join(str(d) for d in arr.shape)
        hasher.update(f"{k}:{shape_str}:".encode("utf-8"))
        hasher.update(arr.astype("<f4", copy=False).tobytes())
    return hasher.hexdigest()


@dataclasses.dataclass(frozen=True)
class HeadDeltaBundle:
    """Serialized Frozen-Trunk 9-Head (or subset-head) Delta Bundle for BEP-44/46 DHT distribution."""

    schema: str
    version: str
    precision: str
    base_trunk_sha256: str
    trunk_param_count: int
    head_param_count: int
    head_names: tuple[str, ...]
    tensor_specs: tuple[dict[str, Any], ...]
    raw_weight_bytes: bytes
    payload_sha256: str
    seq: int = 1
    ed25519_signature_hex: str = ""
    xmss_signature_hex: str = ""

    @property
    def weight_size_bytes(self) -> int:
        return len(self.raw_weight_bytes)

    def signing_payload(self) -> bytes:
        canonical_specs = [
            {
                "name": str(s["name"]),
                "shape": [int(d) for d in s["shape"]],
                "numel": int(s["numel"]),
                "offset": int(s["offset"]),
                "nbytes": int(s["nbytes"]),
                "scale": float(s.get("scale", 1.0)),
            }
            for s in self.tensor_specs
        ]
        doc = {
            "base_trunk_sha256": self.base_trunk_sha256.lower(),
            "head_names": list(self.head_names),
            "head_param_count": int(self.head_param_count),
            "payload_sha256": self.payload_sha256.lower(),
            "precision": self.precision,
            "schema": self.schema,
            "seq": int(self.seq),
            "tensor_specs": canonical_specs,
            "trunk_param_count": int(self.trunk_param_count),
            "version": self.version,
        }
        return json.dumps(doc, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def to_bytes(self, *, compression: str = "none") -> bytes:
        meta = {
            "schema": self.schema,
            "version": self.version,
            "precision": self.precision,
            "base_trunk_sha256": self.base_trunk_sha256,
            "trunk_param_count": self.trunk_param_count,
            "head_param_count": self.head_param_count,
            "head_names": list(self.head_names),
            "tensor_specs": list(self.tensor_specs),
            "payload_sha256": self.payload_sha256,
            "seq": self.seq,
            "ed25519_signature_hex": self.ed25519_signature_hex,
            "xmss_signature_hex": self.xmss_signature_hex,
        }
        meta_bytes = json.dumps(meta, sort_keys=True, separators=(",", ":")).encode("utf-8")
        raw_wire = HDLT_HEADER_PREFIX.pack(HDLT_MAGIC, len(meta_bytes)) + meta_bytes + self.raw_weight_bytes
        comp = compression.strip().lower()
        if comp in ("none", "raw"):
            return raw_wire
        if comp in ("zlib", "zlib6", "zlib9"):
            lvl = 9 if "9" in comp else 6
            return zlib.compress(raw_wire, level=lvl)
        if comp in ("lzma", "lzma6", "xz"):
            return lzma.compress(raw_wire, preset=6)
        raise ValueError(f"Unsupported HeadDeltaBundle compression: {compression!r}")

    @classmethod
    def from_bytes(cls, data: bytes, *, max_bytes: int = MAX_HDLT_DECOMPRESSED_BYTES) -> HeadDeltaBundle:
        if not isinstance(data, (bytes, bytearray, memoryview)) or not data:
            raise ValueError("Empty HeadDeltaBundle payload")
        max_b = int(max_bytes)
        if max_b <= 0:
            raise ValueError(f"Invalid max_bytes bound: {max_bytes}")
        raw = bytes(data)
        if len(raw) > max_b:
            raise ValueError(f"hdlt_size_limit_exceeded:{len(raw)}>{max_b}")
        if not raw.startswith(HDLT_MAGIC):
            if raw.startswith(b"\xfd7zXZ\x00"):
                try:
                    dec = lzma.LZMADecompressor(memlimit=64 * 1024 * 1024)
                    raw = dec.decompress(raw, max_length=max_b + 1)
                except lzma.LZMAError as exc:
                    raise ValueError(f"hdlt_lzma_decompression_error:{exc}") from exc
                if len(raw) > max_b or not dec.eof:
                    raise ValueError("hdlt_lzma_decompression_limit_exceeded")
            else:
                zdec = zlib.decompressobj()
                raw = zdec.decompress(raw, max_length=max_b + 1)
                if len(raw) > max_b or zdec.unconsumed_tail:
                    raise ValueError("hdlt_zlib_decompression_limit_exceeded")
                rem = max_b + 1 - len(raw)
                raw += zdec.flush(max(1, rem))
                if len(raw) > max_b or zdec.unconsumed_tail:
                    raise ValueError("hdlt_zlib_decompression_limit_exceeded")

        if len(raw) > max_b:
            raise ValueError(f"hdlt_size_limit_exceeded:{len(raw)}>{max_b}")
        if len(raw) < HDLT_HEADER_PREFIX.size:
            raise ValueError("HeadDeltaBundle buffer too short for header prefix")
        magic, meta_len = HDLT_HEADER_PREFIX.unpack(raw[: HDLT_HEADER_PREFIX.size])
        if magic != HDLT_MAGIC:
            raise ValueError(f"Invalid HeadDeltaBundle magic: {magic!r}")
        meta_start = HDLT_HEADER_PREFIX.size
        meta_end = meta_start + meta_len
        if meta_end > len(raw):
            raise ValueError("Truncated HeadDeltaBundle metadata header")
        meta = json.loads(raw[meta_start:meta_end].decode("utf-8"))
        weight_bytes = raw[meta_end:]
        actual_sha = hashlib.sha256(weight_bytes).hexdigest()
        if actual_sha != str(meta.get("payload_sha256", "")).lower():
            raise ValueError("hdlt_payload_sha256_mismatch")

        raw_seq = meta.get("seq", 1)
        if isinstance(raw_seq, bool) or not isinstance(raw_seq, int) or raw_seq < 1:
            raise ValueError(f"invalid_head_delta_seq:{raw_seq!r}")

        return cls(
            schema=str(meta["schema"]),
            version=str(meta["version"]),
            precision=str(meta["precision"]),
            base_trunk_sha256=str(meta["base_trunk_sha256"]).lower(),
            trunk_param_count=int(meta["trunk_param_count"]),
            head_param_count=int(meta["head_param_count"]),
            head_names=tuple(meta["head_names"]),
            tensor_specs=tuple(meta["tensor_specs"]),
            raw_weight_bytes=weight_bytes,
            payload_sha256=actual_sha,
            seq=int(raw_seq),
            ed25519_signature_hex=str(meta.get("ed25519_signature_hex", "")),
            xmss_signature_hex=str(meta.get("xmss_signature_hex", "")),
        )


def extract_head_delta_bundle(
    candidate_source: Any,
    *,
    precision: str = "fp16",
    heads: Sequence[str] | None = None,
    base_trunk_sha256: str | None = None,
    version: str = "3.0.0",
    seq: int = 1,
    ed25519_privkey_bytes: bytes | None = None,
    xmss_tree: Any | None = None,
) -> HeadDeltaBundle:
    """Extract, quantize (`fp16` / `int8` / `fp32`), and optionally sign the 9-head delta bundle."""
    if isinstance(seq, bool) or not isinstance(seq, int) or seq < 1:
        raise ValueError(f"invalid_head_delta_seq:{seq!r}")
    prec = precision.strip().lower()
    if prec not in ("fp32", "fp16", "int8"):
        raise ValueError(f"Unsupported head delta precision: {precision!r} (expected fp32, fp16, or int8)")

    trunk_tensors, head_tensors = load_state_tensors(candidate_source)
    computed_trunk_sha = compute_trunk_sha256(trunk_tensors)
    if base_trunk_sha256 is not None and base_trunk_sha256.lower() != computed_trunk_sha.lower():
        raise ValueError(
            f"Candidate model trunk SHA-256 ({computed_trunk_sha}) does not match pinned base_trunk_sha256 ({base_trunk_sha256})"
        )

    selected_heads = tuple(heads) if heads is not None else ALL_9_HEAD_NAMES
    filtered_keys: list[str] = []
    for k in sorted(head_tensors.keys()):
        if any(k.startswith(f"heads.{h}.") or k == f"heads.{h}" for h in selected_heads):
            filtered_keys.append(k)

    if not filtered_keys:
        raise ValueError(f"No head tensors matched requested heads: {selected_heads}")

    trunk_params = sum(int(arr.size) for arr in trunk_tensors.values())
    head_params = 0
    tensor_specs: list[dict[str, Any]] = []
    weight_chunks: list[bytes] = []
    offset = 0

    for k in filtered_keys:
        arr_f32 = _to_numpy_fp32(head_tensors[k])
        n_elem = int(arr_f32.size)
        head_params += n_elem

        if prec == "fp32":
            q_bytes = arr_f32.astype("<f4", copy=False).tobytes()
            scale = 1.0
        elif prec == "fp16":
            q_bytes = arr_f32.astype("<f2").tobytes()
            scale = 1.0
        else:  # int8 symmetric absmax quantization
            max_abs = float(np.max(np.abs(arr_f32))) if n_elem > 0 else 0.0
            scale = (max_abs / 127.0) if max_abs > 1e-12 else 1.0
            q_int8 = np.clip(np.round(arr_f32 / scale), -127, 127).astype(np.int8)
            q_bytes = q_int8.tobytes()

        tensor_specs.append(
            {
                "name": k,
                "shape": list(arr_f32.shape),
                "numel": n_elem,
                "offset": offset,
                "nbytes": len(q_bytes),
                "scale": scale,
            }
        )
        weight_chunks.append(q_bytes)
        offset += len(q_bytes)

    raw_weight_bytes = b"".join(weight_chunks)
    payload_sha256 = hashlib.sha256(raw_weight_bytes).hexdigest()

    unsigned_bundle = HeadDeltaBundle(
        schema="armiger.dht.head-delta/v3",
        version=version,
        precision=prec,
        base_trunk_sha256=computed_trunk_sha,
        trunk_param_count=trunk_params,
        head_param_count=head_params,
        head_names=selected_heads,
        tensor_specs=tuple(tensor_specs),
        raw_weight_bytes=raw_weight_bytes,
        payload_sha256=payload_sha256,
        seq=int(seq),
    )

    sig_payload = unsigned_bundle.signing_payload()
    ed_sig_hex = ""
    if ed25519_privkey_bytes is not None:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        sk = (
            ed25519_privkey_bytes
            if isinstance(ed25519_privkey_bytes, Ed25519PrivateKey)
            else Ed25519PrivateKey.from_private_bytes(bytes(ed25519_privkey_bytes))
        )
        ed_sig_hex = sk.sign(sig_payload).hex()

    xmss_sig_hex = ""
    if xmss_tree is not None:
        xmss_sig = xmss_tree.sign_seq(int(seq), sig_payload)
        xmss_sig_hex = xmss_sig.to_bytes().hex()

    return dataclasses.replace(
        unsigned_bundle,
        ed25519_signature_hex=ed_sig_hex,
        xmss_signature_hex=xmss_sig_hex,
    )


def validate_head_delta_tensor_specs(
    bundle: HeadDeltaBundle,
    *,
    expected_head_shapes: Mapping[str, tuple[int, ...]] | None = None,
    forbidden_trunk_keys: set[str] | None = None,
) -> None:
    """Validate `HeadDeltaBundle.tensor_specs` against permitted head names, expected shapes,
    scales, contiguous byte ranges, and prohibited frozen-trunk keys before weight reconstruction.
    """
    if not bundle.head_names or any(h not in ALL_9_HEAD_NAMES for h in bundle.head_names):
        raise ValueError(f"invalid_head_delta_head_names:{bundle.head_names!r}")
    if not bundle.tensor_specs:
        raise ValueError("empty_head_delta_tensor_specs")

    prec = bundle.precision.strip().lower()
    if prec == "fp32":
        bytes_per_elem = 4
    elif prec == "fp16":
        bytes_per_elem = 2
    elif prec == "int8":
        bytes_per_elem = 1
    else:
        raise ValueError(f"Unsupported precision in bundle: {bundle.precision!r}")

    total_bytes = len(bundle.raw_weight_bytes)
    expected_offset = 0
    total_numel = 0
    seen_names: set[str] = set()
    trunk_deny = set(forbidden_trunk_keys or ())

    for spec in bundle.tensor_specs:
        if not isinstance(spec, Mapping):
            raise ValueError("invalid_tensor_spec_entry")
        name = str(spec.get("name", ""))
        if not name or name in seen_names:
            raise ValueError(f"duplicate_or_empty_tensor_spec_name:{name!r}")
        seen_names.add(name)

        if name in trunk_deny or f"m.{name}" in trunk_deny or name.removeprefix("m.") in trunk_deny:
            raise ValueError(f"head_delta_prohibited_trunk_key_write:{name}")
        if not name.startswith("heads.") or name != canonical_head_key(name) or not is_head_tensor_key(name):
            raise ValueError(f"invalid_head_tensor_name:{name}")
        if not any(name.startswith(f"heads.{h}.") or name == f"heads.{h}" for h in bundle.head_names):
            raise ValueError(f"unpermitted_head_tensor_name:{name}")

        raw_shape = spec.get("shape")
        if not isinstance(raw_shape, (list, tuple)) or not raw_shape:
            raise ValueError(f"invalid_tensor_shape:{name}")
        shape = tuple(int(d) for d in raw_shape)
        if any(d <= 0 for d in shape):
            raise ValueError(f"invalid_tensor_shape_dims:{name}:{shape}")
        computed_numel = int(math.prod(shape))
        numel = int(spec.get("numel", -1))
        if numel <= 0 or numel != computed_numel:
            raise ValueError(f"tensor_numel_mismatch:{name}:{numel}!={computed_numel}")

        if expected_head_shapes is not None:
            if name not in expected_head_shapes:
                raise ValueError(f"unexpected_head_tensor_key:{name}")
            if shape != tuple(expected_head_shapes[name]):
                raise ValueError(f"head_tensor_shape_mismatch:{name}:{shape}!={tuple(expected_head_shapes[name])}")

        raw_scale = spec.get("scale", 1.0)
        if isinstance(raw_scale, bool) or not isinstance(raw_scale, (int, float)):
            raise ValueError(f"invalid_tensor_scale_type:{name}")
        scale = float(raw_scale)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"invalid_tensor_scale_value:{name}:{scale}")
        if prec in ("fp32", "fp16") and abs(scale - 1.0) > 1e-9:
            raise ValueError(f"unexpected_non_unit_scale_for_{prec}:{name}:{scale}")

        off = int(spec.get("offset", -1))
        nbytes = int(spec.get("nbytes", -1))
        expected_nbytes = computed_numel * bytes_per_elem
        if off != expected_offset or nbytes != expected_nbytes or off < 0 or nbytes <= 0 or (off + nbytes) > total_bytes:
            raise ValueError(
                f"invalid_tensor_byte_range:{name}:offset={off}(expected={expected_offset}),nbytes={nbytes}(expected={expected_nbytes})"
            )

        expected_offset += nbytes
        total_numel += computed_numel

    if expected_offset != total_bytes:
        raise ValueError(f"tensor_specs_byte_coverage_mismatch:{expected_offset}!={total_bytes}")
    if total_numel != int(bundle.head_param_count):
        raise ValueError(f"head_param_count_mismatch:{total_numel}!={bundle.head_param_count}")


def dequantize_head_delta_tensors(
    bundle: HeadDeltaBundle,
    *,
    expected_head_shapes: Mapping[str, tuple[int, ...]] | None = None,
    forbidden_trunk_keys: set[str] | None = None,
) -> dict[str, np.ndarray]:
    """Dequantize a `HeadDeltaBundle` back into canonical `heads.*` `float32` NumPy arrays."""
    actual_sha = hashlib.sha256(bundle.raw_weight_bytes).hexdigest()
    if actual_sha != bundle.payload_sha256.lower():
        raise ValueError("hdlt_payload_sha256_mismatch")

    validate_head_delta_tensor_specs(
        bundle,
        expected_head_shapes=expected_head_shapes,
        forbidden_trunk_keys=forbidden_trunk_keys,
    )

    prec = bundle.precision.lower()
    raw = bundle.raw_weight_bytes
    out: dict[str, np.ndarray] = {}

    for spec in bundle.tensor_specs:
        name = str(spec["name"])
        shape = tuple(int(d) for d in spec["shape"])
        off = int(spec["offset"])
        nbytes = int(spec["nbytes"])
        scale = float(spec.get("scale", 1.0))
        chunk = raw[off : off + nbytes]
        if len(chunk) != nbytes:
            raise ValueError(f"Truncated tensor slice for {name}")

        if prec == "fp32":
            arr = np.frombuffer(chunk, dtype="<f4").astype(np.float32, copy=True).reshape(shape)
        elif prec == "fp16":
            arr = np.frombuffer(chunk, dtype="<f2").astype(np.float32).reshape(shape)
        elif prec == "int8":
            arr = (np.frombuffer(chunk, dtype=np.int8).astype(np.float32) * scale).reshape(shape)
        else:
            raise ValueError(f"Unsupported precision in bundle: {prec!r}")
        out[name] = np.ascontiguousarray(arr, dtype=np.float32)

    return out


def reconstruct_from_head_delta(
    base_trunk_or_model: Any,
    delta_bundle_or_bytes: HeadDeltaBundle | bytes,
    *,
    expected_trunk_sha256: str | None = None,
    ed25519_pubkey_bytes: bytes | None = None,
    xmss_root_hex: str | None = None,
    last_verified_seq: int = 0,
) -> dict[str, Any]:
    """Verify pinned frozen-trunk SHA-256 (and optional Ed25519/XMSS signatures) and reconstruct full model state."""
    bundle = (
        delta_bundle_or_bytes
        if isinstance(delta_bundle_or_bytes, HeadDeltaBundle)
        else HeadDeltaBundle.from_bytes(delta_bundle_or_bytes)
    )

    if isinstance(bundle.seq, bool) or not isinstance(bundle.seq, int) or bundle.seq < 1:
        raise ValueError(f"invalid_head_delta_seq:{bundle.seq!r}")
    if isinstance(last_verified_seq, bool) or not isinstance(last_verified_seq, int) or last_verified_seq < 0:
        raise ValueError(f"invalid_last_verified_seq:{last_verified_seq!r}")
    if bundle.seq <= last_verified_seq:
        raise ValueError(f"stale_or_replayed_head_delta_seq:{bundle.seq}<={last_verified_seq}")

    sig_payload = bundle.signing_payload()
    if ed25519_pubkey_bytes is not None:
        if not bundle.ed25519_signature_hex:
            raise ValueError("missing_ed25519_signature_on_head_delta")
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

        pub = (
            ed25519_pubkey_bytes
            if isinstance(ed25519_pubkey_bytes, Ed25519PublicKey)
            else Ed25519PublicKey.from_public_bytes(bytes(ed25519_pubkey_bytes))
        )
        try:
            pub.verify(bytes.fromhex(bundle.ed25519_signature_hex), sig_payload)
        except (InvalidSignature, ValueError) as exc:
            raise ValueError("invalid_ed25519_signature_on_head_delta") from exc

    if xmss_root_hex is not None:
        if not bundle.xmss_signature_hex:
            raise ValueError("missing_xmss_signature_on_head_delta")
        from armiger.pq_xmss_merkle import XMSSSignature, verify_xmss_signature

        try:
            xsig = XMSSSignature.from_bytes(bytes.fromhex(bundle.xmss_signature_hex))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"invalid_xmss_signature_on_head_delta:{exc}") from exc
        if xsig.seq != bundle.seq:
            raise ValueError("xmss_seq_mismatch_with_bundle_seq")
        ok_pq, reason_pq = verify_xmss_signature(
            merkle_root_hex=xmss_root_hex,
            message=sig_payload,
            xmss_sig=xsig,
            last_verified_seq=last_verified_seq,
        )
        if not ok_pq:
            raise ValueError(f"invalid_xmss_signature_on_head_delta:{reason_pq}")

    trunk_tensors, existing_heads = load_state_tensors(base_trunk_or_model)
    actual_trunk_sha = compute_trunk_sha256(trunk_tensors)
    if actual_trunk_sha != bundle.base_trunk_sha256.lower():
        raise ValueError(
            f"base_trunk_sha256_mismatch: bundle requires {bundle.base_trunk_sha256}, got {actual_trunk_sha}"
        )
    if expected_trunk_sha256 is not None and actual_trunk_sha != expected_trunk_sha256.lower():
        raise ValueError(
            f"expected_trunk_sha256_mismatch: expected {expected_trunk_sha256}, got {actual_trunk_sha}"
        )

    forbidden_trunk_keys = set(trunk_tensors.keys()) | {k.removeprefix("m.") for k in trunk_tensors.keys()}
    expected_shapes = (
        {k: tuple(int(d) for d in v.shape) for k, v in existing_heads.items()}
        if existing_heads
        else None
    )
    deq_heads = dequantize_head_delta_tensors(
        bundle,
        expected_head_shapes=expected_shapes,
        forbidden_trunk_keys=forbidden_trunk_keys,
    )
    merged: dict[str, Any] = {}
    for k, v in trunk_tensors.items():
        merged[k] = v
    for k, v in existing_heads.items():
        merged[k] = v
    for k, v in deq_heads.items():
        if k in forbidden_trunk_keys or f"m.{k}" in forbidden_trunk_keys:
            raise ValueError(f"head_delta_prohibited_trunk_key_write:{k}")
        merged[k] = v

    # If the caller supplied an ONNX ModelProto, also return a patched ModelProto copy
    if hasattr(base_trunk_or_model, "graph") and hasattr(base_trunk_or_model.graph, "initializer"):
        from onnx import numpy_helper

        patched_onnx = copy.deepcopy(base_trunk_or_model)
        _, _, onnx_map = _extract_onnx_tensors(patched_onnx)
        init_lookup = {init.name: idx for idx, init in enumerate(patched_onnx.graph.initializer)}
        for canon_name, arr_f32 in deq_heads.items():
            if canon_name in onnx_map:
                onnx_init_name, needs_transpose = onnx_map[canon_name]
                target_arr = np.ascontiguousarray(arr_f32.T if needs_transpose else arr_f32, dtype=np.float32)
                idx = init_lookup[onnx_init_name]
                patched_onnx.graph.initializer[idx].CopyFrom(
                    numpy_helper.from_array(target_arr, name=onnx_init_name)
                )
        return {
            "state_dict": merged,
            "onnx_model": patched_onnx,
            "base_trunk_sha256": actual_trunk_sha,
            "reconstructed_head_tensors": len(deq_heads),
            "precision": bundle.precision,
        }

    return {
        "state_dict": merged,
        "base_trunk_sha256": actual_trunk_sha,
        "reconstructed_head_tensors": len(deq_heads),
        "precision": bundle.precision,
    }
