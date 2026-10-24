"""Armiger BEP-44 / BEP-46 Distributed Recursive Self-Improvement (RSI) & Distributed Data Plugin.

Standalone decentralized RSI engine for Armiger and Armiger Ultra.
Supports:
  1. Universal Adapters: Series, Movie, Audio, and Publication webhooks, ffprobe/MediaInfo stream
     verification, and qBittorrent/Transmission completion telemetry (client-agnostic across media automation stacks).
  2. Foolproof Physical Ground-Truth Verification: Validates physical download completion, stream
     decodability, and canonical catalog matches. Rejects raw unverified model predictions.
  3. Complete Anonymity & Privacy Engine: PII, IP, and private tracker passkey stripping, salted
     blinded title stems, differential privacy (Laplace perturbation + logarithmic size binning),
     and Poisson timing jitter against timing correlation attacks.
  4. Post-Quantum End-to-End Encryption: Hybrid X25519 + NIST FIPS 203 ML-KEM-768 key encapsulation
     combined via SHAKE-256 / HKDF-SHA3-256 with 256-bit ChaCha20-Poly1305 AEAD authentication.
  5. Post-Quantum Spoof-Proof Protocol: BEP-44 strictly monotonic sequence numbers + Hybrid Ed25519
     and WOTS+-SHAKE256 (RFC 8391 / NIST FIPS 205 hash-based PQ) threshold multi-signatures ($M$-of-$N$ quorum).
  6. Poison-Proof Guards: $K$-of-$N$ distinct-ASN Sybil quorum verification, strict format
     whitelisting (SafeTensors, ONNX, JSON only; rejects arbitrary pickle/executable bytecode), and
     a mandatory 500-item golden canary regression gate verifying Mondrian conformal safety bounds.
  7. Opt-In / Opt-Out Modes: 'off', 'pull_only', and 'anonymous_contribute'.
"""
from __future__ import annotations

import dataclasses
import enum
import functools
import hashlib
import hmac
import ipaddress
import json
import math
import os
import pathlib
import random
import secrets
import re
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

from cryptography.exceptions import InvalidSignature, InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ed25519, x25519
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF



import sys

from .dht_rsi_models_and_privacy import (
    AcyclicReleaseEnvelope,
    AnonymityGuard,
    AnonymityViolationError,
    AnonymizedTelemetryRecord,
    AuthenticatedConformalBaseline,
    BenchmarkAttestationPredicate,
    BlindedTitleStem,
    BundleFormat,
    CanaryItem,
    CanaryRegressionReport,
    CandidateObservation,
    CatalogConsensusResult,
    ContinuousAggregationResult,
    ContinuousDeltaSubmission,
    DHTRSIMode,
    DSSE_TYPE_ATTESTATION_V2,
    DSSE_TYPE_BASELINE_V1,
    DSSE_TYPE_RELEASE_V2,
    EncryptedTelemetryBundle,
    MediaCategory,
    PeerAttestation,
    PhysicalVerificationResult,
    QueuedTelemetryItem,
    TelemetryAnonymizer,
    TelemetrySubmission,
    ThresholdDHTManifest,
    VerifiedObservation,
    ZeroCueCatalogDelta,
    _bep44_canonical_sign_buffer,
    dsse_pae,
    pure_laplace_noise,
    resolve_opt_in_sharing,
)
from .dht_rsi_crypto_and_adapters import (
    AudioWebhookAdapter,
    HybridPQCAEADEngine,
    MLKEM768,
    MediaStreamVerifier,
    MovieWebhookAdapter,
    PublicationWebhookAdapter,
    QBittorrentAdapter,
    SeriesWebhookAdapter,
    TransmissionAdapter,
    WOTSPlusSHAKE256,
    X25519ChaCha20Poly1305Engine,
    _OptInAdapterBase,
)
from .dht_rsi_verification import (
    AcyclicProvenanceVerifier,
    AuthoritativeASNVerifier,
    PhysicalGroundTruthVerifier,
    SafeFormatValidator,
    SubnetDiversityLimiter,
    SybilQuorumVerifier,
    ThresholdSignatureVerifier,
)
from .dht_rsi_canary_and_agg import (
    ByzantineResilientAggregator,
    GoldenCanaryRegressionGate,
    generate_golden_canary_suite,
)

# =====================================================================
# 11. Top-Level Armiger DHT RSI Engine
# =====================================================================

class ArmigerDHTRSIEngine:
    """Master engine integrating universal ingestion, physical ground truth verification,

    complete privacy/anonymity, AEAD encryption, spoof-proof threshold signatures,
    distinct-ASN Sybil quorums, safe formats, and the 500-item golden canary regression gate.
    """

    def __init__(
        self,
        mode: Optional[str] = None,
        current_seq: int = 0,
        trusted_signers: Optional[Dict[str, str]] = None,
        threshold_m: int = 1,
        min_distinct_asns: int = 5,
        blinding_salt: Optional[bytes] = None,
        dp_epsilon: float = 1.0,
        poisson_mean_seconds: float = 1800.0,
        aggregator_x25519_pubkey_hex: Optional[str] = None,
        trusted_pq_signers: Optional[Dict[str, str]] = None,
        *,
        require_pq: bool = True,
        opt_in_sharing: Optional[bool] = None,
        ephemeral_identity: bool = True,
        min_k_samples: int = 25,
        clip_norm: float = 1.0,
        dp_sigma: float = 0.05,
        max_seen_delta_digests: int = 4096,
        max_tracked_senders: int = 1024,
        delta_freshness_window_seconds: int = 14 * 86400,
        max_telemetry_queue_size: int = 1024,
        max_item_encryption_retries: int = 3,
        trusted_evaluator_signers: Optional[Dict[str, str]] = None,
        require_acyclic_envelope: bool = True,
    ) -> None:
        self.opt_in_sharing: bool = resolve_opt_in_sharing(opt_in_sharing)
        if mode is None:
            resolved_mode = DHTRSIMode.ANONYMOUS_CONTRIBUTE.value if self.opt_in_sharing else DHTRSIMode.PULL_ONLY.value
        else:
            resolved_mode = mode.value if isinstance(mode, DHTRSIMode) else str(mode)

        if resolved_mode not in [m.value for m in DHTRSIMode]:
            raise ValueError(f"Unsupported mode: {resolved_mode}")

        self.mode = DHTRSIMode(resolved_mode)
        self.current_seq = int(current_seq)
        self.current_bundle_sha256: Optional[str] = None
        self._telemetry_seq = int(current_seq)
        self.trusted_signers = trusted_signers or {}
        self.trusted_pq_signers = trusted_pq_signers or {}
        self.trusted_evaluator_signers = dict(trusted_evaluator_signers or {})
        self.require_acyclic_envelope = bool(require_acyclic_envelope)
        self.require_pq = bool(require_pq)
        self.threshold_m = int(threshold_m)
        self.min_distinct_asns = int(min_distinct_asns)
        if aggregator_x25519_pubkey_hex is not None and str(aggregator_x25519_pubkey_hex).strip() != "":
            self._validate_aggregator_pubkey_hex(str(aggregator_x25519_pubkey_hex).strip())
            self.aggregator_x25519_pubkey_hex: Optional[str] = str(aggregator_x25519_pubkey_hex).strip()
        else:
            self.aggregator_x25519_pubkey_hex = aggregator_x25519_pubkey_hex
        self.max_seen_delta_digests = max(int(max_seen_delta_digests), 1)
        self.max_tracked_senders = max(int(max_tracked_senders), 1)
        self.delta_freshness_window_seconds = max(int(delta_freshness_window_seconds), 60)
        self.max_telemetry_queue_size = max(int(max_telemetry_queue_size), 1)
        self.max_item_encryption_retries = max(int(max_item_encryption_retries), 1)
        self._seen_delta_digests: Dict[str, Tuple[int, Tuple[str, bytes]]] = {}
        self._inbound_sender_seq: Dict[Tuple[str, bytes], Tuple[int, int]] = {}
        self._evicted_sender_seq_floor: int = int(current_seq)

        self.verifier = PhysicalGroundTruthVerifier()
        self.anonymizer = TelemetryAnonymizer(
            blinding_salt=blinding_salt,
            dp_epsilon=dp_epsilon,
            poisson_mean_seconds=poisson_mean_seconds,
        )
        self.anonymity_guard = AnonymityGuard(
            ephemeral_identity=ephemeral_identity,
            min_k_samples=min_k_samples,
            clip_norm=clip_norm,
            dp_sigma=dp_sigma,
        )
        self.canary_gate = GoldenCanaryRegressionGate()
        self.telemetry_queue: List[QueuedTelemetryItem] = []
        self._outbound_bundles_shared: int = 0
        self._outbound_packets_blocked_opt_out: int = 0
        self._outbound_bytes_shared: int = 0
        self._telemetry_encryption_errors: int = 0
        self._telemetry_dropped_backlog: int = 0
        self._last_telemetry_error: str = ""

    @staticmethod
    def _validate_aggregator_pubkey_hex(pubkey_hex: str) -> bytes:
        try:
            raw_bytes = bytes.fromhex(str(pubkey_hex).strip())
        except Exception as exc:
            raise ValueError(f"Invalid aggregator_x25519_pubkey_hex (non-hex): {exc}") from exc
        if len(raw_bytes) not in (32, X25519ChaCha20Poly1305Engine.HYBRID_PUB_LEN):
            raise ValueError(
                f"Invalid aggregator_x25519_pubkey_hex length: {len(raw_bytes)} bytes "
                f"(expected 32 or {X25519ChaCha20Poly1305Engine.HYBRID_PUB_LEN})"
            )
        return raw_bytes

    # -----------------------------------------------------------------
    # Ingestion & Physical Ground-Truth
    # -----------------------------------------------------------------

    def ingest_candidate(
        self,
        candidate: CandidateObservation,
        ffprobe_output: Optional[Dict[str, Any]] = None,
        current_timestamp: Optional[float] = None,
    ) -> Tuple[bool, str, Optional[AnonymizedTelemetryRecord]]:
        """Ingest candidate from any adapter, verify physical ground truth, anonymize, and queue."""
        if self.mode == DHTRSIMode.OFF:
            return False, "plugin_mode_off", None

        if self.mode == DHTRSIMode.PULL_ONLY:
            self._outbound_packets_blocked_opt_out += 1
            return False, "telemetry_disabled_in_pull_only_mode", None

        # Verify physical ground truth (never unverified predictions)
        res = self.verifier.verify(candidate, ffprobe_output)
        if not res.is_verified or res.observation is None:
            return False, f"physical_verification_failed:{res.rejection_reason}", None

        # Anonymize & apply differential privacy
        record = self.anonymizer.anonymize(res.observation)

        # Apply Poisson timing jitter
        now = current_timestamp if current_timestamp is not None else time.time()
        jitter_delay = self.anonymizer.calculate_poisson_jitter()
        scheduled_time = now + jitter_delay

        if not self.opt_in_sharing:
            self._outbound_packets_blocked_opt_out += 1
            return True, "verified_anonymized_and_queued", record

        if len(self.telemetry_queue) >= self.max_telemetry_queue_size:
            self.telemetry_queue.pop(0)
            self._telemetry_dropped_backlog += 1

        self.telemetry_queue.append(QueuedTelemetryItem(
            scheduled_dispatch_time=scheduled_time,
            record=record,
        ))

        return True, "verified_anonymized_and_queued", record

    def sharing_status(self) -> Dict[str, Any]:
        """Return runtime audit telemetry confirming opt-in status and outbound packet/bundle counters."""
        return {
            "opt_in_sharing_enabled": bool(self.opt_in_sharing),
            "mode": self.mode.value,
            "outbound_bundles_shared": int(self._outbound_bundles_shared),
            "outbound_packets_blocked_opt_out": int(self._outbound_packets_blocked_opt_out),
            "outbound_bytes_shared": int(self._outbound_bytes_shared),
            "ephemeral_identity_enabled": bool(self.anonymity_guard.ephemeral_identity),
            "ephemeral_pubkey_hex": self.anonymity_guard.current_pubkey_hex() if self.opt_in_sharing else None,
            "ephemeral_rotations": int(self.anonymity_guard.rotation_count),
            "blocked_anonymity_violations": int(self.anonymity_guard.blocked_violations_count),
            "telemetry_encryption_errors": int(self._telemetry_encryption_errors),
            "telemetry_dropped_backlog": int(self._telemetry_dropped_backlog),
            "last_telemetry_error": str(self._last_telemetry_error),
        }

    def export_signed_delta_bundle(
        self,
        delta_w: Optional[Sequence[float]] = None,
        delta_b: Optional[Sequence[float]] = None,
        *,
        reliability_bins: Optional[Sequence[Any]] = None,
        n_samples: int = 25,
        epoch: Optional[Union[int, float]] = None,
        seq: Optional[int] = None,
        model_tier: str = "ultra",
        salt: bytes = b"armiger-rsi-delta",
        extra_fields: Optional[Mapping[str, Any]] = None,
        raw_wire_override: Optional[Union[bytes, str, Mapping[str, Any]]] = None,
        rotate_key: Optional[bool] = None,
    ) -> Optional[Dict[str, Any]]:
        """Export an L2-clipped, Gaussian-DP, int8-quantized, coarse-timestamped BEP-44 delta bundle.

        Strictly returns None (0 bytes shared) when `opt_in_sharing=False`.
        Raises `AnonymityViolationError` if any PII/canary/schema/k-anonymity invariant is violated.
        """
        if not self.opt_in_sharing or self.mode != DHTRSIMode.ANONYMOUS_CONTRIBUTE:
            self._outbound_packets_blocked_opt_out += 1
            return None

        next_seq = int(seq) if seq is not None else (getattr(self, "_outbound_seq", self.current_seq) + 1)

        if raw_wire_override is not None:
            if isinstance(raw_wire_override, (bytes, bytearray)):
                v_bytes = bytes(raw_wire_override)
            elif isinstance(raw_wire_override, str):
                v_bytes = raw_wire_override.encode("utf-8")
            elif isinstance(raw_wire_override, Mapping):
                v_bytes = json.dumps(dict(raw_wire_override), sort_keys=True, separators=(",", ":")).encode("utf-8")
            else:
                raise AnonymityViolationError("invalid_raw_wire_override")
            signed = self.anonymity_guard.sign_wire_payload(
                v_bytes, seq=next_seq, salt=salt, rotate_key=rotate_key
            )
            self._outbound_seq = max(getattr(self, "_outbound_seq", 0), next_seq)
            self._outbound_bundles_shared += 1
            self._outbound_bytes_shared += len(v_bytes)
            return signed

        w_in = list(delta_w) if delta_w is not None else [0.0] * 8
        b_in = list(delta_b) if delta_b is not None else [0.0] * 2
        dp_res = self.anonymity_guard.clip_and_privatize_weights(
            w_in,
            b_in,
            n_samples=n_samples,
        )
        epoch_bucket = self.anonymity_guard.bucket_timestamp(epoch)

        wire_obj: Dict[str, Any] = {
            "v": 2,
            "epoch": epoch_bucket,
            "seq": next_seq,
            "model_tier": model_tier,
            "delta_w_q8": dp_res["delta_w_q8"],
            "delta_b_q8": dp_res["delta_b_q8"],
            "scale": dp_res["scale"],
            "n_bucket": dp_res["n_bucket"],
        }
        if reliability_bins is not None:
            wire_obj["reliability_bins"] = self.anonymity_guard.sanitize_reliability_bins(reliability_bins)
        if extra_fields:
            for k, val in extra_fields.items():
                wire_obj[k] = val

        v_bytes = json.dumps(wire_obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
        signed = self.anonymity_guard.sign_wire_payload(
            v_bytes, seq=next_seq, salt=salt, rotate_key=rotate_key
        )
        signed["pre_dp_l2_norm"] = dp_res["pre_dp_l2_norm"]
        signed["clipped_l2_norm"] = dp_res["clipped_l2_norm"]
        signed["delta_w_dp"] = dp_res["delta_w_dp"]
        signed["delta_b_dp"] = dp_res["delta_b_dp"]
        self._outbound_seq = max(getattr(self, "_outbound_seq", 0), next_seq)
        self._outbound_bundles_shared += 1
        self._outbound_bytes_shared += len(v_bytes)
        return signed

    def export_signed_bundle(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Optional[Dict[str, Any]]:
        """Alias for `export_signed_delta_bundle` enforcing opt_in_sharing and AnonymityGuard."""
        return self.export_signed_delta_bundle(*args, **kwargs)

    def _prune_inbound_delta_tracking(self, now_ts: int) -> None:
        cutoff = now_ts - self.delta_freshness_window_seconds
        expired_digests = [
            d for d, (seen_ts, _) in self._seen_delta_digests.items() if seen_ts < cutoff
        ]
        for d in expired_digests:
            self._seen_delta_digests.pop(d, None)

        expired_senders = [
            sk for sk, (_, seen_ts) in self._inbound_sender_seq.items() if seen_ts < cutoff
        ]
        for sk in expired_senders:
            self._inbound_sender_seq.pop(sk, None)

    def ingest_signed_delta_bundle(
        self,
        signed_bundle: Mapping[str, Any],
    ) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """Verify Ed25519 signature and AnonymityGuard schema of an inbound delta bundle (works in pull_only mode)."""
        if self.mode == DHTRSIMode.OFF:
            return False, "plugin_mode_off", None

        try:
            v_raw = signed_bundle["v"]
            v_bytes = v_raw if isinstance(v_raw, (bytes, bytearray)) else str(v_raw).encode("utf-8")
            seq = int(signed_bundle["seq"])
            salt = signed_bundle.get("salt", b"armiger-rsi-delta")
            if isinstance(salt, str):
                salt = salt.encode("utf-8")
            salt_bytes = bytes(salt)
            if len(salt_bytes) > 64:
                return False, f"bep44_salt_exceeds_64_bytes:{len(salt_bytes)}", None
            pub_raw = signed_bundle.get("k") or signed_bundle.get("pubkey_hex")
            pub_bytes = bytes.fromhex(pub_raw) if isinstance(pub_raw, str) else bytes(pub_raw)
            sig_raw = signed_bundle.get("sig") or signed_bundle.get("sig_hex")
            sig_bytes = bytes.fromhex(sig_raw) if isinstance(sig_raw, str) else bytes(sig_raw)
        except Exception as exc:
            return False, f"malformed_bundle_envelope:{exc}", None

        try:
            vk = ed25519.Ed25519PublicKey.from_public_bytes(pub_bytes)
            vk.verify(sig_bytes, _bep44_canonical_sign_buffer(v=bytes(v_bytes), seq=seq, salt=salt_bytes))
        except Exception as exc:
            return False, f"invalid_ed25519_signature:{exc}", None

        now_ts = int(time.time())
        self._prune_inbound_delta_tracking(now_ts)

        sender_key = (pub_bytes.hex(), salt_bytes)
        prev_entry = self._inbound_sender_seq.get(sender_key)
        min_seq = max(
            self.current_seq,
            prev_entry[0] if prev_entry is not None else self._evicted_sender_seq_floor,
        )
        if seq <= min_seq:
            return False, f"stale_or_replayed_delta_seq:{seq}<={min_seq}", None

        bundle_digest = hashlib.sha256(bytes(v_bytes) + bytes(sig_bytes)).hexdigest()
        if bundle_digest in self._seen_delta_digests:
            return False, "replayed_delta_bundle_duplicate", None

        try:
            parsed = self.anonymity_guard.inspect_wire_payload(bytes(v_bytes))
        except AnonymityViolationError as exc:
            return False, f"anonymity_firewall_rejected:{exc}", None

        epoch_val = int(parsed.get("epoch") or 0)
        if epoch_val <= 0 or abs(now_ts - epoch_val) > self.delta_freshness_window_seconds:
            return False, f"stale_or_future_delta_epoch:{epoch_val}", None

        # Any prior digest for this same sender_key is now superseded by the higher monotonic seq
        superseded_digests = [
            d for d, (_, sk) in self._seen_delta_digests.items() if sk == sender_key
        ]
        for d in superseded_digests:
            self._seen_delta_digests.pop(d, None)

        while len(self._seen_delta_digests) >= self.max_seen_delta_digests:
            oldest_d = next(iter(self._seen_delta_digests))
            self._seen_delta_digests.pop(oldest_d, None)
        self._seen_delta_digests[bundle_digest] = (now_ts, sender_key)

        if sender_key in self._inbound_sender_seq:
            self._inbound_sender_seq.pop(sender_key, None)
        while len(self._inbound_sender_seq) >= self.max_tracked_senders:
            oldest_sk = next(iter(self._inbound_sender_seq))
            evicted_seq, _ = self._inbound_sender_seq.pop(oldest_sk)
            self._evicted_sender_seq_floor = max(self._evicted_sender_seq_floor, evicted_seq)
        self._inbound_sender_seq[sender_key] = (seq, now_ts)
        scale = float(parsed.get("scale", 1.0 / 127.0))
        w_q8 = parsed.get("delta_w_q8", [])
        b_q8 = parsed.get("delta_b_q8", [])
        decoded = {
            "epoch": parsed.get("epoch"),
            "seq": seq,
            "model_tier": parsed.get("model_tier", "ultra"),
            "n_bucket": parsed.get("n_bucket", 25),
            "delta_w": [float(q) * scale for q in w_q8],
            "delta_b": [float(q) * scale for q in b_q8],
            "reliability_bins": parsed.get("reliability_bins", []),
            "sender_ephemeral_pubkey_hex": pub_bytes.hex(),
        }
        return True, "verified_and_ingested", decoded

    def ingest_signed_bundle(
        self,
        signed_bundle: Mapping[str, Any],
    ) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """Alias for `ingest_signed_delta_bundle`."""
        return self.ingest_signed_delta_bundle(signed_bundle)

    async def publish_to_dht(
        self,
        swarm: Any,
        bundle: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> Tuple[bool, str, int]:
        """Publish an anonymous signed bundle to a UDP BEP-44 swarm. Blocked when `opt_in_sharing=False`."""
        if not self.opt_in_sharing or self.mode != DHTRSIMode.ANONYMOUS_CONTRIBUTE:
            self._outbound_packets_blocked_opt_out += 1
            return False, "opt_in_sharing_disabled", 0

        eff_bundle = dict(bundle) if bundle is not None else self.export_signed_delta_bundle(**kwargs)
        if eff_bundle is None:
            self._outbound_packets_blocked_opt_out += 1
            return False, "opt_in_sharing_disabled", 0

        v_bytes = eff_bundle["v"] if isinstance(eff_bundle["v"], bytes) else str(eff_bundle["v"]).encode("utf-8")
        # Enforce wire firewall right before UDP transmission
        self.anonymity_guard.inspect_wire_payload(v_bytes)

        priv_bytes = eff_bundle.get("privkey_bytes") or self.anonymity_guard.current_privkey_bytes()
        salt_bytes = eff_bundle.get("salt", b"armiger-rsi-delta")
        if isinstance(salt_bytes, str):
            salt_bytes = salt_bytes.encode("utf-8")
        responses = await swarm.put_mutable(
            privkey_bytes=priv_bytes,
            salt=salt_bytes,
            v=v_bytes,
            seq=int(eff_bundle["seq"]),
        )
        ok_count = sum(1 for r in responses if r.get(b"y") == b"r")
        return ok_count > 0, "published_anonymous_bep44", ok_count

    def broadcast_weights(
        self,
        delta_w: Optional[Sequence[float]] = None,
        delta_b: Optional[Sequence[float]] = None,
        *,
        n_samples: int = 25,
        reliability_bins: Optional[Sequence[Any]] = None,
        epoch: Optional[Union[int, float]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Export and sign DP-clipped weight deltas for DHT broadcast. Returns None if `opt_in_sharing=False`."""
        return self.export_signed_delta_bundle(
            delta_w=delta_w,
            delta_b=delta_b,
            reliability_bins=reliability_bins,
            n_samples=n_samples,
            epoch=epoch,
        )

    def get_pending_telemetry_batch(
        self,
        now_timestamp: Optional[float] = None,
    ) -> List[EncryptedTelemetryBundle]:
        """Release scheduled telemetry items whose Poisson jitter delay has elapsed, encrypted with AEAD."""
        if not self.opt_in_sharing or self.mode != DHTRSIMode.ANONYMOUS_CONTRIBUTE:
            if self.telemetry_queue:
                self._outbound_packets_blocked_opt_out += len(self.telemetry_queue)
                self.telemetry_queue.clear()
            else:
                self._outbound_packets_blocked_opt_out += 1
            return []

        if not self.aggregator_x25519_pubkey_hex:
            # Cannot encrypt without aggregator public key; retain queued items for later retry
            return []

        try:
            recipient_pk_bytes = self._validate_aggregator_pubkey_hex(self.aggregator_x25519_pubkey_hex)
        except ValueError as exc:
            self._telemetry_encryption_errors += 1
            self._last_telemetry_error = str(exc)
            now = now_timestamp if now_timestamp is not None else time.time()
            remaining_on_invalid_key: List[QueuedTelemetryItem] = []
            for item in self.telemetry_queue:
                if item.scheduled_dispatch_time <= now:
                    item.retry_count += 1
                    if item.retry_count < self.max_item_encryption_retries:
                        remaining_on_invalid_key.append(item)
                    else:
                        self._telemetry_dropped_backlog += 1
                else:
                    remaining_on_invalid_key.append(item)
            self.telemetry_queue = remaining_on_invalid_key
            return []

        now = now_timestamp if now_timestamp is not None else time.time()
        ready_items: List[QueuedTelemetryItem] = []
        remaining_items: List[QueuedTelemetryItem] = []

        for item in self.telemetry_queue:
            if item.scheduled_dispatch_time <= now:
                ready_items.append(item)
            else:
                remaining_items.append(item)

        if not ready_items:
            return []

        bundles: List[EncryptedTelemetryBundle] = []
        failed_items: List[QueuedTelemetryItem] = []
        for item in ready_items:
            try:
                payload = item.record.canonical_json_bytes()
                next_seq = max(getattr(self, "_telemetry_seq", self.current_seq), self.current_seq) + 1
                bundle = X25519ChaCha20Poly1305Engine.encrypt_bundle(
                    payload_bytes=payload,
                    recipient_pubkey_bytes=recipient_pk_bytes,
                    seq=next_seq,
                )
                self._telemetry_seq = next_seq
                bundles.append(bundle)
            except Exception as exc:
                self._telemetry_encryption_errors += 1
                self._last_telemetry_error = str(exc)
                item.retry_count += 1
                if item.retry_count < self.max_item_encryption_retries:
                    failed_items.append(item)
                else:
                    self._telemetry_dropped_backlog += 1

        self.telemetry_queue = remaining_items + failed_items
        return bundles

    # -----------------------------------------------------------------
    # Update Staging, Verification, and Golden Canary Gate
    # -----------------------------------------------------------------

    def verify_and_stage_update(
        self,
        manifest: ThresholdDHTManifest,
        bundle_bytes: bytes,
        attestations: List[PeerAttestation],
        candidate_predictor_fn: Optional[Callable[[CanaryItem], Dict[str, Any]]] = None,
        baseline_metrics: Optional[Dict[str, float]] = None,
        *,
        require_pq: Optional[bool] = None,
        require_authoritative_asn: Optional[bool] = None,
        asn_verifier: Optional[AuthoritativeASNVerifier] = None,
        bootstrap_checkpoint_seq: Optional[int] = None,
        authenticated_baseline: Optional[AuthenticatedConformalBaseline] = None,
        expected_parent_bundle_sha256: Optional[str] = None,
        expected_canary_suite_sha256: Optional[str] = None,
        release_envelope: Optional[AcyclicReleaseEnvelope] = None,
        trusted_evaluator_signers: Optional[Dict[str, str]] = None,
        require_acyclic_envelope: Optional[bool] = None,
    ) -> Tuple[bool, str, Optional[CanaryRegressionReport]]:
        """Verify signed DHT update manifest, check ASN quorum, validate safe formats, and run 500-canary gate."""
        if self.mode == DHTRSIMode.OFF:
            return False, "plugin_mode_off", None

        if candidate_predictor_fn is None:
            return False, "missing_candidate_predictor_fn", None

        if authenticated_baseline is not None:
            b_ok, b_reason = authenticated_baseline.verify_signature(self.trusted_signers)
            if not b_ok:
                return False, f"authenticated_baseline_failed:{b_reason}", None
            eff_parent_sha = (expected_parent_bundle_sha256 or self.current_bundle_sha256 or "").strip().lower()
            if not eff_parent_sha:
                return False, "authenticated_baseline_missing_expected_parent_bundle_sha256", None
            if authenticated_baseline.parent_bundle_sha256.strip().lower() != eff_parent_sha:
                return False, "authenticated_baseline_parent_bundle_mismatch", None
            eff_suite_sha = (expected_canary_suite_sha256 or self.canary_gate.suite_sha256()).strip().lower()
            if authenticated_baseline.canary_suite_sha256.strip().lower() != eff_suite_sha:
                return False, "authenticated_baseline_canary_suite_mismatch", None
            baseline_metrics = authenticated_baseline.to_metrics_dict()

        # 1. Verify BEP-44 monotonic sequence and Hybrid Ed25519 + WOTS+-SHAKE256 PQ threshold signatures
        eff_pq_signers = dict(self.trusted_pq_signers)
        if not eff_pq_signers:
            for ed_pk_hex, identity in self.trusted_signers.items():
                norm_ed = ThresholdDHTManifest._normalize_pubkey_id(ed_pk_hex)
                for wots_pk_hex in WOTSPlusSHAKE256._ED25519_TO_WOTS_PUBS.get(norm_ed, ()):
                    eff_pq_signers[wots_pk_hex] = identity
        require_pq_flag = self.require_pq if require_pq is None else bool(require_pq)
        ok, reason = ThresholdSignatureVerifier.verify(
            manifest=manifest,
            trusted_signers=self.trusted_signers,
            threshold_m=self.threshold_m,
            current_seq=self.current_seq,
            trusted_pq_signers=eff_pq_signers,
            require_pq=require_pq_flag,
            bootstrap_checkpoint_seq=bootstrap_checkpoint_seq,
        )
        if not ok:
            return False, f"manifest_threshold_failed:{reason}", None

        # 2. Check bundle SHA-256 digest
        actual_sha = hashlib.sha256(bundle_bytes).hexdigest()
        if actual_sha.lower() != manifest.bundle_sha256.lower():
            return False, "bundle_sha256_digest_mismatch", None

        # 2b. Enforce acyclic provenance envelope (disjoint contributors / evaluators / release signers)
        eff_req_env = (
            self.require_acyclic_envelope
            if require_acyclic_envelope is None
            else bool(require_acyclic_envelope)
        )
        if eff_req_env or release_envelope is not None:
            if release_envelope is None:
                return False, "missing_acyclic_release_envelope", None
            if (
                release_envelope.manifest.canonical_payload() != manifest.canonical_payload()
                or dict(release_envelope.manifest.signatures) != dict(manifest.signatures)
                or dict(release_envelope.manifest.pq_signatures) != dict(manifest.pq_signatures)
            ):
                return False, "release_envelope_manifest_mismatch", None
            eff_parent_sha = (expected_parent_bundle_sha256 or self.current_bundle_sha256 or "").strip().lower()
            if not eff_parent_sha:
                return False, "release_envelope_missing_expected_parent_bundle_sha256", None
            if release_envelope.parent_bundle_sha256.strip().lower() != eff_parent_sha:
                return False, "release_envelope_parent_bundle_mismatch", None
            eff_suite_sha = (expected_canary_suite_sha256 or self.canary_gate.suite_sha256()).strip().lower()
            if release_envelope.canary_suite_sha256.strip().lower() != eff_suite_sha:
                return False, "release_envelope_canary_suite_mismatch", None
            eff_eval_signers = (
                dict(trusted_evaluator_signers)
                if trusted_evaluator_signers is not None
                else dict(self.trusted_evaluator_signers)
            )
            env_ok, env_reason = AcyclicProvenanceVerifier.verify_envelope(
                release_envelope,
                trusted_release_signers=self.trusted_signers,
                trusted_evaluator_signers=eff_eval_signers,
                threshold_m=self.threshold_m,
                min_evaluator_asns=max(self.min_distinct_asns, manifest.min_asn_quorum),
                current_seq=self.current_seq,
                bootstrap_checkpoint_seq=bootstrap_checkpoint_seq,
                require_pq=require_pq_flag,
                asn_verifier=asn_verifier,
                trusted_release_pq_signers=eff_pq_signers,
            )
            if not env_ok:
                return False, f"acyclic_provenance_failed:{env_reason}", None
            if authenticated_baseline is None and release_envelope.authenticated_baseline is not None:
                baseline_metrics = release_envelope.authenticated_baseline.to_metrics_dict()

        # 3. Verify K-of-N distinct-ASN Sybil quorum
        q_ok, q_reason, asns = SybilQuorumVerifier.verify_quorum(
            attestations=attestations,
            target_hash=manifest.bundle_sha256,
            min_distinct_asns=max(self.min_distinct_asns, manifest.min_asn_quorum),
            asn_verifier=asn_verifier,
            require_authoritative_asn=require_authoritative_asn,
        )
        if not q_ok:
            return False, f"sybil_quorum_failed:{q_reason}", None

        # 4. Enforce Safe Formats only (SafeTensors, ONNX, JSON)
        f_ok, f_reason = SafeFormatValidator.validate(bundle_bytes, manifest.bundle_format)
        if not f_ok:
            return False, f"safe_format_violation:{f_reason}", None

        # 5. Mandatory 500-Item Golden Canary Regression Gate
        canary_report = self.canary_gate.evaluate(
            predictor_fn=candidate_predictor_fn,
            baseline_metrics=baseline_metrics,
        )
        if not canary_report.passed:
            return False, f"golden_canary_gate_rejected:{canary_report.reason}", canary_report

        # All checks passed! Advance monotonic sequence and active bundle digest
        self.current_seq = manifest.seq
        self.current_bundle_sha256 = actual_sha.lower()
        return True, "update_staged_and_verified", canary_report
