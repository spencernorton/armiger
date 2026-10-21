"""test_anonymous_opt_in_dht_sharing.py

Comprehensive Pytest Suite for Armiger Distributed Anonymous & Opt-In-By-Default
Weight and Catalog Data Sharing (`opt_in_sharing=False` by default + `AnonymityGuard`).
"""
from __future__ import annotations

import asyncio
import json
import math
import os
from pathlib import Path
import sys

import pytest

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from armiger import ArmigerPipeline
from armiger_dht_db_sync import ArmigerDHTDBSync, ArmigerDHTDatabaseSync
from armiger_dht_rsi import (
    AnonymityGuard,
    AnonymityViolationError,
    ArmigerDHTRSIEngine,
    AudioWebhookAdapter,
    MovieWebhookAdapter,
    PublicationWebhookAdapter,
    QBittorrentAdapter,
    SeriesWebhookAdapter,
    TransmissionAdapter,
    resolve_opt_in_sharing,
)
from armiger_ultra import ArmigerUltraPipeline
from bep44_udp_wire_harness import LocalKademliaSwarm
from benchmarks.proof_anonymous_opt_in_sharing import (
    generate_500_adversarial_canaries,
    run_proof_1_default_zero_egress,
    run_proof_2_canary_pii_firewall,
    run_proof_3_unlinkability_and_dp_convergence,
)


def test_opt_in_sharing_env_and_default_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify resolve_opt_in_sharing defaults to False and only enables on explicit True or env 1/true/yes."""
    monkeypatch.delenv("ARMIGER_DHT_SHARE_OPT_IN", raising=False)
    assert resolve_opt_in_sharing() is False
    assert resolve_opt_in_sharing(False) is False
    assert resolve_opt_in_sharing(True) is True

    for falsy in ("0", "false", "no", "off", "", "random"):
        monkeypatch.setenv("ARMIGER_DHT_SHARE_OPT_IN", falsy)
        assert resolve_opt_in_sharing() is False

    for truthy in ("1", "true", "TRUE", "yes", "YES"):
        monkeypatch.setenv("ARMIGER_DHT_SHARE_OPT_IN", truthy)
        assert resolve_opt_in_sharing() is True
        # Explicit False argument still overrides env var
        assert resolve_opt_in_sharing(False) is False


def test_all_adapters_and_engines_default_opt_in_false(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify every adapter, engine, DB sync, pipeline, and bridge defaults to opt_in_sharing=False."""
    monkeypatch.delenv("ARMIGER_DHT_SHARE_OPT_IN", raising=False)
    rsi = ArmigerDHTRSIEngine()
    db = ArmigerDHTDBSync()
    s_ad = SeriesWebhookAdapter()
    m_ad = MovieWebhookAdapter()
    a_ad = AudioWebhookAdapter()
    p_ad = PublicationWebhookAdapter()
    q_ad = QBittorrentAdapter()
    t_ad = TransmissionAdapter()

    for obj in (rsi, db, s_ad, m_ad, a_ad, p_ad, q_ad, t_ad):
        assert obj.opt_in_sharing is False
        st = obj.sharing_status()
        assert st["opt_in_sharing_enabled"] is False
        assert st["outbound_bundles_shared"] == 0
        assert st["outbound_bytes_shared"] == 0

    assert rsi.export_signed_bundle(delta_w=[0.1, 0.2], delta_b=[0.0], n_samples=50) is None
    assert rsi.export_signed_delta_bundle(delta_w=[0.1, 0.2], delta_b=[0.0], n_samples=50) is None
    assert rsi.broadcast_weights([0.1, 0.2], [0.0], n_samples=50) is None
    assert rsi.get_pending_telemetry_batch() == []
    assert rsi.sharing_status()["outbound_packets_blocked_opt_out"] >= 4

    assert db.export_signed_bundle() is None
    assert db.export_signed_delta_bundle() is None
    assert db.sharing_status()["outbound_packets_blocked_opt_out"] >= 2
    db.close()


def test_proof_1_default_zero_egress_100_events() -> None:
    """Execute Proof 1: 100 synthetic webhook/catalog events across default instances -> 0 outbound packets."""
    res = asyncio.run(run_proof_1_default_zero_egress())
    assert res["passed"] is True
    assert res["synthetic_events_processed"] == 100
    assert res["all_components_default_opt_in_false"] is True
    assert res["outbound_bundles_shared"] == 0
    assert res["outbound_bytes_shared"] == 0
    assert res["udp_put_packets_emitted"] == 0
    assert res["outbound_attempts_blocked_by_opt_out"] >= 100
    assert res["inbound_readonly_verification_passed"] is True


def test_anonymity_guard_ephemeral_key_rotation_and_coarse_timestamps() -> None:
    """Verify Ephemeral Ed25519 key rotation and coarse UTC hour timestamp bucketing."""
    guard = AnonymityGuard(ephemeral_identity=True, min_k_samples=25, clip_norm=1.0, dp_sigma=0.05)
    seen_keys = {guard.current_pubkey_hex()}
    for ep in range(20):
        new_pk = guard.rotate_ephemeral_identity(epoch=ep)
        assert len(new_pk) == 64
        assert new_pk not in seen_keys
        seen_keys.add(new_pk)

    # Coarse UTC hour bucketing: floor(ts / 3600) * 3600
    for offset in (0, 1, 42, 1799, 3599):
        b_ts = guard.bucket_timestamp(1792350000 + offset)
        assert b_ts == 1792350000
        assert b_ts % 3600 == 0


def test_anonymity_guard_l2_clipping_dp_and_k_anonymity() -> None:
    """Verify L2 norm clipping, Gaussian DP noise, int8 quantization, and k-anonymity enforcement."""
    guard = AnonymityGuard(min_k_samples=25, clip_norm=1.0, dp_sigma=0.05)

    # Rejects n_samples < 25
    for bad_n in (0, 1, 5, 12, 24):
        with pytest.raises(AnonymityViolationError, match="k_anonymity"):
            guard.clip_and_privatize_weights([0.5, -0.5], [0.1], n_samples=bad_n)

    # Clips large vectors with L2 norm >> 1.0 down to <= 1.0
    huge_w = [10.0, -10.0, 10.0, -10.0]  # L2 norm = 20.0
    huge_b = [5.0, -5.0]
    res = guard.clip_and_privatize_weights(huge_w, huge_b, n_samples=55)
    assert res["pre_dp_l2_norm"] > 20.0
    assert res["clipped_l2_norm"] <= 1.0 + 1e-9
    post_dp_l2 = math.sqrt(sum(x * x for x in res["delta_w_dp"]) + sum(x * x for x in res["delta_b_dp"]))
    assert post_dp_l2 <= 1.05
    assert res["n_bucket"] == 50
    assert all(-127 <= q <= 127 for q in res["delta_w_q8"])

    # Reliability bins k-anonymity check
    with pytest.raises(AnonymityViolationError, match="k_anonymity"):
        guard.sanitize_reliability_bins([{"bin": 0, "conf": 0.5, "acc": 0.5, "n": 10}])

    clean_bins = guard.sanitize_reliability_bins([{"bin": 0, "conf": 0.8, "acc": 0.78, "n": 42}])
    assert len(clean_bins) == 1
    assert clean_bins[0][3] == 40


def test_anonymity_guard_domain_separated_blake2b_catalog_hashing() -> None:
    """Verify 16-hex-char BLAKE2b-64(domain='armiger-catalog-v2' || normalized_key) and PII rejection."""
    h1 = AnonymityGuard.hash_catalog_key("sintel|2010")
    h2 = AnonymityGuard.hash_catalog_key("  SINTEL | 2010  ")
    assert h1 == h2
    assert len(h1) == 16
    assert all(c in "0123456789abcdef" for c in h1)

    # Rejects raw filenames, release titles, IPs, paths, URLs, infohashes, external IDs
    bad_keys = [
        "/" + "Users" + "/alice/Downloads/Sintel.2010.mkv",
        "Sintel.2010.2160p.UHD.BluRay.REMUX.HEVC.TrueHD.7.1",
        "magnet:?xt=urn:btih:1111222233334444555566667777888899990000",
        "198.51.100.42",
        "tt1234567",
        "tmdb:693134",
        "https://tracker.example.org/announce?passkey=secret123",
    ]
    for bk in bad_keys:
        with pytest.raises(AnonymityViolationError):
            AnonymityGuard.hash_catalog_key(bk)


def test_proof_2_500_adversarial_canaries_blocked_zero_wire_leak() -> None:
    """Execute Proof 2: 500 adversarial canaries across 12-node UDP swarm -> 500/500 blocked, 0/500 on wire."""
    canaries = generate_500_adversarial_canaries()
    assert len(canaries) == 500
    res = asyncio.run(run_proof_2_canary_pii_firewall())
    assert res["passed"] is True
    assert res["total_adversarial_canaries_injected"] == 500
    assert res["canaries_blocked_or_stripped_before_wire"] == 500
    assert res["block_rate_pct"] == 100.0
    assert res["canary_substrings_on_udp_wire"] == 0
    assert res["wire_leak_rate_pct"] == 0.0


def test_proof_3_12_node_udp_swarm_unlinkability_and_dp_convergence() -> None:
    """Execute Proof 3: 12-node UDP swarm over 5 epochs -> 60/60 unique keys, ECE & F1 convergence."""
    res = asyncio.run(run_proof_3_unlinkability_and_dp_convergence())
    assert res["passed"] is True
    assert res["swarm_nodes"] == 12
    assert res["epochs_run"] == 5
    assert res["unique_ephemeral_ed25519_pubkeys"] == 60
    assert res["cross_epoch_key_reuse_count"] == 0
    assert res["coarse_hour_timestamp_bucketing_rate_pct"] == 100.0
    assert res["final_calibration_ece"] < res["initial_calibration_ece"] * 0.35
    assert res["final_armiger_ultra_f1"] > res["initial_armiger_ultra_f1"] + 0.025
