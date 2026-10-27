"""End-to-End Integration Test Suite for `ArmigerUltraPipeline` (All 6 Plugins + 11 Heads)."""
from __future__ import annotations

import dataclasses

from pathlib import Path
import sys

import numpy as np
import pytest

from armiger.artifacts import model_directory
_THIS_DIR = model_directory()
ROOT_DIR = _THIS_DIR

from armiger.armiger_ultra_pipeline import (
    ALL_11_HEAD_NAMES,
    ArmigerDHTDBSync,
    ArmigerDHTRSIEngine,
    ArmigerUltraPipeline,
    ArmigerUltraPrediction,
    AudioLanguageGuard,
    CSAMAndAbuseGuard,
    CalibrationAndDHT,
    SubscriptionConfig,
    SwarmSurvivalNet,
    SyncMode,
    ZeroCueCatalog,
    create_mock_quorum_attestations,
    generate_ed25519_keypair,
    sign_tombstone_list,
)


@pytest.fixture(scope="module")
def pipeline() -> ArmigerUltraPipeline:
    """Shared ArmigerUltraPipeline loaded from the release directory via `from_Directory`."""
    pipe = ArmigerUltraPipeline.from_Directory(_THIS_DIR, variant="ultra", preload_catalog=False)
    yield pipe
    pipe.close()


def test_from_directory_and_aliases(pipeline: ArmigerUltraPipeline) -> None:
    """Verify `from_Directory`, `from_directory`, and `from_pretrained` factory methods and 6 plugin instances."""
    assert isinstance(pipeline, ArmigerUltraPipeline)
    assert isinstance(pipeline.calibrator, CalibrationAndDHT)
    assert isinstance(pipeline.audio_guard, AudioLanguageGuard)
    assert isinstance(pipeline.rsi_engine, ArmigerDHTRSIEngine)
    assert isinstance(pipeline.dht_db_sync, ArmigerDHTDBSync)
    assert isinstance(pipeline.csam_guard, CSAMAndAbuseGuard)

    # Verify int8 CPU fast-path loader via snake_case `from_directory` and `from_pretrained`
    cpu_pipe = ArmigerUltraPipeline.from_directory(_THIS_DIR, variant="int8")
    assert cpu_pipe.onnx_path is not None and cpu_pipe.onnx_path.name == "armiger.int8.onnx"
    cpu_pipe.close()

    pretrained_pipe = ArmigerUltraPipeline.from_pretrained(_THIS_DIR, variant="int8")
    assert isinstance(pretrained_pipe, ArmigerUltraPipeline)
    pretrained_pipe.close()


def test_11_head_unified_classification_movie(pipeline: ArmigerUltraPipeline) -> None:
    """Verify full 11-head output structure on a clean 4K BluRay movie torrent."""
    item = {
        "infohash": "1111222233334444555566667777888899990000",
        "name": "Sintel.2010.2160p.UHD.BluRay.REMUX.HEVC.TrueHD.7.1.Atmos-OPEN",
        "size": 74_500_000_000,
        "files": [
            {"path": "Sintel.2010.2160p.UHD.BluRay.REMUX.HEVC.TrueHD.7.1.Atmos-OPEN.mkv", "size": 74_500_000_000}
        ],
        "swarm": {
            "tracker_known": True,
            "tracker_seeders": 320,
            "tracker_leechers": 18,
            "tracker_completed": 2450,
            "best_tracker": "udp://bt1.archive.org:1337/announce",
        },
    }
    pred = pipeline.classify(item, use_zero_cue_catalog=False)
    assert isinstance(pred, ArmigerUltraPrediction)

    # All 11 heads must be present
    for head_name in ALL_11_HEAD_NAMES:
        assert head_name in pred.heads, f"Missing head: {head_name}"
        assert head_name in pred

    assert pred["category"] == "movie"
    assert pred["adult"] == "non_adult"
    assert pred["junk"] == "clean"
    assert pred["owner_scope"] == "allowed_type"
    assert pred["csam_safety"]["is_safe"] is True
    assert pred["mondrian_gates"]["passed_all"] is True
    assert pred["swarm_viability"]["viability_tier"] == "alive_viable"
    assert pred["swarm_viability"]["effective_seeders"] == 320
    assert pred["honeypot_trap"]["is_honeypot_trap"] is False


def test_zero_category_flip_invariance_across_seeders(pipeline: ArmigerUltraPipeline) -> None:
    """Verify 0% category flip guarantee when swarm seeders vary from 0 to 10,000."""
    base_item = {
        "name": "Cosmos.Laundromat.S01.1080p.WEB-DL.DDP5.1.Atmos.H.264-OPEN",
        "size": 28_000_000_000,
        "files": [
            {"path": f"Cosmos.Laundromat.S01E0{i}.1080p.WEB-DL.mkv", "size": 3_100_000_000}
            for i in range(1, 10)
        ],
    }

    item_zero = {
        **base_item,
        "swarm": {"tracker_known": True, "tracker_seeders": 0, "tracker_leechers": 0, "tracker_completed": 0, "liveness_status": "dead"},
    }
    item_live = {
        **base_item,
        "swarm": {"tracker_known": True, "tracker_seeders": 10000, "tracker_leechers": 400, "tracker_completed": 50000, "liveness_status": "alive"},
    }

    pred_zero = pipeline.classify(item_zero, use_zero_cue_catalog=False)
    pred_live = pipeline.classify(item_live, use_zero_cue_catalog=False)

    # Category and category probability distribution must be 100% invariant to seeders
    assert pred_zero["category"] == pred_live["category"] == "tv_show"
    for cls_name, p0 in pred_zero["heads"]["category"]["probs"].items():
        p1 = pred_live["heads"]["category"]["probs"][cls_name]
        assert abs(p0 - p1) < 1e-9, f"Category prob drifted for {cls_name}: {p0} vs {p1}"

    # Meanwhile, orthogonal swarm_viability head accurately reflects dead vs alive swarm
    assert pred_zero["swarm_viability"]["viability_tier"] == "dead_known_zero"
    assert pred_live["swarm_viability"]["viability_tier"] == "alive_viable"
    assert pred_live["swarm_viability"]["viability_prob"] > pred_zero["swarm_viability"]["viability_prob"] + 0.50


def test_plugin_1_and_2_fake_seeder_honeypot_trap_detection(pipeline: ArmigerUltraPipeline) -> None:
    """Verify SwarmSurvivalNet + SwarmPriors Way-3 catch fake-seeder malware honeypots."""
    honeypot_item = {
        "infohash": "aaaa1111bbbb2222cccc3333dddd4444eeee5555",
        "name": "OpenEngine.2026.Fake.Seeder.Malware.Trap",
        "size": 8_500_000,
        "files": [{"path": "OpenEngine_Installer.exe", "size": 8_500_000}],
        "swarm": {
            "tracker_known": True,
            "tracker_seeders": 6500,
            "tracker_leechers": 1200,
            "tracker_completed": 0,
            "has_exec_or_scr": True,
        },
    }
    pred = pipeline.classify(honeypot_item, use_zero_cue_catalog=False)
    assert pred["honeypot_trap"]["is_honeypot_trap"] is True
    assert pred["honeypot_trap"]["label"] == "honeypot_trap"
    assert pred["junk"] == "junk"
    assert pred["mondrian_gates"]["honeypot_guard_passed"] is False
    assert pred["mondrian_gates"]["passed_all"] is False


def test_plugin_3_zero_cue_structural_and_catalog(pipeline: ArmigerUltraPipeline) -> None:
    """Verify Plugin #3 structural disambiguation (audiobook .m4b, multi-episode TV, comics)."""
    audiobook_item = {
        "name": "LibriVox - Pride and Prejudice [Unabridged]",
        "size": 1_450_000_000,
        "files": [{"path": "Pride and Prejudice - Part 1.m4b", "size": 1_450_000_000}],
    }
    pred_ab = pipeline.classify(audiobook_item, use_zero_cue_catalog=True)
    assert pred_ab["category"] == "audiobook"
    assert pred_ab["plugin_trace"]["plugin_3_zero_cue_catalog"]["resolution_source"] == "structural_m4b"


def test_plugin_4_audio_language_and_anime_dub_guard(pipeline: ArmigerUltraPipeline) -> None:
    """Verify Plugin #4 cross-head audio language & anime dub guard."""
    # Case 1: Foreign anime with English subtitles only (.ENG.Sub) -> non_english audio, subbed_only
    subbed_anime = {
        "name": "[OpenAnime] Spring - 01 (1080p) [JPN] [ENG Sub].mkv",
        "size": 1_400_000_000,
        "language": "ja",
        "files": [{"path": "[OpenAnime] Spring - 01 (1080p).mkv", "size": 1_400_000_000}],
    }
    pred_sub = pipeline.classify(subbed_anime, use_zero_cue_catalog=False)
    assert pred_sub["lang"] == "non_english"
    assert pred_sub["heads"]["anime"]["anime_dub_status"] in ("subbed_only", "raw_or_vernacular", "non_anime")

    # Case 2: Explicit Dual-Audio anime release -> multi_or_unknown
    dual_anime = {
        "name": "[OpenAnime] Caminandes - S01 (1080p) [HEVC x265 10bit] [Dual-Audio] [Multi-Subs]",
        "size": 12_000_000_000,
        "files": [
            {"path": f"Caminandes - S01E0{i} [Dual-Audio].mkv", "size": 1_200_000_000}
            for i in range(1, 11)
        ],
    }
    pred_dual = pipeline.classify(dual_anime, use_zero_cue_catalog=False)
    assert pred_dual["lang"] == "multi_or_unknown"


def test_plugin_5_rsi_feedback_ingestion_and_verification(pipeline: ArmigerUltraPipeline) -> None:
    """Verify Plugin #5 universal RSI ground-truth webhook ingestion and anonymization."""
    movie_webhook = {
        "eventType": "Download",
        "movie": {"title": "Tears of Steel", "year": 2024, "movieCatalogId": 693134},
        "release": {
            "releaseTitle": "Tears.of.Steel.2012.2160p.WEB-DL.DDP5.1.Atmos.DV.MKV.x265?passkey=secret999",
            "size": 24_000_000_000,
        },
        "movieFile": {
            "relativePath": "Tears.of.Steel.2012.2160p.mkv",
            "size": 24_000_000_000,
            "duration": 9960,
            "mediaInfo": {
                "videoCodec": "hevc",
                "audioFormat": "eac3",
                "audioChannels": 5.1,
                "audioLanguages": "eng",
                "subtitles": "eng",
            },
        },
    }
    ok, msg, anon_rec = pipeline.ingest_rsi_feedback(movie_webhook, adapter="movie_webhook")
    assert ok is True
    assert msg == "verified_anonymized_and_queued"
    assert anon_rec is not None
    assert anon_rec.category == "movie"
    assert "secret999" not in anon_rec.canonical_json_bytes().decode("utf-8")


def test_plugin_6_csam_guard_dht_db_sync_and_tombstones() -> None:
    """Verify Plugin #6 CSAM/malware blocking, encrypted shard sync, and Ed25519 tombstone purge."""
    root_priv, root_pub = generate_ed25519_keypair()
    sub = SubscriptionConfig(
        categories={"movie", "tv_show"},
        languages={"en"},
        quality_tiers={"uhd_4k", "hd_1080p", "any"},
        enable_xxx=False,
        sync_mode=SyncMode.SUBSCRIBE_AND_SEED,
        min_quorum=5,
        min_distinct_asns=5,
    )
    pipe = ArmigerUltraPipeline.from_Directory(
        _THIS_DIR,
        dht_sync_subscription=sub,
        trusted_root_pubkeys_hex=[root_pub.hex()],
        dht_master_secret=b'armiger-shared-topic-secret-32b!',
    )

    try:
        # 1. CSAM / Abuse title must be blocked immediately and rejected from DHT DB
        csam_item = {
            "infohash": "bad0000000000000000000000000000000000001",
            "name": "c.h.i.l.d p0rn preteen hardcore collection",
            "size": 500_000_000,
            "files": [{"path": "video.mp4", "size": 500_000_000}],
        }
        pred_csam = pipe.classify(csam_item, use_zero_cue_catalog=False, ingest_to_dht_db=True)
        assert pred_csam["csam_safety"]["is_safe"] is False
        assert pred_csam["csam_safety"]["label"] == "blocked"
        assert pred_csam["mondrian_gates"]["passed_all"] is False
        assert pred_csam["plugin_trace"]["plugin_6_dht_db_sync"]["dht_ingest_status"].startswith("rejected:")

        # 2. Disguised malware dropper (.mkv.exe) in movie torrent must be blocked
        malware_item = {
            "infohash": "bad0000000000000000000000000000000000002",
            "name": "Elephants.Dream.2006.1080p.WEB-DL.x264",
            "size": 25_000_000,
            "files": [{"path": "Elephants.Dream.2006.1080p.mkv.exe", "size": 25_000_000}],
        }
        pred_mal = pipe.classify(malware_item, use_zero_cue_catalog=False, ingest_to_dht_db=True)
        assert pred_mal["csam_safety"]["is_safe"] is False
        assert pred_mal["junk"] == "junk"
        assert pred_mal["mondrian_gates"]["passed_all"] is False

        # 3. Clean movie torrent classified & ingested into local DHT DB
        clean_ih = "cafe1234cafe1234cafe1234cafe1234cafe1234"
        clean_item = {
            "infohash": clean_ih,
            "name": "Coffee.Run.2020.2160p.UHD.BluRay.REMUX.HEVC.DTS-HD.MA.5.1-OPEN",
            "size": 62_000_000_000,
            "quality_tier": "uhd_4k",
            "files": [{"path": "Coffee.Run.2020.2160p.mkv", "size": 62_000_000_000}],
            "swarm": {"tracker_known": True, "tracker_seeders": 95, "tracker_completed": 410},
        }
        pred_clean = pipe.classify(clean_item, use_zero_cue_catalog=False, ingest_to_dht_db=True)
        assert pred_clean["csam_safety"]["is_safe"] is True
        assert pred_clean["plugin_trace"]["plugin_6_dht_db_sync"]["dht_ingest_status"] == "ingested"
        assert pipe.dht_db_sync.db.get_record(clean_ih) is not None

        # 4. Build encrypted shard, sign with 5-ASN quorum, and sync into a second pipeline instance
        rec = pipe.dht_db_sync.db.get_record(clean_ih)
        assert rec is not None
        shard_id = f"movie/en/uhd_4k/{rec.epoch}"
        manifest_draft, bundle = pipe.dht_db_sync.build_and_sign_shard(
            shard_id=shard_id,
            records=[rec],
            root_privkey=root_priv,
            seq=1,
        )
        attestations = create_mock_quorum_attestations(
            manifest_draft.canonical_digest(),
            distinct_asns=[13335, 15169, 16509, 8075, 24940],
        )
        manifest_signed = dataclasses.replace(manifest_draft, attestations=attestations)

        peer_pipe = ArmigerUltraPipeline.from_Directory(
            _THIS_DIR,
            dht_sync_subscription=sub,
            trusted_root_pubkeys_hex=[root_pub.hex()],
        dht_master_secret=b'armiger-shared-topic-secret-32b!',
        )
        try:
            ok_sync, msg_sync, count_sync = peer_pipe.sync_dht_shard(manifest_signed, bundle)
            assert ok_sync is True
            assert msg_sync == "shard_applied_successfully"
            assert count_sync == 1
            assert peer_pipe.dht_db_sync.db.get_record(clean_ih) is not None

            # 5. Apply Ed25519-signed blinded SHA-256 tombstone and verify immediate purge
            tombstone_manifest = sign_tombstone_list(
                seq=1,
                infohashes_or_blinded=[clean_ih],
                private_key_bytes=root_priv,
            )
            ok_t, msg_t, new_t, purged_t = peer_pipe.apply_tombstones(tombstone_manifest)
            assert ok_t is True
            assert new_t == 1
            assert purged_t == 1
            assert peer_pipe.dht_db_sync.db.get_record(clean_ih) is None
        finally:
            peer_pipe.close()
    finally:
        pipe.close()


def test_batch_callable_and_live_zero_cue_catalog_lookup(pipeline: ArmigerUltraPipeline) -> None:
    """Verify batch __call__ interface and live 929,400-entry Zero-Cue Catalog disambiguation."""
    batch = [
        {
            "name": "Big.Buck.Bunny.2008.1080p.BluRay.x264",
            "size": 12_000_000_000,
            "files": [{"path": "Big.Buck.Bunny.2008.mkv", "size": 12_000_000_000}],
        },
        {
            "name": "OpenCourseWare - Complete Python Programming Course From Foundations to Systems",
            "size": 4_200_000_000,
            "files": [{"path": "01-Introduction.mp4", "size": 200_000_000}],
        },
        {
            "name": "Pepper.and.Carrot.Issue.001.2024.Digital.CC-BY.cbz",
            "size": 85_000_000,
            "files": [{"path": "Pepper.and.Carrot.001.cbz", "size": 85_000_000}],
        },
    ]
    preds = pipeline(batch, use_zero_cue_catalog=True)
    assert len(preds) == 3
    assert preds[0]["category"] == "movie"
    assert preds[1]["category"] == "course"
    assert preds[2]["category"] == "comic"

    # Also verify direct lookup on the loaded 929.4K Zero-Cue catalog
    assert len(pipeline.zero_cue_catalog.catalog.get("title_year_map", {})) > 400_000


def test_mondrian_conformal_safety_gates_adult_isolation_and_scope(pipeline: ArmigerUltraPipeline) -> None:
    """Verify Mondrian conformal gates isolate adult content and reject unisolated adult leakage."""
    ok_clean, reason_clean, viol_clean = pipeline.safety_gate.evaluate(
        p_adult=0.02,
        p_junk=0.01,
        p_excluded_scope=0.05,
        target_category="movie",
    )
    assert ok_clean is True
    assert reason_clean == "passed"
    assert viol_clean is None

    # Unisolated adult leakage into mainstream category ('movie') must be rejected
    ok_leak, reason_leak, viol_leak = pipeline.safety_gate.evaluate(
        p_adult=0.85,
        p_junk=0.01,
        p_excluded_scope=0.05,
        target_category="movie",
    )
    assert ok_leak is False
    assert "unisolated_adult_leakage" in reason_leak
    assert viol_leak is not None and viol_leak.value == "unisolated_adult"

    # Adult content properly isolated in 'xxx' category passes safety gate
    ok_xxx, reason_xxx, _ = pipeline.safety_gate.evaluate(
        p_adult=0.85,
        p_junk=0.01,
        p_excluded_scope=0.05,
        target_category="xxx",
    )
    assert ok_xxx is True
    assert reason_xxx == "passed"


def test_malware_sideload_zip_lnk_traps_and_benign_controls(pipeline: ArmigerUltraPipeline) -> None:
    """Verify all 7 MawaStealer v1/v2 & Vidar DLL-sideloading/ZIP/LNK traps are caught while legitimate releases pass."""
    malware_samples = [
        {
            "name": "[OpenAnime] Spring Chronicles 2nd Season - 06 [1080p][Multiple Subtitle] [ENG][POR-BR]",
            "size": 706678900,
            "files": [
                {"path": "Subs/[OpenAnime] Spring Chronicles 2nd Season - 06 [1080p][Multiple Subtitle] [ENG][POR-BR].mkv", "size": 702545920},
                {"path": "[OpenAnime] Spring Chronicles 2nd Season - 06 [1080p][Multiple Subtitle] [ENG][POR-BR].mkv.exe", "size": 992672},
                {"path": "libvlc.dll", "size": 3140308},
            ],
        },
        {
            "name": "[OpenAnime] Northern Kingdom - 02 (1080p) [0521E791]",
            "size": 1293942784,
            "files": [
                {"path": "Subs/[OpenAnime] Northern Kingdom - 02 (1080p) [0521E791].mkv", "size": 1289748480},
                {"path": "[OpenAnime] Northern Kingdom - 02 (1080p) [0521E791].mkv .exe", "size": 992672},
                {"path": "libvlc.dll", "size": 3201632},
            ],
        },
        {
            "name": "[OpenAnime] Crimson Academy - 01 (1080p) [F0246482]",
            "size": 1419771904,
            "files": [
                {"path": "Subs/[OpenAnime] Crimson Academy - 01 (1080p) [F0246482].mkv", "size": 1415577600},
                {"path": "[OpenAnime] Crimson Academy - 01 (1080p) [F0246482].mkv.exe", "size": 992672},
                {"path": "libvlc.dll", "size": 3201632},
            ],
        },
        {
            "name": "[OpenAnime] Crimson Academy - 01 (1080p) [F0246482].zip",
            "size": 1153433600,
            "ext": "zip",
            "files": [
                {"path": "[OpenAnime] Crimson Academy - 01 (1080p) [F0246482].zip", "size": 1153433600},
            ],
        },
        {
            "name": "[OpenAnime] Youth New Game - 02 (720p) [D4325D3F]",
            "size": 4194304,
            "files": [
                {"path": "[OpenAnime] Youth New Game - 02 (720p) [D4325D3F].mkv.exe", "size": 992672},
                {"path": "libvlc.dll", "size": 3201632},
            ],
        },
        {
            "name": "[OpenAnime] Stone World S4 - 11 (1080p) [3373774E].zip",
            "size": 452984832,
            "ext": "zip",
            "files": [
                {"path": "[OpenAnime] Stone World S4 - 11 (1080p) [3373774E].zip", "size": 452984832},
            ],
        },
        {
            "name": "[OpenAnime] Stone World S4 - 11 (1080p) [3373774E]",
            "size": 453371904,
            "files": [
                {"path": "My movies.lnk", "size": 286720},
                {"path": "My moviese/[OpenAnime] Stone World S4 - 11 (1080p) [3373774E].mp4", "size": 452984832},
                {"path": "My moviese/1.png", "size": 100352},
            ],
        },
    ]

    for sample in malware_samples:
        pred = pipeline.classify(sample, use_zero_cue_catalog=False)
        assert pred["junk"] == "junk", f"Failed to flag malware sample: {sample['name']}"
        assert pred["heads"]["junk"]["probs"]["junk"] > 0.95
        assert pred["honeypot_trap"]["is_honeypot_trap"] is True
        assert pred["csam_safety"]["is_safe"] is False
        assert pred["mondrian_gates"]["passed_all"] is False

    legit_ep = {
        "name": "[OpenAnime] Stone World S4 - 11 (1080p) [3373774E].mkv",
        "size": 1415577600,
        "ext": "mkv",
        "files": [{"path": "[OpenAnime] Stone World S4 - 11 (1080p) [3373774E].mkv", "size": 1415577600}],
    }
    pred_ep = pipeline.classify(legit_ep, use_zero_cue_catalog=False)
    assert pred_ep["junk"] == "clean"
    assert pred_ep["heads"]["junk"]["probs"]["junk"] < 0.03
    assert pred_ep["honeypot_trap"]["is_honeypot_trap"] is False
    assert pred_ep["csam_safety"]["is_safe"] is True
    assert pred_ep["mondrian_gates"]["passed_all"] is True

    legit_tiny_stub = {
        "name": "Sintel.2010.1080p.BluRay.H264.AAC-OPEN",
        "size": 2_100_054_133,
        "files": [
            {"path": "Sintel.2010.1080p.BluRay.H264.AAC-OPEN.mp4", "size": 2_100_000_000},
            {"path": "README.txt", "size": 34},
            {"path": "DO_NOT_MIRROR.exe", "size": 99},
            {"path": "Subs/2_English.srt", "size": 54_000},
        ],
    }
    pred_stub = pipeline.classify(legit_tiny_stub, use_zero_cue_catalog=False)
    assert pred_stub["junk"] == "clean"
    assert pred_stub["heads"]["junk"]["probs"]["junk"] < 0.05
    assert pred_stub["honeypot_trap"]["is_honeypot_trap"] is False
    assert pred_stub["csam_safety"]["is_safe"] is True
    assert pred_stub["mondrian_gates"]["passed_all"] is True


def test_standalone_armiger_onnx_featurizer_and_parity() -> None:
    """Verify standalone `ArmigerStandaloneONNX` (`armiger_inference.py`) runs both INT8 and FP32 ONNX models."""
    from armiger.armiger_inference import ArmigerStandaloneONNX

    from armiger.artifacts import model_directory
    here = model_directory()
    m_int8 = ArmigerStandaloneONNX(here / "armiger.int8.onnx")
    m_fp32 = ArmigerStandaloneONNX(here / "armiger_ultra.fp32.onnx")

    sample = {
        "name": "Tears.of.Steel.2012.1080p.BluRay.x264.mkv",
        "size": 2_400_000_000,
        "files": [{"path": "Tears.of.Steel.2012.1080p.BluRay.x264.mkv", "size": 2_400_000_000}],
    }
    r_i8 = m_int8.predict(sample)
    r_fp = m_fp32.predict(sample)
    assert r_i8["category"] == "movie"
    assert r_fp["category"] == "movie"
    assert r_i8["heads"]["junk"] == "clean"
    assert r_fp["heads"]["junk"] == "clean"
    assert abs(r_i8["p_junk"] - r_fp["p_junk"]) < 0.01


def test_swarm_residual_adapter_pytorch_forward_pass() -> None:
    """Verify `SwarmResidualAdapter` executes real PyTorch `way2_encoders` and `junk_encoder` layers from `swarm_adapter.pt`."""
    from armiger.swarm_survival import SwarmResidualAdapter

    from armiger.artifacts import model_directory
    here = model_directory()
    adapter = SwarmResidualAdapter.from_pretrained(here / "swarm_adapter.pt")
    assert len(adapter.state) in (48, 54)
    assert hasattr(adapter, "way2_encoders")
    assert hasattr(adapter, "junk_encoder")

    res_warm = adapter.evaluate_swarm(
        title="Sintel.2010.2160p.WEB-DL.x265",
        seeders=8500,
        leechers=12,
        completed=0,
        total_size_bytes=15_000_000,
        files=[{"path": "Codec_Pack_Setup.exe", "size": 15_000_000}],
        sources=[{"source": "dht", "seeders": 8500, "leechers": 12, "completed": 0}],
        category_pred="movie",
        p_junk_neural=0.02,
        swarm_extra={"has_exec_or_scr": True},
    )
    assert res_warm["category"] == "movie"
    assert res_warm["p_junk_adapter"] > 0.90
    assert res_warm["adapter_junk_logit_delta"] > 1.0
    assert res_warm["is_honeypot_blocked"] is True
