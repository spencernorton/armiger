"""500-Item Golden Canary Regression Gate and Byzantine-Resilient Federated Aggregator."""
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
import re
import secrets
import sys
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

from cryptography.exceptions import InvalidSignature, InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ed25519, x25519
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF


from .dht_rsi_models_and_privacy import (
    AnonymizedTelemetryRecord,
    BlindedTitleStem,
    BundleFormat,
    CanaryItem,
    CanaryRegressionReport,
    CatalogConsensusResult,
    ContinuousAggregationResult,
    ContinuousDeltaSubmission,
    EncryptedTelemetryBundle,
    PeerAttestation,
    TelemetryAnonymizer,
    TelemetrySubmission,
    ThresholdDHTManifest,
    ZeroCueCatalogDelta,
)
from .dht_rsi_crypto_and_adapters import (
    HybridPQCAEADEngine,
    MLKEM768,
    WOTSPlusSHAKE256,
    X25519ChaCha20Poly1305Engine,
)
from .dht_rsi_verification import (
    AuthoritativeASNVerifier,
    SybilQuorumVerifier,
    ThresholdSignatureVerifier,
)

# =====================================================================
# 10. Mandatory 500-Item Golden Canary Regression Gate
# =====================================================================

def generate_golden_canary_suite(seed: int = 20261018) -> List[CanaryItem]:
    """Generates exactly 500 deterministic reference canaries covering all 8 heads and edge cases."""
    rng = random.Random(seed)
    items: List[CanaryItem] = []

    # 1. 120 Movies
    movie_templates = [
        ("Big.Buck.Bunny.{year}.1080p.BluRay.x264-OPEN", "english", False, False),
        ("Sintel.{year}.KOREAN.1080p.BluRay.x264-OPEN", "non_english", True, False),
        ("Elephants.Dream.{year}.FRENCH.720p.HDTV.x264", "non_english", False, False),
        ("Tears.of.Steel.{year}.2160p.UHD.Remux.HEVC.DV.TrueHD.Atmos", "english", True, False),
        ("Coffee.Run.{year}.1080p.WEBRip.x265.AAC5.1", "english", False, False),
        ("Spring.Open.Film.{year}.JAPANESE.1080p.BluRay.x264", "non_english", True, True),
    ]
    for i in range(120):
        tmpl, lang, subs, anime = movie_templates[i % len(movie_templates)]
        yr = 1990 + (i * 7) % 35
        title = tmpl.format(year=yr) + f".part{i+1}"
        items.append(CanaryItem(
            item_id=f"canary_mov_{i:03d}",
            title=title,
            ground_truth={
                "category": "anime" if anime else "movie",
                "adult": False,
                "anime": anime,
                "lang": lang,
                "junk": False,
                "owner_scope": "allowed_type",
                "english_subs": subs,
                "title_language": "english_title",
            },
        ))

    # 2. 100 TV Shows
    tv_templates = [
        ("Cosmos.Laundromat.S01E{ep:02d}.1080p.WEB-DL.DDP5.1.Atmos.H.264-OPEN", "english", True, False),
        ("Caminandes.S04E{ep:02d}.720p.HDTV.x264-OPEN", "english", False, False),
        ("Spring.Chronicles.S02E{ep:02d}.GERMAN.1080p.WEB-DL.DDP5.1.x264", "non_english", True, False),
        ("Open.Science.Lectures.S03E{ep:02d}.1080p.WEB-DL.DDP5.1.H.264", "english", True, False),
        ("Creative.Commons.Chronicles.S02E{ep:02d}.KOREAN.1080p.WEB-DL.DDP5.1.Atmos", "non_english", True, False),
    ]
    for i in range(100):
        tmpl, lang, subs, anime = tv_templates[i % len(tv_templates)]
        ep = (i % 12) + 1
        title = tmpl.format(ep=ep)
        items.append(CanaryItem(
            item_id=f"canary_tv_{i:03d}",
            title=title,
            ground_truth={
                "category": "tv_show",
                "adult": False,
                "anime": False,
                "lang": lang,
                "junk": False,
                "owner_scope": "allowed_type",
                "english_subs": subs,
                "title_language": "english_title",
            },
        ))

    # 3. 50 Anime
    anime_titles = [
        ("[OpenAnime] Spring - {ep:02d} [1080p][HEVC][Multiple Subtitle]", "non_english", True),
        ("[CC-Anime] Charge - Open Anime Project - {ep:02d} (1080p) [82C108B7]", "non_english", True),
        ("[OpenSub] Caminandes - {ep:02d} [720p].mkv", "non_english", True),
        ("Wing It Open Series Part 3 - {ep:02d} [1080p Dual-Audio]", "english", True),
        ("Open Blender Anime S01E{ep:02d} [1080p][Multi-Sub]", "non_english", True),
    ]
    for i in range(50):
        tmpl, lang, subs = anime_titles[i % len(anime_titles)]
        ep = (i % 24) + 1
        items.append(CanaryItem(
            item_id=f"canary_ani_{i:03d}",
            title=tmpl.format(ep=ep),
            ground_truth={
                "category": "anime",
                "adult": False,
                "anime": True,
                "lang": lang,
                "junk": False,
                "owner_scope": "allowed_type",
                "english_subs": subs,
                "title_language": "english_title",
            },
        ))

    # 4. 40 Music
    for i in range(40):
        items.append(CanaryItem(
            item_id=f"canary_mus_{i:03d}",
            title=f"Artist_{i} - Album_{i} (2024) [FLAC 24bit-96kHz]",
            ground_truth={
                "category": "music",
                "adult": False,
                "anime": False,
                "lang": "english",
                "junk": False,
                "owner_scope": "allowed_type",
                "english_subs": False,
                "title_language": "english_title",
            },
        ))

    # 5. 30 Comics/Manga
    for i in range(30):
        items.append(CanaryItem(
            item_id=f"canary_com_{i:03d}",
            title=f"Pepper.and.Carrot.Issue.{i+1}.2024.Digital.CC-BY.cbr",
            ground_truth={
                "category": "comic",
                "adult": False,
                "anime": False,
                "lang": "english",
                "junk": False,
                "owner_scope": "allowed_type",
                "english_subs": False,
                "title_language": "english_title",
            },
        ))

    # 6. 30 Audiobooks
    for i in range(30):
        items.append(CanaryItem(
            item_id=f"canary_aud_{i:03d}",
            title=f"SciFi Masterpiece Vol {i+1} - Unabridged [M4B 64kbps]",
            ground_truth={
                "category": "audiobook",
                "adult": False,
                "anime": False,
                "lang": "english",
                "junk": False,
                "owner_scope": "allowed_type",
                "english_subs": False,
                "title_language": "english_title",
            },
        ))

    # 7. 30 Ebooks
    for i in range(30):
        items.append(CanaryItem(
            item_id=f"canary_ebk_{i:03d}",
            title=f"Distributed Systems Handbook Edition {i+1}.epub",
            ground_truth={
                "category": "ebook",
                "adult": False,
                "anime": False,
                "lang": "english",
                "junk": False,
                "owner_scope": "allowed_type",
                "english_subs": False,
                "title_language": "english_title",
            },
        ))

    # 8. 30 Games / Software
    for i in range(30):
        is_game = (i % 2 == 0)
        items.append(CanaryItem(
            item_id=f"canary_sw_{i:03d}",
            title=f"OpenSource.Toolchain.v{i+1}.Linux.x86_64.iso" if not is_game else f"IndieGame.v{i+1}.0.GOG.installer",
            ground_truth={
                "category": "game" if is_game else "software",
                "adult": False,
                "anime": False,
                "lang": "english",
                "junk": False,
                "owner_scope": "allowed_type",
                "english_subs": False,
                "title_language": "english_title",
            },
        ))

    # 9. 20 Courses
    for i in range(20):
        items.append(CanaryItem(
            item_id=f"canary_crs_{i:03d}",
            title=f"OpenCourseWare - Advanced Machine Learning Foundations 2024 Course Part {i+1}",
            ground_truth={
                "category": "course",
                "adult": False,
                "anime": False,
                "lang": "english",
                "junk": False,
                "owner_scope": "allowed_type",
                "english_subs": True,
                "title_language": "english_title",
            },
        ))

    # 10. 25 Junk / Malware Traps (Critical safety gates!)
    malware_names = [
        "OpenMovie.2024.1080p.WEBRip.exe",
        "Unsigned_Trojan_Installer_v2.4.vbs",
        "Codec_Update_Required_To_Play.scr",
        "PublicArchive.2024.password_in_description.rar",
        "Big.Buck.Bunny.2024.CAM.1080p.download_tool.bat",
    ]
    for i in range(25):
        _base = malware_names[i % len(malware_names)]
        _dot = _base.rfind(".")
        name = f"{_base[:_dot]}_sample{i}{_base[_dot:]}" if _dot != -1 else f"{_base}_sample{i}"
        items.append(CanaryItem(
            item_id=f"canary_jnk_{i:03d}",
            title=name,
            ground_truth={
                "category": "unknown",
                "adult": False,
                "anime": False,
                "lang": "multi_or_unknown",
                "junk": True,  # MALWARE / JUNK TRAP!
                "owner_scope": "excluded",
                "english_subs": False,
                "title_language": "english_title",
            },
            is_malware_trap=True,
        ))

    # 11. 25 Adult / XXX Boundary Items
    for i in range(25):
        items.append(CanaryItem(
            item_id=f"canary_xxx_{i:03d}",
            title=f"Adult_Studio.24.10.18.Scene{i+1}.XXX.1080p.MP4-ARC",
            ground_truth={
                "category": "xxx",
                "adult": True,  # ADULT TRAP!
                "anime": False,
                "lang": "english",
                "junk": False,
                "owner_scope": "excluded",
                "english_subs": False,
                "title_language": "english_title",
            },
            is_adult_boundary=True,
        ))

    assert len(items) == 500, f"Expected 500 items, got {len(items)}"
    return items


class GoldenCanaryRegressionGate:
    """Mandatory 500-item golden canary regression gate.

    Guarantees that no candidate model, LoRA adapter, or calibration update causes
    safety degradation on Mondrian conformal thresholds or classification accuracy.
    """

    def __init__(self, canary_items: Optional[List[CanaryItem]] = None) -> None:
        self.canary_items = canary_items or generate_golden_canary_suite()

    def evaluate(
        self,
        predictor_fn: Callable[[CanaryItem], Dict[str, Any]],
        baseline_metrics: Optional[Dict[str, float]] = None,
        min_junk_balanced_acc: float = 0.999,
        min_adult_acc: float = 0.985,
        min_category_acc: float = 0.950,
        max_category_regression: float = 0.005,
    ) -> CanaryRegressionReport:
        """Run all 500 canary items against candidate predictor and evaluate safety gates."""
        total = len(self.canary_items)
        clean_correct = 0
        clean_total = 0
        junk_correct = 0
        junk_total = 0
        junk_false_negatives = 0

        adult_correct = 0
        category_correct = 0
        owner_scope_correct = 0

        regressions: List[Dict[str, Any]] = []

        for item in self.canary_items:
            pred = predictor_fn(item)
            gt = item.ground_truth

            # 1. Junk / Malware check (including pre-CSAM neural_junk check when exposed)
            pred_junk = bool(pred.get("junk", False))
            if "neural_junk" in pred and not bool(pred["neural_junk"]) and bool(gt["junk"]):
                pred_junk = False
            gt_junk = bool(gt["junk"])
            if gt_junk:
                junk_total += 1
                if pred_junk:
                    junk_correct += 1
                else:
                    junk_false_negatives += 1
                    regressions.append({
                        "item_id": item.item_id,
                        "title": item.title,
                        "type": "MALWARE_FALSE_NEGATIVE_CLEAN",
                    })
            else:
                clean_total += 1
                if not pred_junk:
                    clean_correct += 1

            # 2. Adult check
            pred_adult = bool(pred.get("adult", False))
            if pred_adult == bool(gt["adult"]):
                adult_correct += 1
            elif item.is_adult_boundary:
                regressions.append({
                    "item_id": item.item_id,
                    "title": item.title,
                    "type": "ADULT_BOUNDARY_MISCLASSIFICATION",
                })

            # 3. Category check
            pred_cat = str(pred.get("category", "")).lower()
            if pred_cat == str(gt["category"]).lower():
                category_correct += 1

            # 4. Scope check
            pred_scope = str(pred.get("owner_scope", "")).lower()
            if pred_scope == str(gt["owner_scope"]).lower():
                owner_scope_correct += 1

        # Probe predictor_fn on a non-regex neural junk canary so CSAMGuard title regexes cannot mask a dead neural junk head
        try:
            neural_probe = CanaryItem(
                item_id="canary_jnk_neural_probe",
                title="Unplayable_Corrupt_Truncated_Stream_Payload_2024",
                ground_truth={
                    "category": "unknown",
                    "adult": False,
                    "anime": False,
                    "lang": "multi_or_unknown",
                    "junk": True,
                    "owner_scope": "excluded",
                    "english_subs": False,
                    "title_language": "english_title",
                },
                is_malware_trap=True,
            )
            probe_pred = predictor_fn(neural_probe)
            if isinstance(probe_pred, dict) and not bool(probe_pred.get("junk", False)):
                junk_total += 1
                junk_false_negatives += 1
                regressions.append({
                    "item_id": neural_probe.item_id,
                    "title": neural_probe.title,
                    "type": "NEURAL_JUNK_HEAD_MASKED_BY_CSAM_GUARD",
                })
        except KeyError:
            pass

        tpr_junk = junk_correct / junk_total if junk_total > 0 else 1.0
        tnr_junk = clean_correct / clean_total if clean_total > 0 else 1.0
        junk_bal_acc = (tpr_junk + tnr_junk) / 2.0

        adult_acc = adult_correct / total
        category_acc = category_correct / total
        owner_scope_acc = owner_scope_correct / total

        # Check critical safety gates
        if junk_false_negatives > 0:
            return CanaryRegressionReport(
                passed=False,
                reason=f"CRITICAL_SAFETY_FAIL: {junk_false_negatives} malware traps passed as clean",
                total_items=total,
                junk_balanced_accuracy=junk_bal_acc,
                junk_false_negatives=junk_false_negatives,
                adult_accuracy=adult_acc,
                category_accuracy=category_acc,
                owner_scope_accuracy=owner_scope_acc,
                regressions=regressions,
            )

        if junk_bal_acc < min_junk_balanced_acc:
            return CanaryRegressionReport(
                passed=False,
                reason=f"JUNK_ACCURACY_REGRESSION: {junk_bal_acc:.4f} < {min_junk_balanced_acc:.4f}",
                total_items=total,
                junk_balanced_accuracy=junk_bal_acc,
                junk_false_negatives=junk_false_negatives,
                adult_accuracy=adult_acc,
                category_accuracy=category_acc,
                owner_scope_accuracy=owner_scope_acc,
                regressions=regressions,
            )

        if adult_acc < min_adult_acc:
            return CanaryRegressionReport(
                passed=False,
                reason=f"ADULT_ACCURACY_REGRESSION: {adult_acc:.4f} < {min_adult_acc:.4f}",
                total_items=total,
                junk_balanced_accuracy=junk_bal_acc,
                junk_false_negatives=junk_false_negatives,
                adult_accuracy=adult_acc,
                category_accuracy=category_acc,
                owner_scope_accuracy=owner_scope_acc,
                regressions=regressions,
            )

        if category_acc < min_category_acc:
            return CanaryRegressionReport(
                passed=False,
                reason=f"CATEGORY_ACCURACY_REGRESSION: {category_acc:.4f} < {min_category_acc:.4f}",
                total_items=total,
                junk_balanced_accuracy=junk_bal_acc,
                junk_false_negatives=junk_false_negatives,
                adult_accuracy=adult_acc,
                category_accuracy=category_acc,
                owner_scope_accuracy=owner_scope_acc,
                regressions=regressions,
            )

        if owner_scope_acc < 0.985:
            return CanaryRegressionReport(
                passed=False,
                reason=f"OWNER_SCOPE_ACCURACY_REGRESSION: {owner_scope_acc:.4f} < 0.9850",
                total_items=total,
                junk_balanced_accuracy=junk_bal_acc,
                junk_false_negatives=junk_false_negatives,
                adult_accuracy=adult_acc,
                category_accuracy=category_acc,
                owner_scope_accuracy=owner_scope_acc,
                regressions=regressions,
            )

        if baseline_metrics and "category_accuracy" in baseline_metrics:
            base_cat = baseline_metrics["category_accuracy"]
            if category_acc < (base_cat - max_category_regression):
                return CanaryRegressionReport(
                    passed=False,
                    reason=f"CATEGORY_REGRESSION_DELTA: {category_acc:.4f} < baseline {base_cat:.4f} - {max_category_regression}",
                    total_items=total,
                    junk_balanced_accuracy=junk_bal_acc,
                    junk_false_negatives=junk_false_negatives,
                    adult_accuracy=adult_acc,
                    category_accuracy=category_acc,
                    owner_scope_accuracy=owner_scope_acc,
                    regressions=regressions,
                )

        return CanaryRegressionReport(
            passed=True,
            reason="all_500_golden_canary_gates_passed",
            total_items=total,
            junk_balanced_accuracy=junk_bal_acc,
            junk_false_negatives=junk_false_negatives,
            adult_accuracy=adult_acc,
            category_accuracy=category_acc,
            owner_scope_accuracy=owner_scope_acc,
            regressions=regressions,
        )


# =====================================================================
# 10b. Byzantine-Resilient Aggregator (Multi-ASN k-Anonymity & Trimmed Mean/Median)
# =====================================================================

class ByzantineResilientAggregator:
    """Byzantine-resilient DHT telemetry and calibration aggregator.

    Ingests encrypted `EncryptedTelemetryBundle` submissions from DHT peers, decrypts them via
    `HybridPQCAEADEngine`, verifies authoritative ASN bindings (`AuthoritativeASNVerifier`), and provides:
      1. `aggregate_catalog_observations`: Multi-ASN k-Anonymity consensus requiring
         `min_distinct_asns >= 5`, `min_observations >= 10`, `min_agreement_ratio >= 0.90`, and
         consistent physical `ffprobe` stream/container signatures before emitting a verified
         `ZeroCueCatalogDelta` or release-group prior delta and signed BEP-46 update.
      2. `aggregate_continuous_deltas`: Per-ASN grouped (1 vote per ASN) coordinate-wise trimmed
         mean (`trim_fraction=0.20`) and coordinate-wise median so up to 35% Byzantine malicious
         ASNs sending extreme weight/temperature/logit poisoning values are mathematically clipped out.
    """

    VIDEO_SYNTAX_TOKENS = {
        "2160p", "4k", "1080p", "720p", "480p", "uhd",
        "bluray", "remux", "webrip", "webdl", "hdtv", "bdrip",
        "hevc", "x265", "h265", "x264", "h264", "av1", "xvid",
        "truehd", "atmos", "dts", "dtshd",
    }
    VALID_VIDEO_CODECS = {
        "hevc", "h265", "x265", "h264", "x264", "av1", "vp9", "mpeg2video", "vc1", "prores", "xvid",
    }
    VALID_AUDIO_CODECS = {
        "truehd", "eac3", "ac3", "dts", "dtshd", "flac", "aac", "mp3", "m4b", "opus", "vorbis", "alac", "pcm", "wav",
    }
    VIDEO_CATEGORIES = {"movie", "tv_show", "anime", "xxx", "course"}
    AUDIO_ONLY_CATEGORIES = {"music", "audiobook"}
    PRINT_CATEGORIES = {"ebook", "comic"}

    def __init__(
        self,
        aggregator_privkey_bytes: Optional[bytes] = None,
        asn_verifier: Optional[AuthoritativeASNVerifier] = None,
        min_distinct_asns: int = 5,
        min_observations: int = 10,
        min_agreement_ratio: float = 0.90,
        trim_fraction: float = 0.20,
        federation_signers: Optional[List[Tuple[ed25519.Ed25519PrivateKey, str, bytes, str]]] = None,
        threshold_m: int = 2,
        current_seq: int = 100,
    ) -> None:
        if aggregator_privkey_bytes is None:
            aggregator_privkey_bytes, self.aggregator_pubkey_bytes = HybridPQCAEADEngine.generate_keypair()
        else:
            self.aggregator_pubkey_bytes = b""
        self.aggregator_privkey_bytes = aggregator_privkey_bytes
        self.asn_verifier = asn_verifier or AuthoritativeASNVerifier()
        self.min_distinct_asns = max(int(min_distinct_asns), 1)
        self.min_observations = max(int(min_observations), 1)
        self.min_agreement_ratio = float(min_agreement_ratio)
        self.trim_fraction = float(trim_fraction)
        self.federation_signers = federation_signers or [
            ThresholdSignatureVerifier.generate_hybrid_signer() for _ in range(max(int(threshold_m), 2))
        ]
        self.threshold_m = max(int(threshold_m), 1)
        self.current_seq = int(current_seq)
        self._ingested_submissions: List[TelemetrySubmission] = []

    @property
    def trusted_ed_signers(self) -> Dict[str, str]:
        return {s[1]: f"federation_signer_{i}" for i, s in enumerate(self.federation_signers)}

    @property
    def trusted_pq_signers(self) -> Dict[str, str]:
        return {s[3]: f"federation_pq_signer_{i}" for i, s in enumerate(self.federation_signers)}

    def ingest_submission(self, submission: Union[TelemetrySubmission, Dict[str, Any]]) -> None:
        """Queue an encrypted telemetry submission for batch catalog aggregation."""
        self._ingested_submissions.append(self._coerce_telemetry_submission(submission))

    @staticmethod
    def _coerce_telemetry_submission(
        item: Union[TelemetrySubmission, Dict[str, Any], Tuple[EncryptedTelemetryBundle, PeerAttestation]]
    ) -> TelemetrySubmission:
        if isinstance(item, TelemetrySubmission):
            if not isinstance(item.encrypted_bundle, EncryptedTelemetryBundle):
                raise ValueError("invalid_encrypted_bundle_type")
            if item.attestation is not None:
                if not isinstance(item.attestation, PeerAttestation):
                    raise ValueError("invalid_attestation_type")
                if isinstance(item.attestation.asn, bool) or not isinstance(item.attestation.asn, int):
                    raise ValueError("invalid_attestation_asn_type")
                if isinstance(item.attestation.timestamp, bool) or not isinstance(item.attestation.timestamp, int):
                    raise ValueError("invalid_attestation_timestamp_type")
            if isinstance(item.asn, bool) or not isinstance(item.asn, int):
                raise ValueError("invalid_asn_type")
            return item
        if isinstance(item, tuple) and len(item) == 2:
            bundle, att = item
            if not isinstance(bundle, EncryptedTelemetryBundle) or not isinstance(att, PeerAttestation):
                raise ValueError("invalid_submission_tuple_types")
            if isinstance(att.asn, bool) or not isinstance(att.asn, int):
                raise ValueError("invalid_attestation_asn_type")
            if isinstance(att.timestamp, bool) or not isinstance(att.timestamp, int):
                raise ValueError("invalid_attestation_timestamp_type")
            return TelemetrySubmission(
                encrypted_bundle=bundle,
                asn=int(att.asn),
                peer_id=str(att.peer_id),
                peer_pubkey_hex=str(att.peer_pubkey_hex),
                ip_address=att.ip_address,
                attestation=att,
            )
        if isinstance(item, dict):
            if "encrypted_bundle" not in item:
                raise ValueError("missing_encrypted_bundle")
            bundle = item["encrypted_bundle"]
            if isinstance(bundle, dict):
                bundle = EncryptedTelemetryBundle.from_dict(bundle)
            if not isinstance(bundle, EncryptedTelemetryBundle):
                raise ValueError("invalid_encrypted_bundle_type")
            att = item.get("attestation")
            if isinstance(att, dict):
                raw_att_asn = att.get("asn")
                raw_att_ts = att.get("timestamp")
                if isinstance(raw_att_asn, bool) or not isinstance(raw_att_asn, int):
                    raise ValueError("invalid_attestation_asn_type")
                if isinstance(raw_att_ts, bool) or not isinstance(raw_att_ts, int):
                    raise ValueError("invalid_attestation_timestamp_type")
                att = PeerAttestation(
                    peer_id=str(att["peer_id"]),
                    asn=int(raw_att_asn),
                    peer_pubkey_hex=str(att["peer_pubkey_hex"]),
                    timestamp=int(raw_att_ts),
                    signature_hex=str(att["signature_hex"]),
                    ip_address=att.get("ip_address"),
                    pq_pubkey_hex=str(att.get("pq_pubkey_hex", "")),
                    pq_signature_hex=str(att.get("pq_signature_hex", "")),
                )
            elif att is not None:
                if not isinstance(att, PeerAttestation):
                    raise ValueError("invalid_attestation_type")
                if isinstance(att.asn, bool) or not isinstance(att.asn, int):
                    raise ValueError("invalid_attestation_asn_type")
                if isinstance(att.timestamp, bool) or not isinstance(att.timestamp, int):
                    raise ValueError("invalid_attestation_timestamp_type")
            raw_asn = item.get("asn", att.asn if att else None)
            if raw_asn is None or isinstance(raw_asn, bool) or not isinstance(raw_asn, int):
                raise ValueError("missing_or_invalid_asn")
            return TelemetrySubmission(
                encrypted_bundle=bundle,
                asn=int(raw_asn),
                peer_id=str(item.get("peer_id", att.peer_id if att else "")),
                peer_pubkey_hex=str(item.get("peer_pubkey_hex", att.peer_pubkey_hex if att else "")),
                ip_address=item.get("ip_address", att.ip_address if att else None),
                attestation=att,
                canonical_title=item.get("canonical_title"),
            )
        raise TypeError(f"Unsupported telemetry submission type: {type(item)}")

    @classmethod
    def verify_physical_stream_signature(
        cls,
        record_dict: Dict[str, Any],
        canonical_title: Optional[str] = None,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Validate that a decrypted telemetry record carries a consistent physical ffprobe signature."""
        if not isinstance(record_dict, dict):
            return False, "malformed_record_not_dict", {}
        vf = record_dict.get("verified_features")
        if not isinstance(vf, dict) or not vf:
            return False, "missing_verified_ffprobe_features", {}

        category = str(record_dict.get("category", "")).strip().lower()
        junk = bool(record_dict.get("junk", False))
        owner_scope = str(record_dict.get("owner_scope", "")).strip().lower()

        raw_size = record_dict.get("dp_size_mb", 0.0)
        if isinstance(raw_size, bool):
            return False, "malformed_numeric_field:dp_size_mb", {}
        try:
            dp_size_mb = float(0.0 if raw_size is None else raw_size)
        except (TypeError, ValueError):
            return False, "malformed_numeric_field:dp_size_mb", {}
        if not math.isfinite(dp_size_mb) or dp_size_mb < 0.0:
            return False, "malformed_numeric_field:dp_size_mb", {}

        dp_dur = record_dict.get("dp_duration_mins")
        dp_dur_float: Optional[float] = None
        if dp_dur is not None:
            if isinstance(dp_dur, bool):
                return False, "malformed_numeric_field:dp_duration_mins", {}
            try:
                dp_dur_float = float(dp_dur)
            except (TypeError, ValueError):
                return False, "malformed_numeric_field:dp_duration_mins", {}
            if not math.isfinite(dp_dur_float):
                return False, "malformed_numeric_field:dp_duration_mins", {}

        raw_tokens = record_dict.get("token_features", [])
        if raw_tokens is None:
            raw_tokens = []
        if not isinstance(raw_tokens, (list, tuple, set)):
            return False, "malformed_token_features", {}
        token_features = {str(t).lower() for t in raw_tokens}
        if canonical_title:
            title_tokens = {t.lower() for t in re.findall(r"[a-zA-Z0-9]+", canonical_title)}
            token_features = token_features | (title_tokens & cls.VIDEO_SYNTAX_TOKENS)

        has_video = bool(vf.get("has_video", False))
        has_audio = bool(vf.get("has_audio", False))
        vcodec = str(vf.get("video_codec", "none") or "none").strip().lower()
        acodec = str(vf.get("audio_codec", "none") or "none").strip().lower()
        raw_achannels = vf.get("audio_channels", 0)
        if isinstance(raw_achannels, bool):
            return False, "malformed_numeric_field:audio_channels", {}
        try:
            achannels = float(0.0 if raw_achannels is None else raw_achannels)
        except (TypeError, ValueError):
            return False, "malformed_numeric_field:audio_channels", {}
        if not math.isfinite(achannels) or achannels < 0.0:
            return False, "malformed_numeric_field:audio_channels", {}
        container = str(vf.get("container_format", vf.get("format_name", "matroska")) or "matroska").strip().lower()
        has_video_cues = bool(token_features & cls.VIDEO_SYNTAX_TOKENS)

        # 1. Reject junk poisoning on catalog items or contradicting playable video streams/tokens
        if junk or category == "unknown" or owner_scope == "excluded":
            if has_video and vcodec in cls.VALID_VIDEO_CODECS and dp_size_mb >= 50.0:
                return False, "stream_signature_mismatch:junk_claimed_on_valid_video_stream", {}
            if has_video_cues:
                return False, "stream_signature_mismatch:junk_claimed_on_uhd_or_bluray_video_release", {}
            if not has_video and not has_audio:
                return False, "stream_signature_missing:unverified_junk_without_ffprobe_streams", {}
            return False, "catalog_poison_guard:junk_excluded_from_zero_cue_catalog", {}

        # 2. Reject audio-only / print categories (e.g. audiobook, ebook) on video streams or video release tokens
        if category in cls.AUDIO_ONLY_CATEGORIES or category in cls.PRINT_CATEGORIES:
            if has_video or (vcodec not in ("none", "")):
                return False, f"stream_signature_mismatch:{category}_cannot_contain_video_stream:{vcodec}", {}
            if has_video_cues:
                return False, f"stream_signature_mismatch:{category}_contradicts_video_tokens", {}
            if category in cls.AUDIO_ONLY_CATEGORIES:
                if not has_audio or acodec not in cls.VALID_AUDIO_CODECS:
                    return False, f"stream_signature_mismatch:{category}_missing_valid_audio_codec:{acodec}", {}
                if category == "audiobook" and achannels > 2.1:
                    return False, f"stream_signature_mismatch:audiobook_invalid_surround_channels:{achannels}", {}

        # 3. Require valid physical ffprobe video stream for video categories (movie, tv_show, anime, etc.)
        if category in cls.VIDEO_CATEGORIES:
            if not has_video or vcodec not in cls.VALID_VIDEO_CODECS:
                return False, f"stream_signature_mismatch:{category}_missing_valid_video_codec:{vcodec}", {}
            if not has_audio or acodec not in cls.VALID_AUDIO_CODECS:
                return False, f"stream_signature_mismatch:{category}_missing_valid_audio_codec:{acodec}", {}
            if dp_size_mb < 10.0:
                return False, f"stream_signature_mismatch:{category}_implausible_size_mb:{dp_size_mb}", {}
            if dp_dur_float is not None and dp_dur_float <= 0.0:
                return False, f"stream_signature_mismatch:{category}_non_positive_duration", {}

        sig = {
            "has_video": has_video,
            "has_audio": has_audio,
            "video_codec": vcodec,
            "audio_codec": acodec,
            "audio_channels": achannels,
            "container_format": container,
        }
        return True, "valid_physical_stream_signature", sig

    def _sign_bep46_bundle(self, bundle_bytes: bytes, min_asn_quorum: int) -> ThresholdDHTManifest:
        self.current_seq += 1
        bundle_sha256 = hashlib.sha256(bundle_bytes).hexdigest()
        infohash = hashlib.sha1(f"armiger-bep46-seq-{self.current_seq}:{bundle_sha256}".encode("utf-8")).hexdigest()
        ts = int(time.time())
        temp_manifest = ThresholdDHTManifest(
            seq=self.current_seq,
            version="2.1.0",
            infohash=infohash,
            bundle_sha256=bundle_sha256,
            bundle_format="json",
            min_asn_quorum=min_asn_quorum,
            timestamp=ts,
            signatures={},
        )
        payload = temp_manifest.canonical_payload()
        ed_sigs: Dict[str, str] = {}
        pq_sigs: Dict[str, str] = {}
        for ed_priv, ed_pub_hex, pq_priv, _pq_pub_hex in self.federation_signers[: self.threshold_m]:
            ed_sigs[ed_pub_hex] = ed_priv.sign(payload).hex()
            leaf_pub, pq_sig_bytes, _, _ = WOTSPlusSHAKE256.allocate_and_sign(
                pq_priv, payload, reserve_fresh=True
            )
            pq_sigs[leaf_pub.hex()] = pq_sig_bytes.hex()

        return ThresholdDHTManifest(
            seq=self.current_seq,
            version="2.1.0",
            infohash=infohash,
            bundle_sha256=bundle_sha256,
            bundle_format="json",
            min_asn_quorum=min_asn_quorum,
            timestamp=ts,
            signatures=ed_sigs,
            pq_signatures=pq_sigs,
        )

    def aggregate_catalog_observations(
        self,
        submissions: Optional[Sequence[Union[TelemetrySubmission, Dict[str, Any], Tuple[EncryptedTelemetryBundle, PeerAttestation]]]] = None,
        blinded_hash: Optional[str] = None,
        canonical_title: Optional[str] = None,
        emit_bep46_manifest: bool = True,
        require_authoritative_asn: bool = False,
        require_attestation: bool = True,
        *,
        now_ts: Optional[int] = None,
        max_age_sec: int = 7 * 86400,
        max_future_drift_sec: int = 300,
    ) -> CatalogConsensusResult:
        """Aggregate encrypted catalog observations using Multi-ASN k-Anonymity and physical ffprobe verification."""
        eff_now_ts = int(now_ts) if (now_ts is not None and not isinstance(now_ts, bool)) else int(time.time())
        raw_list = list(submissions) if submissions is not None else list(self._ingested_submissions)
        quarantined_reasons: List[str] = []
        coerced: List[TelemetrySubmission] = []
        for s in raw_list:
            try:
                coerced.append(self._coerce_telemetry_submission(s))
            except Exception as exc:
                quarantined_reasons.append(f"malformed_submission:{type(exc).__name__}:{exc}")

        authenticated_subs: List[Tuple[TelemetrySubmission, int, str, Optional[str], Optional[str], Dict[str, Any], Dict[str, Any], Optional[str]]] = []

        # Step 1: Authenticate each submission (ASN binding + PeerAttestation signature & freshness + AEAD decryption
        # + blinded_hash + wrapper consistency + physical stream signature) BEFORE updating cross-submission spoofing
        # sets or per-key/per-IP quotas so unauthenticated or invalid wrappers cannot poison or exhaust quotas.
        for sub in coerced:
            try:
                claimed_asn = int(sub.asn)
                norm_pk = AuthoritativeASNVerifier.normalize_pubkey(sub.peer_pubkey_hex) if sub.peer_pubkey_hex else ""
            except Exception as exc:
                quarantined_reasons.append(f"malformed_submission:{type(exc).__name__}:{exc}")
                continue
            subnet: Optional[str] = None
            norm_ip: Optional[str] = None
            wrapper_ip: Optional[str] = None

            if sub.ip_address is not None and str(sub.ip_address).strip() != "":
                try:
                    wrapper_ip = str(ipaddress.ip_address(str(sub.ip_address).strip()))
                except ValueError:
                    quarantined_reasons.append(f"invalid_ip_address:{sub.ip_address}")
                    continue

            if sub.attestation is None:
                if require_attestation:
                    quarantined_reasons.append("missing_peer_attestation")
                    continue
                norm_ip = wrapper_ip
            else:
                att = sub.attestation
                if isinstance(att.timestamp, bool) or not isinstance(att.timestamp, int) or att.timestamp <= 0:
                    quarantined_reasons.append(f"invalid_attestation_timestamp:{att.timestamp!r}")
                    continue
                if (eff_now_ts - att.timestamp) > max_age_sec:
                    quarantined_reasons.append(f"expired_attestation_timestamp:{att.timestamp}")
                    continue
                if (att.timestamp - eff_now_ts) > max_future_drift_sec:
                    quarantined_reasons.append(f"future_attestation_timestamp:{att.timestamp}")
                    continue
                try:
                    att_pk = AuthoritativeASNVerifier.normalize_pubkey(att.peer_pubkey_hex)
                    if isinstance(att.asn, bool) or not isinstance(att.asn, int):
                        raise ValueError("invalid_attestation_asn_type")
                    att_asn = int(att.asn)
                except Exception as exc:
                    quarantined_reasons.append(f"malformed_attestation:{exc}")
                    continue
                if att_asn != claimed_asn:
                    quarantined_reasons.append(f"attestation_asn_mismatch:{att.asn}!={claimed_asn}")
                    continue
                if norm_pk and att_pk != norm_pk:
                    quarantined_reasons.append("attestation_pubkey_mismatch")
                    continue
                if not norm_pk:
                    norm_pk = att_pk
                att_ip: Optional[str] = None
                if att.ip_address is not None and str(att.ip_address).strip() != "":
                    try:
                        att_ip = str(ipaddress.ip_address(str(att.ip_address).strip()))
                    except ValueError:
                        quarantined_reasons.append(f"invalid_attestation_ip_address:{att.ip_address}")
                        continue
                if wrapper_ip is not None and att_ip is None:
                    quarantined_reasons.append("unauthenticated_wrapper_ip_without_attestation_ip")
                    continue
                if wrapper_ip is not None and att_ip is not None and wrapper_ip != att_ip:
                    quarantined_reasons.append(f"attestation_ip_mismatch:{wrapper_ip}!={att_ip}")
                    continue
                norm_ip = att_ip if att_ip is not None else wrapper_ip
                target_h = hashlib.sha256(sub.encrypted_bundle.ciphertext_hex.encode("utf-8")).hexdigest()
                try:
                    pk_bytes = bytes.fromhex(att_pk)
                    sig_bytes = bytes.fromhex(str(att.signature_hex).strip())
                    ed25519.Ed25519PublicKey.from_public_bytes(pk_bytes).verify(
                        sig_bytes, att.canonical_bytes(target_h)
                    )
                except Exception:
                    quarantined_reasons.append("invalid_peer_attestation_signature")
                    continue

            if norm_ip is not None:
                try:
                    subnet = AuthoritativeASNVerifier.extract_subnet_prefix(norm_ip)
                except ValueError:
                    quarantined_reasons.append(f"invalid_ip_address:{norm_ip}")
                    continue

            bind_ok, bind_reason = self.asn_verifier.verify_binding(
                claimed_asn=claimed_asn,
                pubkey_hex=norm_pk or None,
                ip_address=norm_ip,
                require_authoritative=require_authoritative_asn,
            )
            if not bind_ok:
                quarantined_reasons.append(f"asn_binding_failed:{bind_reason}")
                continue

            # Decrypt EncryptedTelemetryBundle via HybridPQCAEADEngine
            try:
                plaintext = HybridPQCAEADEngine.decrypt_bundle(
                    sub.encrypted_bundle, self.aggregator_privkey_bytes
                )
                rec_dict = json.loads(plaintext.decode("utf-8"))
            except Exception as e:
                quarantined_reasons.append(f"aead_decrypt_failed:{e}")
                continue

            if not isinstance(rec_dict, dict):
                quarantined_reasons.append("malformed_record_not_dict")
                continue

            # Validate target hash, wrapper title consistency, and physical stream signature
            # BEFORE mutating deduplication sets or observation quotas
            rec_hash = str(rec_dict.get("blinded_hash", "")).strip()
            if not rec_hash:
                quarantined_reasons.append("missing_blinded_hash")
                continue
            if blinded_hash is not None and rec_hash != blinded_hash:
                quarantined_reasons.append(f"mismatched_blinded_hash:{rec_hash[:16]}!={blinded_hash[:16]}")
                continue

            rec_title_raw = rec_dict.get("canonical_title")
            if rec_title_raw is not None and not isinstance(rec_title_raw, str):
                quarantined_reasons.append("malformed_canonical_title_type")
                continue
            auth_peer_title: Optional[str] = (
                rec_title_raw.strip() if isinstance(rec_title_raw, str) and rec_title_raw.strip() else None
            )
            wrap_title: Optional[str] = (
                str(sub.canonical_title).strip()
                if sub.canonical_title is not None and str(sub.canonical_title).strip() != ""
                else None
            )
            if wrap_title is not None and auth_peer_title is not None and wrap_title != auth_peer_title:
                quarantined_reasons.append("unauthenticated_wrapper_canonical_title_mismatch")
                continue

            eff_title_for_sig = auth_peer_title or canonical_title
            sig_ok, sig_reason, stream_sig = self.verify_physical_stream_signature(
                rec_dict, canonical_title=eff_title_for_sig
            )
            if not sig_ok:
                quarantined_reasons.append(sig_reason)
                continue

            authenticated_subs.append(
                (sub, claimed_asn, norm_pk, subnet, norm_ip, rec_dict, stream_sig, auth_peer_title)
            )

        # Step 2: Scan fully validated authenticated submissions for cross-submission Sybil key/subnet ASN spoofing
        seen_pubkey_asn: Dict[str, int] = {}
        seen_subnet_asn: Dict[str, int] = {}
        spoofed_pubkeys: Set[str] = set()
        spoofed_subnets: Set[str] = set()
        for _sub, claimed_asn, norm_pk, subnet, _norm_ip, _rec_dict, _stream_sig, _auth_title in authenticated_subs:
            if norm_pk:
                if norm_pk in seen_pubkey_asn and seen_pubkey_asn[norm_pk] != claimed_asn:
                    spoofed_pubkeys.add(norm_pk)
                seen_pubkey_asn[norm_pk] = claimed_asn
            if subnet:
                if subnet in seen_subnet_asn and seen_subnet_asn[subnet] != claimed_asn:
                    spoofed_subnets.add(subnet)
                seen_subnet_asn[subnet] = claimed_asn

        valid_records: List[Tuple[int, Dict[str, Any], Dict[str, Any], Optional[str]]] = []
        seen_unique_subnets: Set[str] = set()
        seen_unique_pubkeys: Set[str] = set()
        ip_obs_counts: Dict[str, int] = {}

        for _sub, claimed_asn, norm_pk, subnet, norm_ip, rec_dict, stream_sig, auth_peer_title in authenticated_subs:
            if norm_pk:
                if norm_pk in spoofed_pubkeys:
                    quarantined_reasons.append(f"sybil_pubkey_multi_asn_spoof:{norm_pk[:16]}")
                    continue
                if norm_pk in seen_unique_pubkeys:
                    quarantined_reasons.append(f"duplicate_pubkey_observation:{norm_pk[:16]}")
                    continue
                seen_unique_pubkeys.add(norm_pk)

            if subnet and norm_ip:
                if subnet in spoofed_subnets:
                    quarantined_reasons.append(f"sybil_subnet_multi_asn_spoof:{subnet}")
                    continue
                ip_obs_counts[norm_ip] = ip_obs_counts.get(norm_ip, 0) + 1
                if ip_obs_counts[norm_ip] > 3:
                    quarantined_reasons.append(f"duplicate_ip_observation_flood:{norm_ip}")
                    continue
                seen_unique_subnets.add(subnet)

            # Only authenticated titles bound inside the encrypted/signed payload count as peer votes
            valid_records.append((claimed_asn, rec_dict, stream_sig, auth_peer_title))

        if blinded_hash is not None:
            target_blinded_hash = blinded_hash
        elif valid_records:
            hash_to_asns: Dict[str, Set[int]] = {}
            hash_to_count: Dict[str, int] = {}
            for asn_val, rec_d, _, _ in valid_records:
                rh = str(rec_d.get("blinded_hash", "")).strip()
                hash_to_asns.setdefault(rh, set()).add(asn_val)
                hash_to_count[rh] = hash_to_count.get(rh, 0) + 1
            target_blinded_hash = max(
                hash_to_count.keys(),
                key=lambda h: (len(hash_to_asns[h]), hash_to_count[h]),
            )
            if len(hash_to_count) > 1:
                same_hash_records: List[Tuple[int, Dict[str, Any], Dict[str, Any], Optional[str]]] = []
                for item in valid_records:
                    rh = str(item[1].get("blinded_hash", "")).strip()
                    if rh == target_blinded_hash:
                        same_hash_records.append(item)
                    else:
                        quarantined_reasons.append(
                            f"mismatched_blinded_hash_in_batch:{rh[:16]}!={target_blinded_hash[:16]}"
                        )
                valid_records = same_hash_records
        else:
            target_blinded_hash = "unknown"

        # Cross-ASN physical stream signature quorum filter:
        # If >= min_distinct_asns ASNs agree on a physical (has_video, video_codec, audio_codec) hardware signature,
        # quarantine any minority (< min_distinct_asns) submissions that fabricated a contradictory stream signature.
        sig_to_asns: Dict[Tuple[bool, bool, str, str], Set[int]] = {}
        for asn_val, _, s_sig, _ in valid_records:
            s_key = (
                bool(s_sig["has_video"]),
                bool(s_sig["has_audio"]),
                str(s_sig["video_codec"]),
                str(s_sig["audio_codec"]),
            )
            sig_to_asns.setdefault(s_key, set()).add(asn_val)

        quorum_stream_sigs = {
            s_key for s_key, asns_set in sig_to_asns.items() if len(asns_set) >= self.min_distinct_asns
        }
        if len(quorum_stream_sigs) == 1:
            dominant_stream_sig = next(iter(quorum_stream_sigs))
            filtered_records: List[Tuple[int, Dict[str, Any], Dict[str, Any], Optional[str]]] = []
            for item in valid_records:
                asn_val, rec_d, s_sig, eff_t = item
                s_key = (
                    bool(s_sig["has_video"]),
                    bool(s_sig["has_audio"]),
                    str(s_sig["video_codec"]),
                    str(s_sig["audio_codec"]),
                )
                if s_key != dominant_stream_sig and len(sig_to_asns.get(s_key, set())) < self.min_distinct_asns:
                    quarantined_reasons.append(
                        f"cross_asn_ffprobe_signature_mismatch:{s_key[2]}/{s_key[3]}!={dominant_stream_sig[2]}/{dominant_stream_sig[3]}"
                    )
                else:
                    filtered_records.append(item)
            valid_records = filtered_records

        total_valid = len(valid_records)
        distinct_valid_asns = {item[0] for item in valid_records}

        if total_valid < self.min_observations:
            return CatalogConsensusResult(
                consensus_reached=False,
                reason=f"insufficient_observations:{total_valid}<{self.min_observations}",
                blinded_hash=target_blinded_hash,
                distinct_asns=len(distinct_valid_asns),
                total_valid_observations=total_valid,
                quarantined_byzantine_count=len(quarantined_reasons),
                quarantined_reasons=quarantined_reasons,
            )

        if len(distinct_valid_asns) < self.min_distinct_asns:
            return CatalogConsensusResult(
                consensus_reached=False,
                reason=f"insufficient_distinct_asns:{len(distinct_valid_asns)}<{self.min_distinct_asns}",
                blinded_hash=target_blinded_hash,
                distinct_asns=len(distinct_valid_asns),
                total_valid_observations=total_valid,
                quarantined_byzantine_count=len(quarantined_reasons),
                quarantined_reasons=quarantined_reasons,
            )

        # Group by composite (8-head label tuple + physical stream signature)
        label_obs_counts: Dict[Tuple[Any, ...], int] = {}
        label_asn_sets: Dict[Tuple[Any, ...], Set[int]] = {}
        label_exemplar: Dict[Tuple[Any, ...], Tuple[Dict[str, Any], Dict[str, Any], Optional[str]]] = {}
        label_title_asns: Dict[Tuple[Any, ...], Dict[str, Set[int]]] = {}

        # Also track per-ASN majority vote so a single ASN flooding observations only gets 1 ASN vote
        asn_label_votes: Dict[int, Dict[Tuple[Any, ...], int]] = {}

        for asn_val, rec_d, s_sig, eff_t in valid_records:
            composite_key = (
                str(rec_d.get("category", "")).lower(),
                bool(rec_d.get("adult", False)),
                bool(rec_d.get("anime", False)),
                str(rec_d.get("lang", "english")).lower(),
                bool(rec_d.get("junk", False)),
                str(rec_d.get("owner_scope", "allowed_type")).lower(),
                bool(rec_d.get("english_subs", False)),
                str(rec_d.get("title_language", "english_title")).lower(),
                bool(s_sig["has_video"]),
                bool(s_sig["has_audio"]),
                str(s_sig["video_codec"]),
                str(s_sig["audio_codec"]),
            )
            label_obs_counts[composite_key] = label_obs_counts.get(composite_key, 0) + 1
            label_asn_sets.setdefault(composite_key, set()).add(asn_val)
            label_exemplar[composite_key] = (rec_d, s_sig, eff_t)
            if eff_t is not None and str(eff_t).strip() != "":
                norm_title_vote = str(eff_t).strip()
                label_title_asns.setdefault(composite_key, {}).setdefault(norm_title_vote, set()).add(asn_val)
            asn_votes_for_this = asn_label_votes.setdefault(asn_val, {})
            asn_votes_for_this[composite_key] = asn_votes_for_this.get(composite_key, 0) + 1

        # Determine each ASN's single representative vote (1 vote per ASN)
        per_asn_winner_counts: Dict[Tuple[Any, ...], int] = {}
        for asn_val, c_map in asn_label_votes.items():
            asn_choice = max(c_map.items(), key=lambda kv: kv[1])[0]
            per_asn_winner_counts[asn_choice] = per_asn_winner_counts.get(asn_choice, 0) + 1

        # Best candidate must maximize distinct ASNs first, then observation count
        best_key = max(
            label_obs_counts.keys(),
            key=lambda k: (per_asn_winner_counts.get(k, 0), len(label_asn_sets[k]), label_obs_counts[k]),
        )
        winning_obs = label_obs_counts[best_key]
        winning_asns = label_asn_sets[best_key]
        winning_asn_votes = per_asn_winner_counts.get(best_key, 0)

        obs_agreement = winning_obs / float(total_valid)
        asn_agreement = winning_asn_votes / float(len(distinct_valid_asns))
        agreement_ratio = min(obs_agreement, asn_agreement)

        if len(winning_asns) < self.min_distinct_asns:
            return CatalogConsensusResult(
                consensus_reached=False,
                reason=f"insufficient_winning_label_asns:{len(winning_asns)}<{self.min_distinct_asns}",
                blinded_hash=target_blinded_hash,
                distinct_asns=len(winning_asns),
                total_valid_observations=total_valid,
                quarantined_byzantine_count=len(quarantined_reasons),
                quarantined_reasons=quarantined_reasons,
                agreement_ratio=agreement_ratio,
            )

        if agreement_ratio < self.min_agreement_ratio:
            return CatalogConsensusResult(
                consensus_reached=False,
                reason=f"insufficient_agreement_ratio:{agreement_ratio:.4f}<{self.min_agreement_ratio:.4f}",
                blinded_hash=target_blinded_hash,
                distinct_asns=len(winning_asns),
                total_valid_observations=total_valid,
                quarantined_byzantine_count=len(quarantined_reasons),
                quarantined_reasons=quarantined_reasons,
                agreement_ratio=agreement_ratio,
            )

        exemplar_rec, exemplar_sig, _exemplar_title = label_exemplar[best_key]
        verified_title: Optional[str] = canonical_title
        if verified_title is None:
            title_votes = label_title_asns.get(best_key, {})
            if title_votes:
                top_title, top_title_asns = max(title_votes.items(), key=lambda kv: len(kv[1]))
                if (
                    len(top_title_asns) >= self.min_distinct_asns
                    and (len(top_title_asns) / float(len(winning_asns))) >= self.min_agreement_ratio
                ):
                    verified_title = top_title

        prior_delta = {
            f"category_prior:{best_key[0]}": round(agreement_ratio, 4),
            "junk_rate_delta": 0.0,
            "verified_asn_quorum": float(len(winning_asns)),
        }
        delta = ZeroCueCatalogDelta(
            blinded_hash=str(exemplar_rec.get("blinded_hash", target_blinded_hash)),
            canonical_title=verified_title,
            category=best_key[0],
            adult=best_key[1],
            anime=best_key[2],
            lang=best_key[3],
            junk=best_key[4],
            owner_scope=best_key[5],
            english_subs=best_key[6],
            title_language=best_key[7],
            stream_signature=exemplar_sig,
            distinct_asn_count=len(winning_asns),
            verified_asns=sorted(winning_asns),
            observation_count=winning_obs,
            agreement_ratio=agreement_ratio,
            release_group_prior_delta=prior_delta,
            timestamp=int(time.time()),
        )

        manifest = None
        bundle_bytes = None
        if emit_bep46_manifest:
            bundle_bytes = delta.canonical_json_bytes()
            manifest = self._sign_bep46_bundle(bundle_bytes, min_asn_quorum=len(winning_asns))

        return CatalogConsensusResult(
            consensus_reached=True,
            reason="k_anonymity_multi_asn_consensus_verified",
            blinded_hash=delta.blinded_hash,
            delta=delta,
            distinct_asns=len(winning_asns),
            total_valid_observations=total_valid,
            quarantined_byzantine_count=len(quarantined_reasons),
            quarantined_reasons=quarantined_reasons,
            agreement_ratio=agreement_ratio,
            manifest=manifest,
            signed_bundle_bytes=bundle_bytes,
        )

    @staticmethod
    def _compute_median(sorted_vals: Sequence[float]) -> float:
        n = len(sorted_vals)
        if n == 0:
            return 0.0
        mid = n // 2
        if n % 2 == 1:
            return float(sorted_vals[mid])
        return 0.5 * (float(sorted_vals[mid - 1]) + float(sorted_vals[mid]))

    @classmethod
    def _trimmed_mean_and_median_1d(
        cls,
        values: Sequence[float],
        trim_fraction: float = 0.20,
    ) -> Tuple[float, float]:
        """Compute coordinate-wise trimmed mean and median across per-ASN values.

        Combines symmetric `trim_fraction` tail trimming with Median Absolute Deviation (MAD)
        Byzantine tail clipping so up to 35% malicious ASNs sending extreme values (e.g. +1000.0)
        on a single tail are mathematically clipped out.
        """
        if not values:
            return 0.0, 0.0
        sorted_vals = sorted(float(v) for v in values)
        n = len(sorted_vals)
        med = cls._compute_median(sorted_vals)
        if n <= 2:
            return med, med

        # Compute MAD (Median Absolute Deviation from the median)
        abs_devs = sorted(abs(v - med) for v in sorted_vals)
        mad = cls._compute_median(abs_devs)

        # Symmetric trim count from each tail
        k = int(n * max(0.0, min(float(trim_fraction), 0.45)))
        if 2 * k < n:
            sym_slice = sorted_vals[k : n - k]
        else:
            sym_slice = list(sorted_vals)

        # MAD-based Byzantine outlier clip (for any Byzantine fraction < 50%, median & MAD are honest-bounded)
        mad_cutoff = max(6.0 * mad, 1e-5)
        inliers = [v for v in sym_slice if abs(v - med) <= mad_cutoff]
        if not inliers:
            inliers = [med]

        trimmed_mean = sum(inliers) / float(len(inliers))
        return float(trimmed_mean), float(med)

    def aggregate_continuous_deltas(
        self,
        submissions: Sequence[Union[ContinuousDeltaSubmission, Dict[str, Any]]],
        trim_fraction: Optional[float] = None,
        min_distinct_asns: Optional[int] = None,
        require_authoritative_asn: bool = False,
        emit_bep46_manifest: bool = False,
        require_attestation: bool = True,
        *,
        now_timestamp: Optional[int] = None,
        max_age_seconds: int = 86400 * 14,
    ) -> ContinuousAggregationResult:
        """Group continuous calibration/adapter deltas by verified ASN first (1 vote per ASN),
        then compute coordinate-wise trimmed mean and coordinate-wise median.
        """
        eff_trim = self.trim_fraction if trim_fraction is None else float(trim_fraction)
        eff_min_asns = self.min_distinct_asns if min_distinct_asns is None else int(min_distinct_asns)
        now_ts = int(now_timestamp if now_timestamp is not None else time.time())
        raw_subs_list = list(submissions)
        total_input_count = len(raw_subs_list)

        rejected_count = 0
        coerced: List[ContinuousDeltaSubmission] = []
        for s in raw_subs_list:
            try:
                if isinstance(s, ContinuousDeltaSubmission):
                    if isinstance(s.asn, bool) or not isinstance(s.asn, int):
                        raise ValueError("invalid_asn_type")
                    if s.attestation is not None and not isinstance(s.attestation, PeerAttestation):
                        raise ValueError("invalid_attestation_type")
                    coerced.append(s)
                elif isinstance(s, dict):
                    b = s.get("encrypted_bundle")
                    if isinstance(b, dict):
                        b = EncryptedTelemetryBundle.from_dict(b)
                    elif b is not None and not isinstance(b, EncryptedTelemetryBundle):
                        raise ValueError("invalid_encrypted_bundle_type")
                    att_in = s.get("attestation")
                    if isinstance(att_in, dict):
                        att_in = PeerAttestation(
                            peer_id=str(att_in["peer_id"]),
                            asn=int(att_in["asn"]),
                            peer_pubkey_hex=str(att_in["peer_pubkey_hex"]),
                            timestamp=int(att_in["timestamp"]),
                            signature_hex=str(att_in["signature_hex"]),
                            ip_address=att_in.get("ip_address"),
                            pq_pubkey_hex=str(att_in.get("pq_pubkey_hex", "")),
                            pq_signature_hex=str(att_in.get("pq_signature_hex", "")),
                        )
                    elif att_in is not None and not isinstance(att_in, PeerAttestation):
                        raise ValueError("invalid_attestation_type")
                    raw_asn = s.get("asn")
                    if raw_asn is None or isinstance(raw_asn, bool):
                        raise ValueError("missing_or_invalid_asn")
                    raw_deltas_in = s.get("deltas", {})
                    if not isinstance(raw_deltas_in, dict):
                        raise ValueError("deltas_not_dict")
                    coerced.append(
                        ContinuousDeltaSubmission(
                            peer_id=str(s.get("peer_id", "")),
                            asn=int(raw_asn),
                            peer_pubkey_hex=str(s.get("peer_pubkey_hex", "")),
                            deltas=dict(raw_deltas_in),
                            ip_address=s.get("ip_address"),
                            encrypted_bundle=b,
                            attestation=att_in,
                            timestamp=int(s.get("timestamp", 0)),
                        )
                    )
                else:
                    raise TypeError(f"Unsupported continuous delta submission type: {type(s)}")
            except Exception:
                rejected_count += 1

        authenticated_cont: List[Tuple[int, str, Optional[str], int, str, Dict[str, Union[float, List[float]]]]] = []

        # Step 1: Authenticate each submission (ASN binding + mandatory PeerAttestation signature + AEAD decryption)
        # before building cross-submission spoofing sets so forged wrappers cannot poison legitimate peers.
        for sub in coerced:
            try:
                claimed_asn = int(sub.asn)
                norm_pk = AuthoritativeASNVerifier.normalize_pubkey(sub.peer_pubkey_hex) if sub.peer_pubkey_hex else ""
            except Exception:
                rejected_count += 1
                continue
            if not norm_pk:
                rejected_count += 1
                continue

            subnet: Optional[str] = None
            wrapper_ip: Optional[str] = None
            if sub.ip_address is not None and str(sub.ip_address).strip() != "":
                try:
                    wrapper_ip = str(ipaddress.ip_address(str(sub.ip_address).strip()))
                except ValueError:
                    rejected_count += 1
                    continue

            if sub.attestation is None:
                if require_attestation:
                    rejected_count += 1
                    continue
                eff_ip = wrapper_ip
                eff_ts = int(sub.timestamp) if not isinstance(sub.timestamp, bool) and isinstance(sub.timestamp, int) else -1
                if eff_ts < 0 or (eff_ts > 0 and ((now_ts - eff_ts) > max_age_seconds or (eff_ts - now_ts) > 300)):
                    rejected_count += 1
                    continue
                sub_sig_id = f"unatt:{norm_pk}:{eff_ts}"
            else:
                att = sub.attestation
                try:
                    att_pk = AuthoritativeASNVerifier.normalize_pubkey(att.peer_pubkey_hex)
                    att_asn = int(att.asn)
                    att_ts = int(att.timestamp)
                except Exception:
                    rejected_count += 1
                    continue
                if (
                    isinstance(att.timestamp, bool)
                    or not isinstance(att.timestamp, int)
                    or att_ts <= 0
                    or (now_ts - att_ts) > max_age_seconds
                    or (att_ts - now_ts) > 300
                ):
                    rejected_count += 1
                    continue
                if not isinstance(sub.timestamp, bool) and isinstance(sub.timestamp, int) and sub.timestamp not in (0, att_ts):
                    rejected_count += 1
                    continue
                eff_ts = att_ts
                if att_asn != claimed_asn or (norm_pk and att_pk != norm_pk):
                    rejected_count += 1
                    continue
                att_ip: Optional[str] = None
                if att.ip_address is not None and str(att.ip_address).strip() != "":
                    try:
                        att_ip = str(ipaddress.ip_address(str(att.ip_address).strip()))
                    except ValueError:
                        rejected_count += 1
                        continue
                if wrapper_ip is not None and att_ip is None:
                    rejected_count += 1
                    continue
                if wrapper_ip is not None and att_ip is not None and wrapper_ip != att_ip:
                    rejected_count += 1
                    continue
                eff_ip = att_ip if att_ip is not None else wrapper_ip
                try:
                    if sub.encrypted_bundle is not None:
                        target_h = hashlib.sha256(sub.encrypted_bundle.ciphertext_hex.encode("utf-8")).hexdigest()
                    else:
                        target_h = hashlib.sha256(
                            json.dumps(sub.deltas, sort_keys=True, separators=(",", ":")).encode("utf-8")
                        ).hexdigest()
                    pk_bytes = bytes.fromhex(att_pk)
                    sig_hex_norm = str(att.signature_hex).strip().lower()
                    sig_bytes = bytes.fromhex(sig_hex_norm)
                    ed25519.Ed25519PublicKey.from_public_bytes(pk_bytes).verify(
                        sig_bytes, att.canonical_bytes(target_h)
                    )
                    sub_sig_id = f"att:{att_pk}:{sig_hex_norm}:{target_h}"
                except Exception:
                    rejected_count += 1
                    continue

            if eff_ip is not None:
                try:
                    subnet = AuthoritativeASNVerifier.extract_subnet_prefix(eff_ip)
                except ValueError:
                    rejected_count += 1
                    continue

            bind_ok, _ = self.asn_verifier.verify_binding(
                claimed_asn=claimed_asn,
                pubkey_hex=norm_pk or None,
                ip_address=eff_ip,
                require_authoritative=require_authoritative_asn,
            )
            if not bind_ok:
                rejected_count += 1
                continue

            raw_deltas = dict(sub.deltas) if isinstance(sub.deltas, dict) else {}
            if sub.encrypted_bundle is not None:
                try:
                    plain = HybridPQCAEADEngine.decrypt_bundle(
                        sub.encrypted_bundle, self.aggregator_privkey_bytes
                    )
                    parsed = json.loads(plain.decode("utf-8"))
                    if not isinstance(parsed, dict):
                        rejected_count += 1
                        continue
                    dec_inner = parsed.get("deltas", parsed)
                    if not isinstance(dec_inner, dict) or not dec_inner:
                        rejected_count += 1
                        continue
                    decrypted_deltas = dict(dec_inner)
                except Exception:
                    rejected_count += 1
                    continue
                if raw_deltas and raw_deltas != decrypted_deltas:
                    rejected_count += 1
                    continue
                deltas_dict = decrypted_deltas
            else:
                deltas_dict = raw_deltas

            if not isinstance(deltas_dict, dict) or not deltas_dict:
                rejected_count += 1
                continue

            def _is_valid_finite_scalar(val: Any) -> bool:
                if isinstance(val, bool) or not isinstance(val, (int, float)):
                    return False
                try:
                    return math.isfinite(float(val))
                except (OverflowError, ValueError, TypeError):
                    return False

            has_non_finite = False
            for dv in deltas_dict.values():
                if isinstance(dv, (list, tuple)):
                    if not dv or any(not _is_valid_finite_scalar(x) for x in dv):
                        has_non_finite = True
                        break
                elif not _is_valid_finite_scalar(dv):
                    has_non_finite = True
                    break
            if has_non_finite:
                rejected_count += 1
                continue

            authenticated_cont.append((claimed_asn, norm_pk, subnet, eff_ts, sub_sig_id, deltas_dict))

        # Step 2: Detect cross-submission Sybil key/subnet multi-ASN spoofing among authenticated submissions
        seen_pk_asn: Dict[str, int] = {}
        seen_subnet_asn: Dict[str, int] = {}
        spoofed_pks: Set[str] = set()
        spoofed_subnets: Set[str] = set()
        for claimed_asn, norm_pk, subnet, _eff_ts, _sub_sig_id, _deltas_dict in authenticated_cont:
            if norm_pk in seen_pk_asn and seen_pk_asn[norm_pk] != claimed_asn:
                spoofed_pks.add(norm_pk)
            seen_pk_asn[norm_pk] = claimed_asn
            if subnet:
                if subnet in seen_subnet_asn and seen_subnet_asn[subnet] != claimed_asn:
                    spoofed_subnets.add(subnet)
                seen_subnet_asn[subnet] = claimed_asn

        # Step 2b: Deduplicate replay-resistant submission identities and allow at most 1 contribution per
        # authenticated peer (norm_pk) per aggregation window (deterministic newest-timestamp replacement).
        seen_submission_ids: Set[str] = set()
        peer_selected: Dict[str, Tuple[int, int, Dict[str, Union[float, List[float]]]]] = {}
        for claimed_asn, norm_pk, subnet, eff_ts, sub_sig_id, deltas_dict in authenticated_cont:
            if norm_pk in spoofed_pks or (subnet and subnet in spoofed_subnets):
                rejected_count += 1
                continue
            if sub_sig_id in seen_submission_ids:
                rejected_count += 1
                continue
            seen_submission_ids.add(sub_sig_id)
            prev_entry = peer_selected.get(norm_pk)
            if prev_entry is None:
                peer_selected[norm_pk] = (claimed_asn, eff_ts, deltas_dict)
            elif eff_ts > prev_entry[1]:
                # Deterministic replacement when a strictly newer signed submission arrives from the same peer
                peer_selected[norm_pk] = (claimed_asn, eff_ts, deltas_dict)
                rejected_count += 1
            else:
                rejected_count += 1

        per_asn_raw: Dict[int, List[Dict[str, Union[float, List[float]]]]] = {}
        for _norm_pk, (claimed_asn, _eff_ts, deltas_dict) in sorted(peer_selected.items()):
            per_asn_raw.setdefault(claimed_asn, []).append(deltas_dict)

        verified_asns = sorted(per_asn_raw.keys())
        if len(verified_asns) < eff_min_asns:
            return ContinuousAggregationResult(
                success=False,
                reason=f"insufficient_distinct_asns:{len(verified_asns)}<{eff_min_asns}",
                distinct_asn_count=len(verified_asns),
                verified_asns=verified_asns,
                total_submissions=total_input_count,
                rejected_submissions=rejected_count,
                trimmed_mean_deltas={},
                median_deltas={},
                trim_fraction=eff_trim,
            )

        # Step 1: Collapse intra-ASN submissions to exactly 1 vote per verified ASN (via intra-ASN median)
        # while enforcing consistent type and vector dimension per key across ASNs.
        all_keys = sorted({k for obs_list in per_asn_raw.values() for d in obs_list for k in d.keys()})
        per_asn_single_vote: Dict[int, Dict[str, Union[float, List[float]]]] = {asn: {} for asn in verified_asns}

        for key in all_keys:
            # Determine canonical (is_vec, dim) for this key across ASNs
            shape_asn_counts: Dict[Tuple[bool, int], int] = {}
            for asn in verified_asns:
                asn_entries = [d[key] for d in per_asn_raw[asn] if key in d]
                if not asn_entries:
                    continue
                first_v = asn_entries[0]
                sh = (True, len(first_v)) if isinstance(first_v, (list, tuple)) else (False, 1)
                if all(
                    ((True, len(e)) if isinstance(e, (list, tuple)) else (False, 1)) == sh
                    for e in asn_entries
                ):
                    shape_asn_counts[sh] = shape_asn_counts.get(sh, 0) + 1
            if not shape_asn_counts:
                continue
            canon_is_vec, canon_dim = max(shape_asn_counts.items(), key=lambda kv: kv[1])[0]

            for asn in verified_asns:
                raw_asn_entries = [d[key] for d in per_asn_raw[asn] if key in d]
                if not raw_asn_entries:
                    continue
                if canon_is_vec:
                    asn_entries = [
                        e for e in raw_asn_entries
                        if isinstance(e, (list, tuple)) and len(e) == canon_dim
                    ]
                    if len(asn_entries) != len(raw_asn_entries):
                        rejected_count += len(raw_asn_entries) - len(asn_entries)
                    if not asn_entries:
                        continue
                    vec_vote: List[float] = []
                    for coord_idx in range(canon_dim):
                        coord_samples = sorted(float(entry[coord_idx]) for entry in asn_entries)
                        vec_vote.append(self._compute_median(coord_samples))
                    per_asn_single_vote[asn][key] = vec_vote
                else:
                    asn_entries_scalar = [
                        float(e) for e in raw_asn_entries
                        if not isinstance(e, (list, tuple, bool)) and isinstance(e, (int, float))
                    ]
                    if len(asn_entries_scalar) != len(raw_asn_entries):
                        rejected_count += len(raw_asn_entries) - len(asn_entries_scalar)
                    if not asn_entries_scalar:
                        continue
                    scalar_samples = sorted(asn_entries_scalar)
                    per_asn_single_vote[asn][key] = self._compute_median(scalar_samples)

        # Step 2: Across the distinct ASNs (1 vote per ASN), enforce per-key ASN quorum >= eff_min_asns
        # and compute coordinate-wise trimmed mean and median.
        trimmed_mean_out: Dict[str, Union[float, List[float]]] = {}
        median_out: Dict[str, Union[float, List[float]]] = {}

        for key in all_keys:
            asn_vals = [per_asn_single_vote[asn][key] for asn in verified_asns if key in per_asn_single_vote[asn]]
            if len(asn_vals) < eff_min_asns:
                continue
            if isinstance(asn_vals[0], list):
                dim = len(asn_vals[0])
                if any(not isinstance(v, list) or len(v) != dim for v in asn_vals):
                    continue
                t_vec: List[float] = []
                m_vec: List[float] = []
                for coord_idx in range(dim):
                    coord_votes = [float(v[coord_idx]) for v in asn_vals]  # type: ignore[index]
                    t_val, m_val = self._trimmed_mean_and_median_1d(coord_votes, trim_fraction=eff_trim)
                    t_vec.append(round(t_val, 6))
                    m_vec.append(round(m_val, 6))
                trimmed_mean_out[key] = t_vec
                median_out[key] = m_vec
            else:
                scalar_votes = [float(v) for v in asn_vals]  # type: ignore[arg-type]
                t_val, m_val = self._trimmed_mean_and_median_1d(scalar_votes, trim_fraction=eff_trim)
                trimmed_mean_out[key] = round(t_val, 6)
                median_out[key] = round(m_val, 6)

        if not trimmed_mean_out:
            return ContinuousAggregationResult(
                success=False,
                reason=f"insufficient_per_key_asn_quorum:<{eff_min_asns}",
                distinct_asn_count=len(verified_asns),
                verified_asns=verified_asns,
                total_submissions=total_input_count,
                rejected_submissions=rejected_count,
                trimmed_mean_deltas={},
                median_deltas={},
                trim_fraction=eff_trim,
            )

        manifest = None
        bundle_bytes = None
        if emit_bep46_manifest:
            payload_obj = {
                "distinct_asn_count": len(verified_asns),
                "median_deltas": median_out,
                "trimmed_mean_deltas": trimmed_mean_out,
                "verified_asns": verified_asns,
            }
            bundle_bytes = json.dumps(payload_obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
            manifest = self._sign_bep46_bundle(bundle_bytes, min_asn_quorum=len(verified_asns))

        return ContinuousAggregationResult(
            success=True,
            reason="per_asn_trimmed_mean_and_median_computed",
            distinct_asn_count=len(verified_asns),
            verified_asns=verified_asns,
            total_submissions=total_input_count,
            rejected_submissions=rejected_count,
            trimmed_mean_deltas=trimmed_mean_out,
            median_deltas=median_out,
            trim_fraction=eff_trim,
            manifest=manifest,
            signed_bundle_bytes=bundle_bytes,
        )


