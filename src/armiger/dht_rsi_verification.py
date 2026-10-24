"""Physical Ground-Truth Verifier, Hybrid PQ Threshold Signatures, ASN/Sybil Quorum, and Safe Format Validator."""
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
    AcyclicReleaseEnvelope,
    AuthenticatedConformalBaseline,
    BenchmarkAttestationPredicate,
    BundleFormat,
    CandidateObservation,
    DSSE_TYPE_ATTESTATION_V2,
    DSSE_TYPE_BASELINE_V1,
    DSSE_TYPE_RELEASE_V2,
    MediaCategory,
    PeerAttestation,
    PhysicalVerificationResult,
    ThresholdDHTManifest,
    VerifiedObservation,
    dsse_pae,
)
from .dht_rsi_crypto_and_adapters import (
    MediaStreamVerifier,
    WOTSPlusSHAKE256,
)

# =====================================================================
# 6. Physical Ground-Truth Verifier
# =====================================================================

class PhysicalGroundTruthVerifier:
    """Enforces foolproof physical ground truth verification (NEVER unverified model predictions)."""

    JUNK_EXTENSIONS = {".exe", ".scr", ".vbs", ".bat", ".cmd", ".lnk", ".pif", ".ps1", ".msi", ".com"}
    DANGEROUS_MASQUERADES = re.compile(
        r'(\.(mkv|mp4|avi|mov|wmv|flv|webm|flac|mp3|m4b|epub|pdf|cbr|cbz)\.(exe|scr|vbs|bat|cmd|lnk|pif|ps1|msi|com|jar)$|\.(scr|vbs|bat|cmd|lnk|pif|ps1)$)',
        re.IGNORECASE,
    )
    SUSPICIOUS_PASSWORDS = re.compile(
        r'(password|pass\s*in\s*desc|pw\s*txt|unlock_code)',
        re.IGNORECASE,
    )

    def verify(
        self,
        candidate: CandidateObservation,
        ffprobe_output: Optional[Dict[str, Any]] = None,
    ) -> PhysicalVerificationResult:
        """Verify candidate observation against physical ground truth."""
        # Rule 1: Download client completion is strictly required
        if not candidate.download_completed:
            return PhysicalVerificationResult(
                is_verified=False,
                rejection_reason="incomplete_download_rejected",
            )

        if (
            not isinstance(candidate.total_bytes, (int, float))
            or not math.isfinite(candidate.total_bytes)
            or candidate.total_bytes <= 0
        ):
            return PhysicalVerificationResult(
                is_verified=False,
                rejection_reason="zero_byte_content_rejected",
            )
        if (
            not isinstance(candidate.bytes_downloaded, (int, float))
            or not math.isfinite(candidate.bytes_downloaded)
            or candidate.bytes_downloaded < candidate.total_bytes
        ):
            return PhysicalVerificationResult(
                is_verified=False,
                rejection_reason="incomplete_bytes_downloaded_rejected",
            )

        # Rule 2: Detect active malware masquerades and passworded fake archives
        raw_name = candidate.raw_title.lower()
        is_exe_media_masquerade = (
            raw_name.endswith((".exe", ".msi", ".com"))
            and candidate.claimed_category not in ("software", "game")
        )
        if (
            self.DANGEROUS_MASQUERADES.search(raw_name)
            or self.SUSPICIOUS_PASSWORDS.search(raw_name)
            or is_exe_media_masquerade
        ):
            # Confirmed structural masquerade / password trap: only junk and excluded owner_scope are verified facts.
            verified_obs = VerifiedObservation(
                category="unknown",
                adult=False,
                anime=False,
                lang="multi_or_unknown",
                junk=True,  # Confirmed physical junk / malware!
                owner_scope="excluded",
                english_subs=False,
                title_language="english_title",
                source_adapter=candidate.source_adapter,
                catalog_id_type="none",
                catalog_id_value="malware_trap",
                size_bytes=candidate.total_bytes,
                raw_title=candidate.raw_title,
                stream_info={},
                verified_at=int(time.time()),
                evidence_masks={
                    "category": False,
                    "adult": False,
                    "anime": False,
                    "lang": False,
                    "junk": True,
                    "owner_scope": True,
                    "english_subs": False,
                    "title_language": False,
                },
                evidence_sources={
                    "category": "unverified_on_masquerade_trap",
                    "adult": "unverified_on_masquerade_trap",
                    "anime": "unverified_on_masquerade_trap",
                    "lang": "unverified_on_masquerade_trap",
                    "junk": "observed_fact:structural_masquerade_trap",
                    "owner_scope": "derived_policy:excluded_masquerade_trap",
                    "english_subs": "unverified_on_masquerade_trap",
                    "title_language": "unverified_on_masquerade_trap",
                },
            )
            return PhysicalVerificationResult(is_verified=True, observation=verified_obs)

        # Rule 3: Reconcile with ffprobe/MediaInfo streams if video or audio is claimed
        stream_info: Dict[str, Any] = {}
        if ffprobe_output is not None:
            stream_info = MediaStreamVerifier.verify_ffprobe_json(ffprobe_output)
            if not stream_info.get("is_valid_media"):
                # File claims to be video/audio but has no playable stream
                return PhysicalVerificationResult(
                    is_verified=False,
                    rejection_reason="invalid_unplayable_media_stream",
                )
        elif candidate.stream_metadata:
            stream_info = dict(candidate.stream_metadata)

        # Rule 4: Canonical Provenance Verification
        # For non-junk items, we require an authoritative catalog ID or verified stream match
        cat_type = candidate.catalog_id_type or "none"
        cat_val = str(candidate.catalog_id_value or "")

        is_catalog_verified = bool(cat_type in ("series_catalog", "movie_catalog", "audio_catalog", "isbn", "anime_catalog") and cat_val)
        is_stream_verified = bool(stream_info.get("video_codec") or stream_info.get("audio_codec"))

        if not (is_catalog_verified or is_stream_verified):
            return PhysicalVerificationResult(
                is_verified=False,
                rejection_reason="unverified_provenance_rejected",
            )

        # Derive scoped ground-truth labels and explicit verification masks across 8 heads.
        # Observed facts, independent reference annotations, and unverified/unknown targets are strictly separated.
        category, cat_mask, cat_src = self._resolve_ground_truth_category_with_evidence(candidate, stream_info)
        adult, adult_mask, adult_src = self._resolve_ground_truth_adult_with_evidence(candidate)
        anime, anime_mask, anime_src = self._resolve_ground_truth_anime_with_evidence(candidate)
        lang, lang_mask, lang_src = self._resolve_ground_truth_audio_lang_with_evidence(candidate, stream_info)
        english_subs, subs_mask, subs_src = self._resolve_ground_truth_english_subs_with_evidence(candidate, stream_info)
        title_lang, tlang_mask, tlang_src = self._resolve_ground_truth_title_lang_with_evidence(candidate)
        junk, junk_mask, junk_src = self._resolve_ground_truth_junk_with_evidence(candidate)
        owner_scope, scope_mask, scope_src = self._resolve_ground_truth_owner_scope_with_evidence(
            candidate,
            category=category,
            cat_mask=cat_mask,
            adult=adult,
            adult_mask=adult_mask,
            junk=junk,
            junk_mask=junk_mask,
        )

        obs = VerifiedObservation(
            category=category,
            adult=adult,
            anime=anime,
            lang=lang,
            junk=junk,
            owner_scope=owner_scope,
            english_subs=english_subs,
            title_language=title_lang,
            source_adapter=candidate.source_adapter,
            catalog_id_type=cat_type,
            catalog_id_value=cat_val,
            size_bytes=candidate.total_bytes,
            raw_title=candidate.raw_title,
            stream_info=stream_info,
            verified_at=int(time.time()),
            evidence_masks={
                "category": cat_mask,
                "adult": adult_mask,
                "anime": anime_mask,
                "lang": lang_mask,
                "junk": junk_mask,
                "owner_scope": scope_mask,
                "english_subs": subs_mask,
                "title_language": tlang_mask,
            },
            evidence_sources={
                "category": cat_src,
                "adult": adult_src,
                "anime": anime_src,
                "lang": lang_src,
                "junk": junk_src,
                "owner_scope": scope_src,
                "english_subs": subs_src,
                "title_language": tlang_src,
            },
        )
        return PhysicalVerificationResult(is_verified=True, observation=obs)

    @staticmethod
    def _independent_ref(c: CandidateObservation) -> Dict[str, Any]:
        extra = c.extra_metadata if isinstance(c.extra_metadata, Mapping) else {}
        ref = extra.get("independent_evidence")
        return dict(ref) if isinstance(ref, Mapping) else {}

    @classmethod
    def _resolve_ground_truth_category_with_evidence(
        cls,
        c: CandidateObservation,
        s: Dict[str, Any],
    ) -> Tuple[str, bool, str]:
        ref = cls._independent_ref(c)
        valid_cats = {m.value for m in MediaCategory if m != MediaCategory.UNKNOWN}
        if ref.get("category") in valid_cats:
            return str(ref["category"]), True, "independent_reference:extra_metadata"

        cat_type = str(c.catalog_id_type or "").strip()
        cat_val = str(c.catalog_id_value or "").strip()
        if not cat_val or cat_type in ("", "none"):
            # Do NOT echo adapter claimed_category or filename heuristics without catalog/reference provenance.
            return "unknown", False, "unverified_without_catalog_reference"

        # If physical ffprobe output was provided, check that stream media type does not contradict catalog type.
        ffprobe_checked = "is_valid_media" in s
        has_video = bool(s.get("video_codec"))
        has_audio = bool(s.get("audio_codec"))

        if cat_type == "series_catalog":
            if ffprobe_checked and not has_video:
                return "unknown", False, "contradicted_catalog_video_missing"
            return "tv_show", True, "independent_reference:series_catalog"
        if cat_type == "movie_catalog":
            if ffprobe_checked and not has_video:
                return "unknown", False, "contradicted_catalog_video_missing"
            return "movie", True, "independent_reference:movie_catalog"
        if cat_type == "anime_catalog":
            if ffprobe_checked and not has_video:
                return "unknown", False, "contradicted_catalog_video_missing"
            return "anime", True, "independent_reference:anime_catalog"
        if cat_type == "audio_catalog":
            if ffprobe_checked and (has_video or not has_audio):
                return "unknown", False, "contradicted_catalog_audio_mismatch"
            return "music", True, "independent_reference:audio_catalog"
        if cat_type == "isbn":
            fmt = str(s.get("format") or "").strip().lower()
            if fmt in ("m4b", "mp3", "aa", "aax", "flac", "ogg", "opus") or (has_audio and not has_video):
                return "audiobook", True, "independent_reference:isbn+audio_format"
            if fmt in ("epub", "mobi", "azw3", "pdf", "djvu", "fb2") and not has_video:
                return "ebook", True, "independent_reference:isbn+ebook_format"
            return "unknown", False, "unverified_isbn_format_ambiguous"
        return "unknown", False, "unverified_unknown_catalog_type"

    @classmethod
    def _resolve_ground_truth_category(cls, c: CandidateObservation, s: Dict[str, Any]) -> str:
        cat, _mask, _src = cls._resolve_ground_truth_category_with_evidence(c, s)
        return cat

    @classmethod
    def _resolve_ground_truth_adult_with_evidence(cls, c: CandidateObservation) -> Tuple[bool, bool, str]:
        ref = cls._independent_ref(c)
        if isinstance(ref.get("adult"), bool):
            return bool(ref["adult"]), True, "independent_reference:adult"
        # Never infer verified adult truth from title keywords or unverified claimed_category.
        return False, False, "unverified_without_independent_adult_reference"

    @classmethod
    def _resolve_ground_truth_adult(cls, c: CandidateObservation) -> bool:
        val, _mask, _src = cls._resolve_ground_truth_adult_with_evidence(c)
        return val

    @classmethod
    def _resolve_ground_truth_anime_with_evidence(cls, c: CandidateObservation) -> Tuple[bool, bool, str]:
        ref = cls._independent_ref(c)
        if isinstance(ref.get("anime"), bool):
            return bool(ref["anime"]), True, "independent_reference:anime"
        if c.catalog_id_type == "anime_catalog" and c.catalog_id_value:
            return True, True, "independent_reference:anime_catalog"
        # Never infer verified anime truth from canary title words or release group tags.
        return False, False, "unverified_without_anime_catalog_reference"

    @classmethod
    def _resolve_ground_truth_anime(cls, c: CandidateObservation) -> bool:
        val, _mask, _src = cls._resolve_ground_truth_anime_with_evidence(c)
        return val

    _UNDETERMINED_LANG_CODES = {"", "und", "unk", "unknown", "mul", "zxx", "mis", "none"}
    _ENGLISH_LANG_CODES = {"eng", "en", "en-us", "en-gb", "english"}

    @classmethod
    def _resolve_ground_truth_audio_lang_with_evidence(
        cls,
        c: CandidateObservation,
        s: Dict[str, Any],
    ) -> Tuple[str, bool, str]:
        ref = cls._independent_ref(c)
        if ref.get("lang") in ("english", "non_english", "multi_or_unknown"):
            return str(ref["lang"]), True, "independent_reference:lang"

        langs = s.get("audio_languages")
        raw_tokens: List[str] = []
        if isinstance(langs, list):
            for item in langs:
                if item is not None:
                    raw_tokens.extend(t for t in re.split(r"[/,;|\s]+", str(item).strip().lower()) if t)
        elif isinstance(langs, str) and langs.strip():
            raw_tokens.extend(t for t in re.split(r"[/,;|\s]+", langs.strip().lower()) if t)

        if not raw_tokens:
            # Missing audio_languages is unknown, never defaulted to 'english'.
            return "multi_or_unknown", False, "unverified_missing_audio_languages"

        if any(t in cls._UNDETERMINED_LANG_CODES for t in raw_tokens):
            return "multi_or_unknown", False, "unverified_undetermined_audio_language_tag"

        norm_langs: Set[str] = set()
        for tok in raw_tokens:
            norm_langs.add("english" if tok in cls._ENGLISH_LANG_CODES else f"non_en:{tok}")

        if len(norm_langs) > 1:
            return "multi_or_unknown", True, "observed_fact:multi_language_audio_streams"
        only = next(iter(norm_langs))
        if only == "english":
            return "english", True, "observed_fact:english_audio_stream_tags"
        return "non_english", True, "observed_fact:non_english_audio_stream_tags"

    @classmethod
    def _resolve_ground_truth_audio_lang(cls, c: CandidateObservation, s: Dict[str, Any]) -> str:
        val, _mask, _src = cls._resolve_ground_truth_audio_lang_with_evidence(c, s)
        return val

    @classmethod
    def _resolve_ground_truth_english_subs_with_evidence(
        cls,
        c: CandidateObservation,
        s: Dict[str, Any],
    ) -> Tuple[bool, bool, str]:
        ref = cls._independent_ref(c)
        if isinstance(ref.get("english_subs"), bool):
            return bool(ref["english_subs"]), True, "independent_reference:english_subs"

        if "sub_languages" in s and isinstance(s.get("sub_languages"), list):
            tokens: Set[str] = set()
            for x in s["sub_languages"]:
                tokens.update(re.findall(r"[a-z]+", str(x).lower()))
            has_eng = bool(tokens & {"eng", "en", "english"})
            return has_eng, True, "observed_fact:ffprobe_subtitle_streams"

        if s.get("subtitles") is not None and str(s.get("subtitles")).strip() != "":
            tokens = set(re.findall(r"[a-z]+", str(s.get("subtitles")).lower()))
            has_eng = bool(tokens & {"eng", "en", "english"})
            return has_eng, True, "observed_fact:mediainfo_subtitle_tags"

        return False, False, "unverified_subtitle_inventory_not_inspected"

    @classmethod
    def _resolve_ground_truth_english_subs(cls, s: Dict[str, Any]) -> bool:
        dummy = CandidateObservation(
            source_adapter="direct",
            raw_title="",
            claimed_category="unknown",
            download_completed=True,
            bytes_downloaded=1,
            total_bytes=1,
        )
        val, _mask, _src = cls._resolve_ground_truth_english_subs_with_evidence(dummy, s)
        return val

    @classmethod
    def _resolve_ground_truth_title_lang_with_evidence(cls, c: CandidateObservation) -> Tuple[str, bool, str]:
        ref = cls._independent_ref(c)
        if ref.get("title_language") in ("english_title", "non_english_title"):
            return str(ref["title_language"]), True, "independent_reference:title_language"
        # ASCII encodability does not independently establish English vs non-English title language.
        fallback = "english_title" if cls._is_english_title(c.raw_title) else "non_english_title"
        return fallback, False, "unverified_ascii_heuristic_excluded"

    @classmethod
    def _resolve_ground_truth_junk_with_evidence(cls, c: CandidateObservation) -> Tuple[bool, bool, str]:
        ref = cls._independent_ref(c)
        if isinstance(ref.get("junk"), bool):
            return bool(ref["junk"]), True, "independent_reference:junk"
        return False, False, "scoped_observation:no_masquerade_trap_detected"

    @classmethod
    def _resolve_ground_truth_owner_scope_with_evidence(
        cls,
        c: CandidateObservation,
        *,
        category: str,
        cat_mask: bool,
        adult: bool,
        adult_mask: bool,
        junk: bool,
        junk_mask: bool,
    ) -> Tuple[str, bool, str]:
        ref = cls._independent_ref(c)
        if ref.get("owner_scope") in ("allowed_type", "excluded"):
            return str(ref["owner_scope"]), True, "independent_reference:owner_scope"
        if junk and junk_mask:
            return "excluded", True, "derived_policy:excluded_junk"
        if adult and adult_mask:
            return "excluded", True, "derived_policy:excluded_adult"
        if cat_mask and category == "xxx":
            return "excluded", True, "derived_policy:excluded_xxx_category"
        if cat_mask and adult_mask and junk_mask and not junk and not adult and category != "xxx":
            return "allowed_type", True, "derived_policy:verified_allowed_type"
        return "allowed_type", False, "unverified_without_complete_policy_inputs"

    @staticmethod
    def _is_english_title(title: str) -> bool:
        try:
            title.encode('ascii')
            return True
        except UnicodeEncodeError:
            return False


# =====================================================================
# 7. Spoof-Proof Protocol (BEP-44 Monotonic Sequences & Ed25519 Threshold Signatures)
# =====================================================================

class AuthoritativeASNVerifier:
    """Authoritative BGP CIDR IPv4/IPv6-to-ASN route verifier and pinned ASN public-key binding guard.

    Prevents self-asserted ASN spoofing where a malicious peer or single /24 (IPv4) or /48 (IPv6)
    subnet claims multiple distinct ASNs, reuses public keys across attestations, or claims an ASN
    inconsistent with BGP routing tables and pinned public-key bindings.
    """

    DEFAULT_ASN_CIDR_ROUTES: Dict[str, int] = {
        **{
            f"{a}.{b}.{c}.{d}/{plen}": asn
            for (a, b, c, d, plen, asn) in (
                (104, 16, 0, 0, 12, 13335),
                (172, 64, 0, 0, 13, 13335),
                (8, 8, 8, 0, 24, 15169),
                (8, 8, 4, 0, 24, 15169),
                (142, 250, 0, 0, 15, 15169),
                (52, 94, 0, 0, 16, 16509),
                (54, 239, 0, 0, 16, 16509),
                (73, 0, 0, 0, 8, 7922),
                (12, 0, 0, 0, 8, 7018),
                (80, 128, 0, 0, 11, 3320),
                (88, 198, 0, 0, 16, 24940),
                (51, 38, 0, 0, 16, 16276),
                (104, 131, 0, 0, 16, 14061),
                (45, 33, 0, 0, 16, 63949),
                (23, 32, 0, 0, 11, 20940),
                (20, 36, 0, 0, 14, 8075),
                (157, 240, 0, 0, 16, 32934),
                (45, 57, 0, 0, 17, 2906),
                (185, 42, 204, 0, 22, 46489),
                (151, 101, 0, 0, 16, 54113),
                (34, 64, 0, 0, 10, 396982),
                (62, 115, 0, 0, 16, 1299),
                (4, 0, 0, 0, 9, 3356),
                (38, 0, 0, 0, 8, 174),
            )
        },
        **{
            f"{h1}:{h2}" + "::" + f"/{plen}": asn
            for (h1, h2, plen, asn) in (
                ("2606", "4700", 32, 13335),
                ("2001", "4860", 32, 15169),
                ("2600", "1f00", 24, 16509),
                ("2001", "558", 32, 7922),
                ("2001", "1890", 32, 7018),
                ("2003", "0", 19, 3320),
                ("2a01", "4f8", 32, 24940),
                ("2001", "41d0", 32, 16276),
                ("2604", "a880", 32, 14061),
                ("2600", "3c00", 32, 63949),
            )
        },
    }

    _GLOBAL_PINNED_PUBKEY_ASN: Dict[str, int] = {}
    _GLOBAL_PINNED_PUBKEY_PQ_ROOT: Dict[str, str] = {}

    @classmethod
    def register_global_pinned_key(
        cls,
        pubkey_hex: str,
        asn: int,
        pq_root_pubkey_hex: Optional[str] = None,
    ) -> None:
        """Pin a peer public key hex globally to a single authoritative ASN (and optional PQ root pubkey)."""
        if int(asn) <= 0:
            raise ValueError(f"ASN must be positive, got {asn}")
        raw_str = str(pubkey_hex).strip().lower()
        if ":" in raw_str and not pq_root_pubkey_hex:
            _, pq_part = raw_str.split(":", 1)
            pq_root_pubkey_hex = pq_part.strip()
        norm_pk = cls.normalize_pubkey(pubkey_hex)
        if not norm_pk:
            raise ValueError("Empty public key cannot be pinned")
        cls._GLOBAL_PINNED_PUBKEY_ASN[norm_pk] = int(asn)
        eff_pq = (pq_root_pubkey_hex.strip().lower() if pq_root_pubkey_hex else None) or WOTSPlusSHAKE256.get_root_wots_pub(norm_pk)
        if eff_pq and len(eff_pq) == 64:
            cls._GLOBAL_PINNED_PUBKEY_PQ_ROOT[norm_pk] = eff_pq

    def __init__(
        self,
        cidr_asn_map: Optional[Dict[str, int]] = None,
        pinned_pubkey_asn: Optional[Dict[str, int]] = None,
        require_authoritative_binding: bool = True,
        include_default_routes: bool = True,
        pinned_pubkey_pq_root: Optional[Dict[str, str]] = None,
    ) -> None:
        self.require_authoritative_binding = bool(require_authoritative_binding)
        self.pinned_pubkey_asn: Dict[str, int] = dict(self._GLOBAL_PINNED_PUBKEY_ASN)
        self.pinned_pubkey_pq_root: Dict[str, str] = dict(self._GLOBAL_PINNED_PUBKEY_PQ_ROOT)
        self._ipv4_routes: List[Tuple[ipaddress.IPv4Network, int]] = []
        self._ipv6_routes: List[Tuple[ipaddress.IPv6Network, int]] = []

        if include_default_routes:
            for cidr, asn in self.DEFAULT_ASN_CIDR_ROUTES.items():
                self.register_cidr(cidr, asn)

        if cidr_asn_map:
            for cidr, asn in cidr_asn_map.items():
                self.register_cidr(cidr, asn)

        if pinned_pubkey_asn:
            for pk, asn in pinned_pubkey_asn.items():
                self.register_pinned_key(pk, asn)

        if pinned_pubkey_pq_root:
            for pk, pq_root in pinned_pubkey_pq_root.items():
                norm_pk = self.normalize_pubkey(pk)
                norm_pq = self.normalize_pubkey(pq_root)
                if norm_pk and len(norm_pq) == 64:
                    self.pinned_pubkey_pq_root[norm_pk] = norm_pq

    @staticmethod
    def normalize_pubkey(pubkey_hex: str) -> str:
        cleaned = str(pubkey_hex).strip().lower()
        if cleaned.startswith("0x"):
            cleaned = cleaned[2:]
        return re.split(r"[:#/\s]", cleaned)[0]

    def register_cidr(self, cidr: str, asn: int) -> None:
        """Register an authoritative IPv4 or IPv6 CIDR prefix to ASN route."""
        if int(asn) <= 0:
            raise ValueError(f"ASN must be positive, got {asn}")
        net = ipaddress.ip_network(str(cidr).strip(), strict=False)
        if isinstance(net, ipaddress.IPv4Network):
            self._ipv4_routes.append((net, int(asn)))
            self._ipv4_routes.sort(key=lambda item: item[0].prefixlen, reverse=True)
        else:
            self._ipv6_routes.append((net, int(asn)))
            self._ipv6_routes.sort(key=lambda item: item[0].prefixlen, reverse=True)

    def register_pinned_key(
        self,
        pubkey_hex: str,
        asn: int,
        pq_root_pubkey_hex: Optional[str] = None,
    ) -> None:
        """Pin a peer or signer public key hex to a single authoritative ASN (and optional PQ root pubkey)."""
        if int(asn) <= 0:
            raise ValueError(f"ASN must be positive, got {asn}")
        raw_str = str(pubkey_hex).strip().lower()
        if ":" in raw_str and not pq_root_pubkey_hex:
            _, pq_part = raw_str.split(":", 1)
            pq_root_pubkey_hex = pq_part.strip()
        norm_pk = self.normalize_pubkey(pubkey_hex)
        if not norm_pk:
            raise ValueError("Empty public key cannot be pinned")
        self.pinned_pubkey_asn[norm_pk] = int(asn)
        eff_pq = (pq_root_pubkey_hex.strip().lower() if pq_root_pubkey_hex else None) or WOTSPlusSHAKE256.get_root_wots_pub(norm_pk)
        if eff_pq and len(eff_pq) == 64:
            self.pinned_pubkey_pq_root[norm_pk] = eff_pq

    @staticmethod
    def extract_subnet_prefix(ip_address: str) -> str:
        """Return canonical /24 (IPv4) or /48 (IPv6) Sybil aggregation subnet for an IP address.

        Unwraps IPv4-mapped IPv6 addresses (`::ffff:a.b.c.d`) to their canonical IPv4 `/24`
        so a dual-stack host cannot bypass `/24` Sybil limits by presenting `::ffff:` addresses.
        """
        addr = ipaddress.ip_address(str(ip_address).strip())
        if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
            addr = addr.ipv4_mapped
        if isinstance(addr, ipaddress.IPv4Address):
            return str(ipaddress.IPv4Network(f"{addr}/24", strict=False))
        return str(ipaddress.IPv6Network(f"{addr}/48", strict=False))

    def lookup_ip_asn(self, ip_address: str) -> Optional[int]:
        """Resolve an IPv4 or IPv6 address to its longest-prefix-match ASN, or None if unrouted."""
        addr = ipaddress.ip_address(str(ip_address).strip())
        if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
            addr = addr.ipv4_mapped
        if isinstance(addr, ipaddress.IPv4Address):
            for net, asn in self._ipv4_routes:
                if addr in net:
                    return asn
        else:
            for net, asn in self._ipv6_routes:
                if addr in net:
                    return asn
        return None

    def verify_binding(
        self,
        claimed_asn: int,
        pubkey_hex: Optional[str] = None,
        ip_address: Optional[str] = None,
        require_authoritative: Optional[bool] = None,
        require_pinned_key: bool = False,
    ) -> Tuple[bool, str]:
        """Verify that claimed_asn matches pinned public-key bindings and/or BGP CIDR IP routes."""
        if int(claimed_asn) <= 0:
            return False, f"invalid_asn:{claimed_asn}"

        req_auth = self.require_authoritative_binding if require_authoritative is None else bool(require_authoritative)
        has_verified_pin = False
        has_verified_route = False

        if pubkey_hex:
            norm_pk = self.normalize_pubkey(pubkey_hex)
            pinned_asn = self.pinned_pubkey_asn.get(norm_pk)
            if pinned_asn is None:
                pinned_asn = self._GLOBAL_PINNED_PUBKEY_ASN.get(norm_pk)
            if pinned_asn is not None:
                if pinned_asn != int(claimed_asn):
                    return False, f"pinned_pubkey_asn_mismatch:claimed={claimed_asn}!=pinned={pinned_asn}"
                has_verified_pin = True

        if ip_address is not None and str(ip_address).strip() != "":
            try:
                routed_asn = self.lookup_ip_asn(str(ip_address).strip())
            except ValueError:
                return False, f"invalid_ip_address:{ip_address}"
            if routed_asn is not None:
                if routed_asn != int(claimed_asn):
                    return False, f"cidr_route_asn_mismatch:claimed={claimed_asn}!=routed={routed_asn}"
                has_verified_route = True
            elif req_auth and not has_verified_pin:
                return False, f"unrouted_ip_for_asn:{ip_address}->asn={claimed_asn}"

        if require_pinned_key and not has_verified_pin:
            return False, f"unverified_authoritative_asn_binding:asn={claimed_asn}"

        if req_auth and not (has_verified_pin or has_verified_route):
            return False, f"unverified_authoritative_asn_binding:asn={claimed_asn}"

        return True, "authoritative_asn_verified"


class ThresholdSignatureVerifier:
    """Verifies Hybrid Ed25519 + WOTS+-SHAKE256 Post-Quantum threshold multi-signatures and BEP-44 sequences."""

    _USED_PQ_LEAVES: Dict[str, Tuple[int, str]] = {}

    @staticmethod
    def generate_hybrid_signer() -> Tuple[ed25519.Ed25519PrivateKey, str, bytes, str]:
        """Generate a paired (Ed25519PrivateKey, ed25519_pub_hex, wots_priv_seed_64B, wots_pub_hex) signer."""
        from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat, PrivateFormat, NoEncryption
        ed_priv = ed25519.Ed25519PrivateKey.generate()
        ed_pub_hex = ed_priv.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
        ed_raw_priv = ed_priv.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())
        pq_priv, pq_pub = WOTSPlusSHAKE256.derive_keypair_from_ed25519_priv(ed_raw_priv)
        return ed_priv, ed_pub_hex, pq_priv, pq_pub.hex()

    @staticmethod
    def verify(
        manifest: ThresholdDHTManifest,
        trusted_signers: Dict[str, str],  # pubkey_hex -> name/identity
        threshold_m: int,
        current_seq: int,
        trusted_pq_signers: Optional[Dict[str, str]] = None,  # pq_pubkey_hex -> name/identity
        bootstrap_checkpoint_seq: Optional[int] = None,
        require_pq: bool = True,
        asn_verifier: Optional[AuthoritativeASNVerifier] = None,
        verified_asn_quorum: Optional[Set[int]] = None,
    ) -> Tuple[bool, str]:
        """Verify monotonic sequence number, strict signer uniqueness, and M-of-N Hybrid threshold signatures."""
        # 0. Strict type & policy bounds
        if isinstance(manifest.seq, bool) or not isinstance(manifest.seq, int):
            return False, f"invalid_seq_type:{type(manifest.seq).__name__}"
        if isinstance(manifest.timestamp, bool) or not isinstance(manifest.timestamp, int):
            return False, f"invalid_timestamp_type:{type(manifest.timestamp).__name__}"
        if isinstance(threshold_m, bool) or not isinstance(threshold_m, int) or threshold_m <= 0:
            return False, f"invalid_threshold_m:{threshold_m}"
        if not trusted_signers:
            return False, "empty_trusted_signers"
        if manifest.seq <= 0:
            return False, f"non_positive_seq:{manifest.seq}"

        # 1. Monotonic sequence check (downgrade, replay & huge-seq lockout protection)
        if manifest.seq <= current_seq:
            return False, f"stale_or_replayed_seq:{manifest.seq}<={current_seq}"
        effective_base_seq = int(current_seq)
        if effective_base_seq == 0 and bootstrap_checkpoint_seq is not None and int(bootstrap_checkpoint_seq) > 0:
            if manifest.seq < int(bootstrap_checkpoint_seq):
                return False, f"below_bootstrap_checkpoint_seq:{manifest.seq}<{int(bootstrap_checkpoint_seq)}"
            effective_base_seq = int(bootstrap_checkpoint_seq)
        if manifest.seq > (2**63 - 1) or manifest.seq > effective_base_seq + 1024:
            return False, f"seq_jump_exceeds_max_window:{manifest.seq}"
        # 1b. Manifest timestamp sanity bound (reject far-future or >14d expired timestamps)
        now_ts = int(time.time())
        if (
            manifest.timestamp <= 0
            or (manifest.timestamp - now_ts) > 300
            or (manifest.timestamp > 1792400000 and (now_ts - manifest.timestamp) > 86400 * 14)
        ):
            return False, f"manifest_timestamp_out_of_bounds:{manifest.timestamp}"

        # 2. Threshold format check
        if manifest.bundle_format.lower() not in [b.value for b in BundleFormat]:
            return False, f"disallowed_bundle_format:{manifest.bundle_format}"

        # 3. Strict signer uniqueness check (reject duplicate public keys or reused signatures)
        unique_ok, unique_reason = manifest.check_signer_uniqueness()
        if not unique_ok:
            return False, unique_reason

        # 3b. Canonical lowercase hex enforcement
        if manifest.bundle_sha256 != manifest.bundle_sha256.lower() or manifest.infohash != manifest.infohash.lower():
            return False, "non_canonical_uppercase_hex_in_manifest"
        for raw_pk, raw_sig in manifest.iter_ed25519_signatures():
            if raw_pk != raw_pk.lower() or raw_sig != raw_sig.lower():
                return False, "non_canonical_uppercase_hex_in_signatures"
        for raw_pq_pk, raw_pq_sig in manifest.iter_pq_signatures():
            if raw_pq_pk != raw_pq_pk.lower() or raw_pq_sig != raw_pq_sig.lower():
                return False, "non_canonical_uppercase_hex_in_pq_signatures"

        # Optional signer attestation ASN uniqueness & binding check
        if manifest.signer_attestations is not None:
            q_ok, q_reason, _ = SybilQuorumVerifier.verify_quorum(
                attestations=manifest.signer_attestations,
                target_hash=manifest.bundle_sha256,
                min_distinct_asns=min(threshold_m, manifest.min_asn_quorum),
                require_pq=require_pq,
                asn_verifier=asn_verifier,
                reject_duplicate_pubkeys=True,
            )
            if not q_ok:
                return False, f"manifest_signer_attestation_failed:{q_reason}"

        # 4. Canonical payload derivation
        payload_bytes = manifest.canonical_payload()
        norm_trusted = {
            ThresholdDHTManifest._normalize_pubkey_id(k): v for k, v in trusted_signers.items()
        }

        # 5. Check classical Ed25519 signatures against trusted federation signers
        valid_signatures: Set[str] = set()
        seen_identities: Set[str] = set()
        ed_pk_by_identity: Dict[str, str] = {}
        for pubkey_hex, sig_hex in manifest.iter_ed25519_signatures():
            norm_pk = ThresholdDHTManifest._normalize_pubkey_id(pubkey_hex)
            if norm_pk not in norm_trusted:
                continue

            identity = norm_trusted[norm_pk]
            if identity in seen_identities:
                return False, f"duplicate_signer_identity:{identity}"

            try:
                pk_bytes = bytes.fromhex(norm_pk)
                sig_bytes = bytes.fromhex(sig_hex.strip())
                verifier = ed25519.Ed25519PublicKey.from_public_bytes(pk_bytes)
                verifier.verify(sig_bytes, payload_bytes)
                valid_signatures.add(norm_pk)
                seen_identities.add(identity)
                ed_pk_by_identity[identity] = norm_pk
            except (ValueError, InvalidSignature):
                continue

        if len(valid_signatures) < threshold_m:
            return False, f"insufficient_threshold_signatures:{len(valid_signatures)}<{threshold_m}"

        # 6. Check Post-Quantum WOTS+-SHAKE256 signatures (fail-closed: never trust unpinned self-supplied PQ keys)
        if require_pq or manifest.pq_signatures or trusted_pq_signers:
            if not trusted_pq_signers:
                return False, "untrusted_pq_signers_when_pq_trust_store_unset"
            valid_pq_signatures: Set[str] = set()
            seen_pq_identities: Set[str] = set()
            pq_root_by_identity: Dict[str, str] = {}
            staged_leaves: Dict[str, Tuple[int, str]] = {}
            norm_trusted_pq = {
                ThresholdDHTManifest._normalize_pubkey_id(k): v for k, v in trusted_pq_signers.items()
            }
            for pq_pk_hex, pq_sig_hex in manifest.iter_pq_signatures():
                norm_pq_pk = ThresholdDHTManifest._normalize_pubkey_id(pq_pk_hex)
                meta = WOTSPlusSHAKE256.extract_signature_metadata(norm_pq_pk, pq_sig_hex.strip())
                if meta is None:
                    continue
                root_pq_pk = str(meta["root_pq_pub_hex"]).lower()
                leaf_pq_pk = str(meta["leaf_pub_hex"]).lower()
                if norm_pq_pk not in (root_pq_pk, leaf_pq_pk):
                    continue
                matched_trust_pk = root_pq_pk if root_pq_pk in norm_trusted_pq else (
                    norm_pq_pk if norm_pq_pk in norm_trusted_pq else ""
                )
                if not matched_trust_pk:
                    continue
                pq_id = norm_trusted_pq[matched_trust_pk]
                if pq_id in seen_pq_identities:
                    return False, f"duplicate_pq_signer_identity:{pq_id}"
                prev_use = ThresholdSignatureVerifier._USED_PQ_LEAVES.get(leaf_pq_pk)
                curr_use = (int(manifest.seq), manifest.bundle_sha256.lower())
                if prev_use is not None and prev_use != curr_use:
                    return False, f"wots_one_time_leaf_reused:{leaf_pq_pk[:16]}"
                try:
                    pq_pk_bytes = bytes.fromhex(norm_pq_pk)
                    pq_sig_bytes = bytes.fromhex(pq_sig_hex.strip())
                    if WOTSPlusSHAKE256.verify(pq_pk_bytes, payload_bytes, pq_sig_bytes):
                        valid_pq_signatures.add(matched_trust_pk)
                        seen_pq_identities.add(pq_id)
                        pq_root_by_identity[pq_id] = matched_trust_pk
                        staged_leaves[leaf_pq_pk] = curr_use
                except ValueError:
                    continue

            if len(valid_pq_signatures) < threshold_m:
                return False, f"insufficient_pq_threshold_signatures:{len(valid_pq_signatures)}<{threshold_m}"

            def _base_signer_id(raw_id: str) -> str:
                s = str(raw_id).strip()
                s = re.sub(r"(?i)([_:-])(?:pq|ed)(?=[_:-]|$)", "", s)
                return s

            ed_by_base: Dict[str, Tuple[str, str]] = {
                _base_signer_id(ed_id): (ed_id, ed_pk) for ed_id, ed_pk in ed_pk_by_identity.items()
            }
            paired_identities: Set[str] = set()
            for pq_id, matched_pq_pk in pq_root_by_identity.items():
                base_id = _base_signer_id(pq_id)
                ed_match = (
                    (pq_id, ed_pk_by_identity[pq_id])
                    if pq_id in ed_pk_by_identity
                    else ed_by_base.get(base_id)
                )
                if ed_match is None:
                    continue
                _ed_id, ed_pk = ed_match
                pinned_pq = (
                    AuthoritativeASNVerifier._GLOBAL_PINNED_PUBKEY_PQ_ROOT.get(ed_pk)
                    or WOTSPlusSHAKE256.get_root_wots_pub(ed_pk)
                )
                if pinned_pq and pinned_pq.strip().lower() in norm_trusted_pq:
                    if matched_pq_pk != pinned_pq.strip().lower():
                        continue
                paired_identities.add(base_id)

            if len(paired_identities) < threshold_m:
                return False, f"insufficient_paired_hybrid_threshold_signatures:{len(paired_identities)}<{threshold_m}"

            ThresholdSignatureVerifier._USED_PQ_LEAVES.update(staged_leaves)

        return True, f"threshold_verified:{len(valid_signatures)}_of_{len(trusted_signers)}"


# =====================================================================
# 8. Sybil Quorum Engine (K-of-N Distinct-ASN Quorum)
# =====================================================================

class SybilQuorumVerifier:
    """Guarantees distributed attestations originate from at least K distinct Autonomous System Numbers (ASNs)."""

    _USED_ATTESTATION_PQ_LEAVES: Dict[str, str] = {}

    @staticmethod
    def verify_quorum(
        attestations: List[PeerAttestation],
        target_hash: str,
        min_distinct_asns: int = 5,
        max_age_seconds: int = 86400,
        now_timestamp: Optional[int] = None,
        require_pq: bool = False,
        asn_verifier: Optional[AuthoritativeASNVerifier] = None,
        require_authoritative_asn: Optional[bool] = None,
        reject_duplicate_pubkeys: bool = True,
    ) -> Tuple[bool, str, Set[int]]:
        """Verify attestations against freshness, Hybrid Ed25519 + WOTS+-SHAKE256 signatures, uniqueness, and ASN bindings."""
        if isinstance(min_distinct_asns, bool) or not isinstance(min_distinct_asns, int) or min_distinct_asns <= 0:
            return False, f"invalid_min_distinct_asns:{min_distinct_asns}", set()
        if not attestations:
            return False, "empty_attestations", set()
        now = int(time.time()) if now_timestamp is None else int(now_timestamp)
        verified_asns: Set[int] = set()
        seen_pubkey_asn: Dict[str, int] = {}
        seen_pq_pubkey_asn: Dict[str, int] = {}
        seen_peer_id_asn: Dict[str, int] = {}
        seen_subnet_asn: Dict[str, int] = {}
        eff_req_auth = (
            bool(asn_verifier.require_authoritative_binding)
            if (require_authoritative_asn is None and asn_verifier is not None)
            else bool(require_authoritative_asn)
        )
        req_pinned = bool(require_authoritative_asn is True)
        active_verifier = asn_verifier or AuthoritativeASNVerifier(
            require_authoritative_binding=eff_req_auth
        )

        first_rejection_reason: str = ""
        for att in attestations:
            if (
                isinstance(att.timestamp, bool)
                or not isinstance(att.timestamp, int)
                or isinstance(att.asn, bool)
                or not isinstance(att.asn, int)
            ):
                continue
            norm_pk = AuthoritativeASNVerifier.normalize_pubkey(att.peer_pubkey_hex)
            claimed_asn = int(att.asn)
            if claimed_asn <= 0:
                continue

            # 1. Check freshness first (allow at most 300s future clock skew and max_age_seconds past age)
            if (att.timestamp - now) > 300 or (now - att.timestamp) > max_age_seconds:
                continue

            # 2. Verify peer Ed25519 signature over target hash BEFORE recording any identity/subnet state
            # so an unauthenticated frame cannot poison seen_pubkey_asn / seen_subnet_asn or veto a quorum
            expected_bytes = att.canonical_bytes(target_hash)
            try:
                pk_bytes = bytes.fromhex(norm_pk)
                sig_bytes = bytes.fromhex(att.signature_hex.strip())
                verifier = ed25519.Ed25519PublicKey.from_public_bytes(pk_bytes)
                verifier.verify(sig_bytes, expected_bytes)
            except (ValueError, InvalidSignature):
                continue

            # 3. Verify Post-Quantum WOTS+-SHAKE256 peer attestation signature if present or required
            eff_pq_id: Optional[str] = None
            leaf_att_pk: Optional[str] = None
            att_digest_hex: Optional[str] = None
            if require_pq or att.pq_signature_hex or att.pq_pubkey_hex:
                if not att.pq_pubkey_hex.strip() or not att.pq_signature_hex.strip():
                    continue
                try:
                    norm_pq_pk = AuthoritativeASNVerifier.normalize_pubkey(att.pq_pubkey_hex)
                    meta_att = WOTSPlusSHAKE256.extract_signature_metadata(
                        att.pq_pubkey_hex.strip().lower(), att.pq_signature_hex.strip()
                    )
                    if meta_att is None:
                        continue
                    att_root_pq = str(meta_att["root_pq_pub_hex"]).lower()
                    leaf_att_pk = str(meta_att["leaf_pub_hex"]).lower()
                    eff_pq_id = att_root_pq if att_root_pq else norm_pq_pk
                    pinned_pq_root = (
                        active_verifier.pinned_pubkey_pq_root.get(norm_pk)
                        or AuthoritativeASNVerifier._GLOBAL_PINNED_PUBKEY_PQ_ROOT.get(norm_pk)
                        or WOTSPlusSHAKE256.get_root_wots_pub(norm_pk)
                    )
                    if pinned_pq_root:
                        if att_root_pq != pinned_pq_root.strip().lower():
                            continue
                    elif not (
                        WOTSPlusSHAKE256.is_bound_to_ed25519_pub(norm_pk, att_root_pq)
                        or WOTSPlusSHAKE256.is_bound_to_ed25519_pub(norm_pk, leaf_att_pk)
                    ):
                        continue
                    att_digest_hex = hashlib.sha256(expected_bytes).hexdigest()
                    pq_pk_bytes = bytes.fromhex(att.pq_pubkey_hex.strip().lower())
                    pq_sig_bytes = bytes.fromhex(att.pq_signature_hex.strip())
                    if not WOTSPlusSHAKE256.verify(pq_pk_bytes, expected_bytes, pq_sig_bytes):
                        continue
                    prev_att_digest = SybilQuorumVerifier._USED_ATTESTATION_PQ_LEAVES.get(leaf_att_pk)
                    if prev_att_digest is not None and prev_att_digest != att_digest_hex:
                        return False, f"wots_one_time_leaf_reused:{leaf_att_pk[:16]}", set()
                except ValueError:
                    continue

            # 4. Validate unallocated ASN / IP presence and authoritative ASN binding on the cryptographically signed attestation
            if (
                claimed_asn > 65534
                and claimed_asn not in AuthoritativeASNVerifier.DEFAULT_ASN_CIDR_ROUTES.values()
                and (att.ip_address is None or str(att.ip_address).strip() == "")
                and norm_pk not in active_verifier.pinned_pubkey_asn
            ):
                if not first_rejection_reason:
                    first_rejection_reason = f"unverified_unallocated_asn_without_ip_or_pin:{claimed_asn}"
                continue

            subnet: Optional[str] = None
            if att.ip_address is not None and str(att.ip_address).strip() != "":
                try:
                    subnet = AuthoritativeASNVerifier.extract_subnet_prefix(att.ip_address)
                except ValueError:
                    if not first_rejection_reason:
                        first_rejection_reason = f"invalid_attestation_ip:{att.ip_address}"
                    continue
                if subnet in seen_subnet_asn and seen_subnet_asn[subnet] != claimed_asn:
                    return (
                        False,
                        f"sybil_spoof_rejected:subnet_claimed_multiple_asns:{subnet}:{seen_subnet_asn[subnet]}!={claimed_asn}",
                        set(),
                    )
                seen_subnet_asn[subnet] = claimed_asn

            if asn_verifier is not None or eff_req_auth or att.ip_address is not None:
                bind_ok, bind_reason = active_verifier.verify_binding(
                    claimed_asn=claimed_asn,
                    pubkey_hex=norm_pk,
                    ip_address=att.ip_address,
                    require_authoritative=eff_req_auth if asn_verifier is None else None,
                    require_pinned_key=req_pinned,
                )
                if not bind_ok:
                    if not first_rejection_reason:
                        first_rejection_reason = f"asn_binding_failed:{bind_reason}"
                    continue

            # 5. Strict public-key, PQ-key, peer_id, and subnet uniqueness & multi-ASN spoofing check (authenticated only)
            if norm_pk in seen_pubkey_asn:
                if seen_pubkey_asn[norm_pk] != claimed_asn:
                    return (
                        False,
                        f"sybil_spoof_rejected:pubkey_claimed_multiple_asns:{seen_pubkey_asn[norm_pk]}!={claimed_asn}",
                        set(),
                    )
                if reject_duplicate_pubkeys:
                    return False, f"duplicate_pubkey_rejected:{norm_pk[:16]}", set()
            seen_pubkey_asn[norm_pk] = claimed_asn

            if eff_pq_id:
                if eff_pq_id in seen_pq_pubkey_asn:
                    if seen_pq_pubkey_asn[eff_pq_id] != claimed_asn:
                        return (
                            False,
                            f"sybil_spoof_rejected:pq_pubkey_claimed_multiple_asns:{seen_pq_pubkey_asn[eff_pq_id]}!={claimed_asn}",
                            set(),
                        )
                    if reject_duplicate_pubkeys:
                        return False, f"duplicate_pq_pubkey_rejected:{eff_pq_id[:16]}", set()
                seen_pq_pubkey_asn[eff_pq_id] = claimed_asn

            if att.peer_id:
                if att.peer_id in seen_peer_id_asn:
                    if seen_peer_id_asn[att.peer_id] != claimed_asn:
                        return (
                            False,
                            f"sybil_spoof_rejected:peer_id_claimed_multiple_asns:{seen_peer_id_asn[att.peer_id]}!={claimed_asn}",
                            set(),
                        )
                    if reject_duplicate_pubkeys:
                        return False, f"duplicate_peer_id_rejected:{att.peer_id}", set()
                seen_peer_id_asn[att.peer_id] = claimed_asn

            if subnet is not None:
                if subnet in seen_subnet_asn and seen_subnet_asn[subnet] != claimed_asn:
                    return (
                        False,
                        f"sybil_spoof_rejected:subnet_claimed_multiple_asns:{subnet}:{seen_subnet_asn[subnet]}!={claimed_asn}",
                        set(),
                    )
                seen_subnet_asn[subnet] = claimed_asn

            if leaf_att_pk is not None and att_digest_hex is not None:
                SybilQuorumVerifier._USED_ATTESTATION_PQ_LEAVES[leaf_att_pk] = att_digest_hex

            # Valid attestation from verified ASN
            if claimed_asn > 0:
                verified_asns.add(claimed_asn)

        if len(verified_asns) < min_distinct_asns:
            if first_rejection_reason and not verified_asns:
                return False, first_rejection_reason, verified_asns
            return False, f"insufficient_distinct_asns:{len(verified_asns)}<{min_distinct_asns}", verified_asns

        return True, f"quorum_achieved:{len(verified_asns)}_distinct_asns", verified_asns


# =====================================================================
# 9. Safe Format Enforcer (Poison-Proof)
# =====================================================================

class SafeFormatValidator:
    """Enforces that candidate updates contain only safe formats (SafeTensors, ONNX, JSON). Rejects pickle."""

    PROHIBITED_BYTE_PATTERNS = [
        b"\x80\x01",  # Python pickle protocol 1
        b"\x80\x02",  # Python pickle protocol 2
        b"\x80\x03",  # Python pickle protocol 3
        b"\x80\x04",  # Python pickle protocol 4
        b"\x80\x05",  # Python pickle protocol 5
        b"c__builtin__",
        b"cbuiltins\n",
        b"cposix\n",
        b"cos\n",
        b"csubprocess\n",
        b"csys\n",
        b"cshutil\n",
        b"csocket\n",
        b"cctypes\n",
        b"cmarshal\n",
        b"cimportlib\n",
        b"__reduce__",
        b"__import__",
    ]
    RE_PICKLE_PROTO0_GLOBAL = re.compile(rb"(?:^|[\x00-\x7f])c[a-zA-Z_][a-zA-Z0-9_.]*\n[a-zA-Z_][a-zA-Z0-9_]*\n")
    DTYPE_BYTE_SIZES = {
        "F32": 4, "I32": 4, "F16": 2, "BF16": 2, "I64": 8, "I8": 1, "U8": 1, "BOOL": 1,
    }

    MAX_JSON_BUNDLE_BYTES = 2 * 1024 * 1024
    MAX_BINARY_BUNDLE_BYTES = 64 * 1024 * 1024

    @classmethod
    def validate(cls, bundle_bytes: bytes, format_name: str) -> Tuple[bool, str]:
        """Validate that bundle conforms to declared safe format and contains no executable bytecode."""
        fmt = format_name.lower().strip()
        if fmt not in ("safetensors", "onnx", "json", "json_calibration"):
            return False, f"unsupported_format:{fmt}"

        if fmt in ("json", "json_calibration") and len(bundle_bytes) > cls.MAX_JSON_BUNDLE_BYTES:
            return False, f"json_bundle_exceeds_max_size:{len(bundle_bytes)}>{cls.MAX_JSON_BUNDLE_BYTES}"
        if fmt in ("safetensors", "onnx") and len(bundle_bytes) > cls.MAX_BINARY_BUNDLE_BYTES:
            return False, f"binary_bundle_exceeds_max_size:{len(bundle_bytes)}>{cls.MAX_BINARY_BUNDLE_BYTES}"

        # 1. Check for prohibited pickle bytecode patterns (for ONNX/SafeTensors, do not scan raw numeric weight buffers for pickle opcodes/substrings)
        if fmt == "onnx":
            if bundle_bytes[:2] in (b"\x80\x01", b"\x80\x02", b"\x80\x03", b"\x80\x04", b"\x80\x05"):
                return False, "poison_detected:prohibited_bytecode_pattern"
        elif fmt == "safetensors":
            if bundle_bytes[:2] in (b"\x80\x01", b"\x80\x02", b"\x80\x03", b"\x80\x04", b"\x80\x05"):
                return False, "poison_detected:prohibited_bytecode_pattern"
            for ascii_pat in (
                b"csubprocess\n",
                b"c__builtin__",
                b"cbuiltins\n",
                b"cposix\n",
                b"__reduce__",
                b"__import__",
                b"\x80\x04evil",
                b"\x80\x04\x95",
            ):
                if ascii_pat in bundle_bytes:
                    return False, "poison_detected:prohibited_bytecode_pattern"
        else:
            for pattern in cls.PROHIBITED_BYTE_PATTERNS:
                if pattern in bundle_bytes:
                    return False, "poison_detected:prohibited_bytecode_pattern"
            if cls.RE_PICKLE_PROTO0_GLOBAL.search(bundle_bytes):
                return False, "poison_detected:prohibited_pickle_proto0_global"

        # 2. Format-specific structural validation
        if fmt == "safetensors":
            return cls._validate_safetensors(bundle_bytes)
        elif fmt == "onnx":
            return cls._validate_onnx(bundle_bytes)
        elif fmt in ("json", "json_calibration"):
            return cls._validate_json(bundle_bytes)

        return False, "unknown_validation_error"

    @classmethod
    def _validate_safetensors(cls, data: bytes) -> Tuple[bool, str]:
        if len(data) < 8:
            return False, "safetensors_too_short"
        header_len = int.from_bytes(data[:8], "little")
        if header_len <= 0 or header_len > 10 * 1024 * 1024 or 8 + header_len > len(data):
            return False, "safetensors_invalid_header_length"

        header_bytes = data[8:8 + header_len]
        for pattern in cls.PROHIBITED_BYTE_PATTERNS:
            if pattern in header_bytes:
                return False, "poison_detected:prohibited_bytecode_pattern"
        if cls.RE_PICKLE_PROTO0_GLOBAL.search(header_bytes):
            return False, "poison_detected:prohibited_pickle_proto0_global"

        try:
            header_json = json.loads(header_bytes.decode("utf-8"))
            if not isinstance(header_json, dict):
                return False, "safetensors_header_not_dict"

            total_buffer_len = len(data) - 8 - header_len
            intervals: List[Tuple[int, int]] = []
            tensor_count = 0

            for tensor_name, tensor_meta in header_json.items():
                if tensor_name == "__metadata__":
                    if not isinstance(tensor_meta, dict):
                        return False, "safetensors_invalid_metadata_type"
                    for mk, mv in tensor_meta.items():
                        if not isinstance(mk, str) or not isinstance(mv, str):
                            return False, "safetensors_metadata_non_string_value"
                    continue
                tensor_count += 1
                if not isinstance(tensor_meta, dict):
                    return False, f"safetensors_invalid_tensor_meta:{tensor_name}"
                dtype = tensor_meta.get("dtype")
                if dtype not in cls.DTYPE_BYTE_SIZES:
                    return False, f"safetensors_invalid_dtype:{dtype}"
                offsets = tensor_meta.get("data_offsets")
                if (
                    not isinstance(offsets, list)
                    or len(offsets) != 2
                    or isinstance(offsets[0], bool)
                    or isinstance(offsets[1], bool)
                    or not isinstance(offsets[0], int)
                    or not isinstance(offsets[1], int)
                ):
                    return False, f"safetensors_invalid_offsets:{tensor_name}"
                if offsets[0] < 0 or offsets[1] < offsets[0] or offsets[1] > total_buffer_len:
                    return False, f"safetensors_offset_out_of_bounds:{tensor_name}"

                shape = tensor_meta.get("shape")
                if not isinstance(shape, list) or any(isinstance(dim, bool) or not isinstance(dim, int) or dim < 0 for dim in shape):
                    return False, f"safetensors_invalid_shape:{tensor_name}"
                numel = 1
                for dim in shape:
                    numel *= dim
                expected_bytes = numel * cls.DTYPE_BYTE_SIZES[dtype]
                if (offsets[1] - offsets[0]) != expected_bytes:
                    return False, f"safetensors_shape_byte_size_mismatch:{tensor_name}"
                intervals.append((offsets[0], offsets[1]))

            if tensor_count == 0:
                return False, "safetensors_empty_tensor_map"

            intervals.sort()
            for idx in range(1, len(intervals)):
                if intervals[idx][0] < intervals[idx - 1][1]:
                    return False, "safetensors_overlapping_tensor_buffers"

            return True, "safetensors_valid"
        except Exception as e:
            return False, f"safetensors_parse_error:{e}"

    @staticmethod
    def _validate_onnx(data: bytes) -> Tuple[bool, str]:
        if len(data) < 16:
            return False, "onnx_too_short"
        # ONNX ModelProto starts with field 1 (ir_version, tag 0x08) with valid ONNX IR version 1..12
        if data[0] != 0x08 or not (1 <= data[1] <= 12):
            return False, "onnx_invalid_protobuf_magic"
        if b"../../" in data or b"..\\..\\" in data or b"/etc/" in data or b"#!/bin/" in data:
            return False, "onnx_external_data_or_polyglot_rejected"
        try:
            import onnx
            model = onnx.ModelProto()
            parsed_len = model.ParseFromString(data)
            if parsed_len is not None and parsed_len != len(data):
                return False, "onnx_trailing_polyglot_bytes"
            if not (1 <= int(model.ir_version) <= 12):
                return False, f"onnx_invalid_ir_version:{model.ir_version}"
            if len(model.opset_import) == 0 or len(model.graph.node) == 0:
                return False, "onnx_empty_graph_or_missing_opset"

            def _check_tensor(t: Any) -> Optional[str]:
                if int(getattr(t, "data_location", 0)) != 0 or len(getattr(t, "external_data", ())) > 0:
                    return f"onnx_external_data_forbidden:{getattr(t, 'name', '')}"
                return None

            def _check_graph(g: Any, depth: int = 0) -> Optional[str]:
                if depth > 16:
                    return "onnx_subgraph_nesting_exceeded"
                for init in g.initializer:
                    err = _check_tensor(init)
                    if err:
                        return err
                for sp in getattr(g, "sparse_initializer", ()):
                    for sub_t in (getattr(sp, "values", None), getattr(sp, "indices", None)):
                        if sub_t is not None:
                            err = _check_tensor(sub_t)
                            if err:
                                return err
                for node in g.node:
                    if node.domain not in ("", "ai.onnx", "ai.onnx.ml"):
                        return f"onnx_custom_operator_domain_forbidden:{node.domain}"
                    for attr in node.attribute:
                        if attr.HasField("t"):
                            err = _check_tensor(attr.t)
                            if err:
                                return err
                        for sub_t in getattr(attr, "tensors", ()):
                            err = _check_tensor(sub_t)
                            if err:
                                return err
                        if hasattr(attr, "sparse_tensor") and attr.HasField("sparse_tensor"):
                            for sub_t in (attr.sparse_tensor.values, attr.sparse_tensor.indices):
                                err = _check_tensor(sub_t)
                                if err:
                                    return err
                        for sp in getattr(attr, "sparse_tensors", ()):
                            for sub_t in (getattr(sp, "values", None), getattr(sp, "indices", None)):
                                if sub_t is not None:
                                    err = _check_tensor(sub_t)
                                    if err:
                                        return err
                        if attr.HasField("g"):
                            err = _check_graph(attr.g, depth + 1)
                            if err:
                                return err
                        for sub_g in getattr(attr, "graphs", ()):
                            err = _check_graph(sub_g, depth + 1)
                            if err:
                                return err
                return None

            graph_err = _check_graph(model.graph)
            if graph_err:
                return False, graph_err
        except Exception as exc:
            return False, f"onnx_protobuf_parse_error:{type(exc).__name__}"
        return True, "onnx_valid"

    @staticmethod
    def _validate_json(data: bytes) -> Tuple[bool, str]:
        try:
            parsed = json.loads(data.decode("utf-8"))
            if not isinstance(parsed, dict):
                return False, "json_not_dictionary"
            return True, "json_valid"
        except Exception as e:
            return False, f"json_parse_error:{e}"


class SubnetDiversityLimiter:
    """Enforces per-subnet (/24 IPv4 and /48 IPv6) vote caps with ::ffff:0:0/96 IPv4-mapped IPv6 normalization."""

    def __init__(self, max_per_ipv4_24: int = 2, max_per_ipv6_48: int = 2) -> None:
        self.max_per_ipv4_24 = int(max_per_ipv4_24)
        self.max_per_ipv6_48 = int(max_per_ipv6_48)
        self._counts: Dict[str, int] = {}

    @staticmethod
    def normalize_subnet(ip_address: str) -> str:
        return AuthoritativeASNVerifier.extract_subnet_prefix(ip_address)

    def allow_vote(self, ip_address: str) -> Tuple[bool, str]:
        subnet = self.normalize_subnet(ip_address)
        cap = self.max_per_ipv4_24 if subnet.endswith("/24") else self.max_per_ipv6_48
        cur = self._counts.get(subnet, 0)
        if cur >= cap:
            return False, f"subnet_diversity_limit_exceeded:{subnet}:{cur}>={cap}"
        self._counts[subnet] = cur + 1
        return True, f"allowed:{subnet}:{cur + 1}/{cap}"





class AcyclicProvenanceVerifier:
    """Verifies TUF / in-toto acyclic provenance separation across contributors, evaluators, and release signers."""

    @staticmethod
    def sign_benchmark_attestation(
        *,
        peer_id: str,
        asn: int,
        parent_bundle_sha256: str,
        candidate_bundle_sha256: str,
        canary_suite_sha256: str,
        canary_metrics: Mapping[str, Union[float, bool]],
        evaluator_version: str,
        timestamp: int,
        ed25519_priv: ed25519.Ed25519PrivateKey,
        wots_priv_seed: Optional[bytes] = None,
        ip_address: Optional[str] = None,
    ) -> BenchmarkAttestationPredicate:
        from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat, PrivateFormat, NoEncryption
        ed_pub_hex = ed25519_priv.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
        if wots_priv_seed is None:
            raw_ed = ed25519_priv.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())
            wots_priv_seed, wots_pub_bytes = WOTSPlusSHAKE256.derive_keypair_from_ed25519_priv(raw_ed)
        else:
            wots_priv_seed, wots_pub_bytes = WOTSPlusSHAKE256._normalize_root_priv(wots_priv_seed)
        if not isinstance(canary_metrics, Mapping) or not canary_metrics:
            raise ValueError("missing_canary_metrics")
        if "passed" not in canary_metrics or not isinstance(canary_metrics["passed"], bool):
            raise ValueError("missing_or_invalid_passed_flag")
        raw_acc = canary_metrics.get("accuracy", canary_metrics.get("category_accuracy"))
        if isinstance(raw_acc, bool) or not isinstance(raw_acc, (int, float)) or not math.isfinite(float(raw_acc)) or not (0.0 <= float(raw_acc) <= 1.0):
            raise ValueError("missing_or_invalid_accuracy")
        for req_key in ("ece", "macro_f1", "mondrian_coverage", "max_false_accept_rate"):
            if req_key not in canary_metrics:
                raise ValueError(f"missing_canary_metric:{req_key}")
            req_val = canary_metrics[req_key]
            if isinstance(req_val, bool) or not isinstance(req_val, (int, float)) or not math.isfinite(float(req_val)) or not (0.0 <= float(req_val) <= 1.0):
                raise ValueError(f"invalid_canary_metric:{req_key}")
        clean_metrics: Dict[str, Union[float, bool]] = {}
        for k, v in canary_metrics.items():
            key_str = str(k)
            if isinstance(v, bool):
                if key_str != "passed":
                    raise ValueError(f"unexpected_boolean_canary_metric:{key_str}")
                clean_metrics[key_str] = bool(v)
            elif isinstance(v, (int, float)) and math.isfinite(float(v)):
                clean_metrics[key_str] = float(v)
            else:
                raise ValueError(f"invalid_canary_metric:{key_str}")
        unsigned = BenchmarkAttestationPredicate(
            peer_id=peer_id,
            asn=int(asn),
            parent_bundle_sha256=parent_bundle_sha256.strip().lower(),
            candidate_bundle_sha256=candidate_bundle_sha256.strip().lower(),
            canary_suite_sha256=canary_suite_sha256.strip().lower(),
            canary_metrics=clean_metrics,
            evaluator_version=evaluator_version,
            timestamp=int(timestamp),
            peer_pubkey_hex=ed_pub_hex,
            signature_hex="",
            pq_pubkey_hex=wots_pub_bytes.hex(),
            pq_signature_hex="",
            ip_address=ip_address,
        )
        pae = unsigned.pae_bytes()
        ed_sig = ed25519_priv.sign(pae).hex()
        pq_sig = WOTSPlusSHAKE256.sign(wots_priv_seed, pae).hex()
        return dataclasses.replace(unsigned, signature_hex=ed_sig, pq_signature_hex=pq_sig)

    @staticmethod
    def sign_release_envelope(
        envelope: AcyclicReleaseEnvelope,
        release_signers: Sequence[Tuple[ed25519.Ed25519PrivateKey, str, bytes, str]],
    ) -> AcyclicReleaseEnvelope:
        """Sign the DSSE PAE of an AcyclicReleaseEnvelope with M-of-N hybrid release signers."""
        pae = envelope.pae_bytes()
        env_sigs: Dict[str, str] = {}
        env_pq_sigs: Dict[str, str] = {}
        for r_priv, r_pub, r_wots_priv, r_wots_pub in release_signers:
            norm_pub = ThresholdDHTManifest._normalize_pubkey_id(r_pub)
            norm_pq = ThresholdDHTManifest._normalize_pubkey_id(r_wots_pub)
            env_sigs[norm_pub] = r_priv.sign(pae).hex()
            env_pq_sigs[norm_pq] = WOTSPlusSHAKE256.sign(r_wots_priv, pae).hex()
        return dataclasses.replace(
            envelope,
            envelope_signatures=env_sigs,
            envelope_pq_signatures=env_pq_sigs,
        )

    @staticmethod
    def verify_envelope(
        envelope: AcyclicReleaseEnvelope,
        *,
        trusted_release_signers: Dict[str, str],
        trusted_evaluator_signers: Dict[str, str],
        threshold_m: int = 2,
        min_evaluator_asns: int = 3,
        current_seq: int = 0,
        bootstrap_checkpoint_seq: Optional[int] = None,
        require_pq: bool = True,
        asn_verifier: Optional[AuthoritativeASNVerifier] = None,
        enforce_evaluator_release_disjoint: bool = True,
        trusted_release_pq_signers: Optional[Mapping[str, str]] = None,
        trusted_evaluator_pq_signers: Optional[Mapping[str, str]] = None,
        trusted_contributor_signers: Optional[Mapping[str, str]] = None,
    ) -> Tuple[bool, str]:
        """Verify disjoint contributor / evaluator / release-quorum roles, evaluator ASN bindings, and DSSE PAE release signatures."""
        if not trusted_release_signers:
            return False, "empty_trusted_release_signers"
        if not trusted_evaluator_signers:
            return False, "empty_trusted_evaluator_signers"

        contributors = {
            ThresholdDHTManifest._normalize_pubkey_id(pk)
            for pk in envelope.contributor_pubkeys
            if str(pk).strip() and len(ThresholdDHTManifest._normalize_pubkey_id(pk)) == 64
        }
        if not contributors:
            return False, "missing_contributor_pubkeys"

        manifest_release_signers = {
            ThresholdDHTManifest._normalize_pubkey_id(pk)
            for pk, _ in envelope.manifest.iter_ed25519_signatures()
        }
        envelope_release_signers = {
            ThresholdDHTManifest._normalize_pubkey_id(pk)
            for pk in (envelope.envelope_signatures or {}).keys()
        }
        release_signers = manifest_release_signers | envelope_release_signers
        evaluator_signers = {
            ThresholdDHTManifest._normalize_pubkey_id(att.peer_pubkey_hex)
            for att in envelope.benchmark_attestations
        }

        if contributors & evaluator_signers:
            return False, "cyclic_provenance_violation:contributor_in_evaluators"
        if contributors & release_signers:
            return False, "cyclic_provenance_violation:contributor_in_release_signers"
        if enforce_evaluator_release_disjoint and (evaluator_signers & release_signers):
            return False, "cyclic_provenance_violation:evaluator_in_release_signers"

        if envelope.authenticated_baseline is None:
            return False, "missing_authenticated_baseline"
        b_ok, b_reason = envelope.authenticated_baseline.verify_signature(trusted_release_signers)
        if not b_ok:
            return False, f"baseline_verification_failed:{b_reason}"
        if envelope.authenticated_baseline.parent_bundle_sha256 != envelope.parent_bundle_sha256.strip().lower():
            return False, "baseline_parent_bundle_mismatch"
        if envelope.authenticated_baseline.canary_suite_sha256 != envelope.canary_suite_sha256.strip().lower():
            return False, "baseline_canary_suite_mismatch"

        eff_asn_verifier = asn_verifier if asn_verifier is not None else AuthoritativeASNVerifier(require_authoritative_binding=True)

        norm_eval_trust: Dict[str, str] = {}
        eval_pq_by_ed: Dict[str, Set[str]] = {}
        eval_id_by_pq: Dict[str, str] = {}
        for k, v in trusted_evaluator_signers.items():
            norm_k = ThresholdDHTManifest._normalize_pubkey_id(k)
            if not norm_k:
                continue
            norm_eval_trust[norm_k] = v
            raw_k = str(k).strip().lower()
            if ":" in raw_k:
                _, pq_part = raw_k.split(":", 1)
                norm_pq_part = ThresholdDHTManifest._normalize_pubkey_id(pq_part)
                if len(norm_pq_part) == 64:
                    eval_pq_by_ed.setdefault(norm_k, set()).add(norm_pq_part)
                    eval_id_by_pq[norm_pq_part] = v
        if trusted_evaluator_pq_signers:
            for pq_k, pq_v in trusted_evaluator_pq_signers.items():
                norm_pq_k = ThresholdDHTManifest._normalize_pubkey_id(pq_k)
                if len(norm_pq_k) == 64:
                    eval_id_by_pq[norm_pq_k] = str(pq_v)
        if not norm_eval_trust:
            return False, "empty_trusted_evaluator_signers"

        eff_pq_signers: Dict[str, str] = {}
        if trusted_release_pq_signers:
            for pq_k, pq_id in trusted_release_pq_signers.items():
                norm_pq_k = ThresholdDHTManifest._normalize_pubkey_id(pq_k)
                if len(norm_pq_k) == 64:
                    eff_pq_signers[norm_pq_k] = str(pq_id)
        norm_rel_trust: Dict[str, str] = {}
        for ed_pk_hex, identity in trusted_release_signers.items():
            norm_ed = ThresholdDHTManifest._normalize_pubkey_id(ed_pk_hex)
            norm_rel_trust[norm_ed] = identity
            raw_ed_str = str(ed_pk_hex).strip().lower()
            if ":" in raw_ed_str:
                _, pq_part = raw_ed_str.split(":", 1)
                norm_pq_part = ThresholdDHTManifest._normalize_pubkey_id(pq_part)
                if len(norm_pq_part) == 64:
                    eff_pq_signers[norm_pq_part] = identity
            pinned_rel_pq = (
                eff_asn_verifier.pinned_pubkey_pq_root.get(norm_ed)
                or AuthoritativeASNVerifier._GLOBAL_PINNED_PUBKEY_PQ_ROOT.get(norm_ed)
                or WOTSPlusSHAKE256.get_root_wots_pub(norm_ed)
            )
            if pinned_rel_pq and len(ThresholdDHTManifest._normalize_pubkey_id(pinned_rel_pq)) == 64:
                eff_pq_signers[ThresholdDHTManifest._normalize_pubkey_id(pinned_rel_pq)] = identity
            for wots_pk_hex in WOTSPlusSHAKE256._ED25519_TO_WOTS_PUBS.get(norm_ed, ()):
                eff_pq_signers[wots_pk_hex] = identity

        norm_contrib_trust: Dict[str, str] = {}
        if trusted_contributor_signers:
            for c_k, c_id in trusted_contributor_signers.items():
                norm_ck = ThresholdDHTManifest._normalize_pubkey_id(c_k)
                if len(norm_ck) == 64:
                    norm_contrib_trust[norm_ck] = str(c_id)
                raw_ck = str(c_k).strip().lower()
                if ":" in raw_ck:
                    _, pq_part = raw_ck.split(":", 1)
                    norm_pq_part = ThresholdDHTManifest._normalize_pubkey_id(pq_part)
                    if len(norm_pq_part) == 64:
                        norm_contrib_trust[norm_pq_part] = str(c_id)

        contributor_ids: Set[str] = set()
        for c_pk in contributors:
            for mapping in (norm_contrib_trust, norm_eval_trust, eval_id_by_pq, norm_rel_trust, eff_pq_signers):
                mapped_id = mapping.get(c_pk)
                if mapped_id is not None:
                    contributor_ids.add(str(mapped_id))

        verified_eval_asns: Set[int] = set()
        seen_eval_pks: Set[str] = set()
        seen_eval_ids: Set[str] = set()
        subnet_limiter = SubnetDiversityLimiter(max_per_ipv4_24=1, max_per_ipv6_48=1)

        for att in envelope.benchmark_attestations:
            norm_pk = ThresholdDHTManifest._normalize_pubkey_id(att.peer_pubkey_hex)
            if norm_pk in seen_eval_pks:
                return False, f"duplicate_evaluator_pubkey:{norm_pk[:16]}"
            seen_eval_pks.add(norm_pk)
            if norm_pk not in norm_eval_trust:
                return False, f"untrusted_evaluator_pubkey:{norm_pk[:16]}"
            eval_id = norm_eval_trust[norm_pk]
            if eval_id in contributor_ids:
                return False, f"cyclic_provenance_violation:contributor_identity_in_evaluators:{eval_id}"
            if eval_id in seen_eval_ids:
                return False, f"duplicate_evaluator_identity:{eval_id}"
            seen_eval_ids.add(eval_id)

            if att.parent_bundle_sha256.strip().lower() != envelope.parent_bundle_sha256.strip().lower():
                return False, "attestation_parent_bundle_mismatch"
            if att.candidate_bundle_sha256.strip().lower() != envelope.manifest.bundle_sha256.strip().lower():
                return False, "attestation_candidate_bundle_mismatch"
            if att.canary_suite_sha256.strip().lower() != envelope.canary_suite_sha256.strip().lower():
                return False, "attestation_canary_suite_mismatch"

            metrics = att.canary_metrics
            if not isinstance(metrics, Mapping) or not metrics:
                return False, "attestation_missing_canary_metrics"
            if "passed" not in metrics:
                return False, "attestation_missing_passed_flag"
            if not isinstance(metrics["passed"], bool) or metrics["passed"] is not True:
                return False, "attestation_canary_marked_failed"
            for m_k, m_v in metrics.items():
                if isinstance(m_v, bool):
                    if str(m_k) != "passed":
                        return False, f"attestation_unexpected_boolean_metric:{m_k}"
                elif not isinstance(m_v, (int, float)) or not math.isfinite(float(m_v)):
                    return False, f"attestation_invalid_metric_value:{m_k}"
            raw_acc = metrics.get("accuracy", metrics.get("category_accuracy"))
            if isinstance(raw_acc, bool) or not isinstance(raw_acc, (int, float)) or not math.isfinite(float(raw_acc)):
                return False, "attestation_missing_or_invalid_accuracy"
            acc_val = float(raw_acc)
            if not (0.0 <= acc_val <= 1.0):
                return False, "attestation_accuracy_out_of_range"
            if acc_val + 1e-9 < float(envelope.authenticated_baseline.accuracy) - 0.005:
                return False, "attestation_accuracy_regression_exceeded"
            for m_key in ("ece", "macro_f1", "mondrian_coverage", "max_false_accept_rate"):
                if m_key not in metrics:
                    return False, f"attestation_missing_{m_key}"
                m_raw = metrics[m_key]
                if isinstance(m_raw, bool) or not isinstance(m_raw, (int, float)) or not math.isfinite(float(m_raw)) or not (0.0 <= float(m_raw) <= 1.0):
                    return False, f"attestation_invalid_{m_key}"
            if float(metrics["macro_f1"]) + 1e-9 < float(envelope.authenticated_baseline.macro_f1) - 0.005:
                return False, "attestation_macro_f1_regression_exceeded"
            if float(metrics["ece"]) - 1e-9 > float(envelope.authenticated_baseline.ece) + 0.02:
                return False, "attestation_ece_regression_exceeded"
            if float(metrics["mondrian_coverage"]) + 1e-9 < float(envelope.authenticated_baseline.mondrian_coverage):
                return False, "attestation_mondrian_coverage_below_threshold"
            if float(metrics["max_false_accept_rate"]) - 1e-9 > float(envelope.authenticated_baseline.max_false_accept_rate):
                return False, "attestation_false_accept_rate_exceeded"

            pae = att.pae_bytes()
            try:
                vk = ed25519.Ed25519PublicKey.from_public_bytes(bytes.fromhex(norm_pk))
                vk.verify(bytes.fromhex(att.signature_hex.strip()), pae)
            except Exception as exc:
                return False, f"invalid_evaluator_ed25519_signature:{exc}"

            if require_pq:
                if not att.pq_pubkey_hex or not att.pq_signature_hex:
                    return False, "missing_evaluator_pq_signature"
                norm_eval_pq = ThresholdDHTManifest._normalize_pubkey_id(att.pq_pubkey_hex)
                bound_wots = set(WOTSPlusSHAKE256._ED25519_TO_WOTS_PUBS.get(norm_pk, set()))
                bound_wots.update(eval_pq_by_ed.get(norm_pk, set()))
                pinned_eval_pq = (
                    eff_asn_verifier.pinned_pubkey_pq_root.get(norm_pk)
                    or AuthoritativeASNVerifier._GLOBAL_PINNED_PUBKEY_PQ_ROOT.get(norm_pk)
                    or WOTSPlusSHAKE256.get_root_wots_pub(norm_pk)
                )
                if pinned_eval_pq:
                    bound_wots.add(ThresholdDHTManifest._normalize_pubkey_id(pinned_eval_pq))
                meta_eval = WOTSPlusSHAKE256.extract_signature_metadata(norm_eval_pq, att.pq_signature_hex.strip())
                eval_root_pq = str(meta_eval.get("root_pq_pub_hex", "")).lower() if meta_eval else ""
                eval_leaf_pq = str(meta_eval.get("leaf_pub_hex", "")).lower() if meta_eval else ""
                pq_bound_ok = (
                    norm_eval_pq in bound_wots
                    or (eval_root_pq and eval_root_pq in bound_wots)
                    or (eval_leaf_pq and eval_leaf_pq in bound_wots)
                    or eval_id_by_pq.get(norm_eval_pq) == eval_id
                    or (eval_root_pq and eval_id_by_pq.get(eval_root_pq) == eval_id)
                )
                if not pq_bound_ok:
                    return False, "untrusted_or_unbound_evaluator_pq_pubkey"
                try:
                    if not WOTSPlusSHAKE256.verify(
                        bytes.fromhex(norm_eval_pq),
                        pae,
                        bytes.fromhex(att.pq_signature_hex.strip()),
                    ):
                        return False, "invalid_evaluator_pq_signature"
                except Exception as exc:
                    return False, f"invalid_evaluator_pq_signature:{exc}"

            if not att.ip_address or not str(att.ip_address).strip():
                return False, "missing_evaluator_ip_address"
            s_ok, s_reason = subnet_limiter.allow_vote(att.ip_address)
            if not s_ok:
                return False, f"evaluator_subnet_limit:{s_reason}"

            b_ok, b_reason = eff_asn_verifier.verify_binding(
                claimed_asn=att.asn,
                pubkey_hex=norm_pk,
                ip_address=att.ip_address,
                require_authoritative=True,
                require_pinned_key=True,
            )
            if not b_ok:
                return False, f"evaluator_asn_binding_failed:{b_reason}"
            routed_asn = eff_asn_verifier.lookup_ip_asn(att.ip_address)
            if routed_asn != int(att.asn):
                return False, f"evaluator_ip_asn_route_mismatch:{routed_asn}!={att.asn}"

            verified_eval_asns.add(int(att.asn))

        if len(verified_eval_asns) < int(min_evaluator_asns):
            return False, f"insufficient_evaluator_asns:{len(verified_eval_asns)}<{min_evaluator_asns}"

        for raw_m_pk, _ in envelope.manifest.iter_ed25519_signatures():
            norm_m_pk = ThresholdDHTManifest._normalize_pubkey_id(raw_m_pk)
            m_rel_id = norm_rel_trust.get(norm_m_pk)
            if m_rel_id is not None and m_rel_id in contributor_ids:
                return False, f"cyclic_provenance_violation:contributor_identity_in_release_signers:{m_rel_id}"
            if enforce_evaluator_release_disjoint and m_rel_id is not None and m_rel_id in seen_eval_ids:
                return False, f"cyclic_provenance_violation:evaluator_identity_in_release_signers:{m_rel_id}"
        for raw_m_pq, raw_m_pq_sig in envelope.manifest.iter_pq_signatures():
            norm_m_pq = ThresholdDHTManifest._normalize_pubkey_id(raw_m_pq)
            meta_m_pq = WOTSPlusSHAKE256.extract_signature_metadata(norm_m_pq, str(raw_m_pq_sig).strip())
            root_m_pq = str(meta_m_pq.get("root_pq_pub_hex", "")).lower() if meta_m_pq else ""
            m_pq_id = eff_pq_signers.get(norm_m_pq) or (eff_pq_signers.get(root_m_pq) if root_m_pq else None)
            if m_pq_id is not None and m_pq_id in contributor_ids:
                return False, f"cyclic_provenance_violation:contributor_identity_in_release_signers:{m_pq_id}"
            if enforce_evaluator_release_disjoint and m_pq_id is not None and m_pq_id in seen_eval_ids:
                return False, f"cyclic_provenance_violation:evaluator_identity_in_release_signers:{m_pq_id}"

        m_ok, m_reason = ThresholdSignatureVerifier.verify(
            manifest=envelope.manifest,
            trusted_signers=trusted_release_signers,
            threshold_m=threshold_m,
            current_seq=current_seq,
            trusted_pq_signers=eff_pq_signers,
            require_pq=require_pq,
            asn_verifier=eff_asn_verifier,
            verified_asn_quorum=verified_eval_asns,
            bootstrap_checkpoint_seq=bootstrap_checkpoint_seq,
        )
        if not m_ok:
            return False, f"release_manifest_verification_failed:{m_reason}"

        # Verify threshold release signatures over the DSSE PAE of the provenance envelope itself
        if not envelope.envelope_signatures:
            return False, "missing_envelope_release_signatures"
        env_pae = envelope.pae_bytes()
        valid_env_rel_ids: Set[str] = set()
        for raw_pk, raw_sig in envelope.envelope_signatures.items():
            norm_pk = ThresholdDHTManifest._normalize_pubkey_id(raw_pk)
            if norm_pk in contributors:
                return False, "cyclic_provenance_violation:contributor_in_release_signers"
            if enforce_evaluator_release_disjoint and norm_pk in evaluator_signers:
                return False, "cyclic_provenance_violation:evaluator_in_release_signers"
            if norm_pk not in norm_rel_trust:
                continue
            rel_id = norm_rel_trust[norm_pk]
            if rel_id in contributor_ids:
                return False, f"cyclic_provenance_violation:contributor_identity_in_release_signers:{rel_id}"
            if enforce_evaluator_release_disjoint and rel_id in seen_eval_ids:
                return False, f"cyclic_provenance_violation:evaluator_identity_in_release_signers:{rel_id}"
            if rel_id in valid_env_rel_ids:
                return False, f"duplicate_envelope_release_signer:{rel_id}"
            try:
                vk = ed25519.Ed25519PublicKey.from_public_bytes(bytes.fromhex(norm_pk))
                vk.verify(bytes.fromhex(raw_sig.strip()), env_pae)
                valid_env_rel_ids.add(rel_id)
            except Exception:
                continue
        if len(valid_env_rel_ids) < int(threshold_m):
            return False, f"insufficient_envelope_release_signatures:{len(valid_env_rel_ids)}<{threshold_m}"

        if require_pq:
            if not envelope.envelope_pq_signatures:
                return False, "missing_envelope_pq_release_signatures"
            valid_env_pq_ids: Set[str] = set()
            for raw_pq_pk, raw_pq_sig in envelope.envelope_pq_signatures.items():
                norm_pq = ThresholdDHTManifest._normalize_pubkey_id(raw_pq_pk)
                meta_env_pq = WOTSPlusSHAKE256.extract_signature_metadata(norm_pq, raw_pq_sig.strip())
                root_env_pq = str(meta_env_pq.get("root_pq_pub_hex", "")).lower() if meta_env_pq else ""
                pq_id = eff_pq_signers.get(norm_pq) or (eff_pq_signers.get(root_env_pq) if root_env_pq else None)
                if pq_id is None:
                    continue
                if pq_id in contributor_ids:
                    return False, f"cyclic_provenance_violation:contributor_identity_in_release_signers:{pq_id}"
                if enforce_evaluator_release_disjoint and pq_id in seen_eval_ids:
                    return False, f"cyclic_provenance_violation:evaluator_identity_in_release_signers:{pq_id}"
                if pq_id not in valid_env_rel_ids or pq_id in valid_env_pq_ids:
                    continue
                try:
                    if WOTSPlusSHAKE256.verify(bytes.fromhex(norm_pq), env_pae, bytes.fromhex(raw_pq_sig.strip())):
                        valid_env_pq_ids.add(pq_id)
                except Exception:
                    continue
            if len(valid_env_pq_ids) < int(threshold_m):
                return False, f"insufficient_envelope_pq_release_signatures:{len(valid_env_pq_ids)}<{threshold_m}"

        return True, "acyclic_provenance_envelope_verified"
