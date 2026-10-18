"""Unit tests for Armiger BEP-44 / BEP-46 Distributed RSI & Distributed Data Plugin.

Comprehensive test suite verifying:
  1. Universal Adapters (media library: Series, Movie, Audio, and Publication; Clients: qBittorrent, Transmission; ffprobe/MediaInfo).
  2. Foolproof Physical Ground-Truth Verification (zero unverified model predictions; download completion + stream decodability).
  3. Complete Anonymity (PII/IP/passkey stripping, blinded stems, differential privacy, Poisson jitter).
  4. End-to-End Encryption (X25519 ECDH + HKDF-SHA256 + ChaCha20-Poly1305 AEAD).
  5. Spoof-Proof Protocol (Ed25519 threshold multi-signatures + BEP-44 monotonic sequences).
  6. Poison-Proof Guards (K-of-N distinct-ASN Sybil quorum, safe formats [SafeTensors/ONNX/JSON], and 500-item golden canary gate).
  7. Opt-In / Opt-Out Modes ('off', 'pull_only', 'anonymous_contribute').
  8. End-to-End Integration.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import unittest
from typing import Any, Dict

from cryptography.hazmat.primitives.asymmetric import ed25519, x25519
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat, PrivateFormat, NoEncryption

from armiger_dht_rsi import (
    AnonymizedTelemetryRecord,
    ArmigerDHTRSIEngine,
    BlindedTitleStem,
    BundleFormat,
    CanaryItem,
    CandidateObservation,
    DHTRSIMode,
    EncryptedTelemetryBundle,
    GoldenCanaryRegressionGate,
    AudioWebhookAdapter,
    MediaCategory,
    MediaStreamVerifier,
    PeerAttestation,
    PhysicalGroundTruthVerifier,
    QBittorrentAdapter,
    MovieWebhookAdapter,
    PublicationWebhookAdapter,
    SafeFormatValidator,
    SeriesWebhookAdapter,
    SybilQuorumVerifier,
    TelemetryAnonymizer,
    ThresholdDHTManifest,
    ThresholdSignatureVerifier,
    HybridPQCAEADEngine,
    MLKEM768,
    TransmissionAdapter,
    VerifiedObservation,
    WOTSPlusSHAKE256,
    X25519ChaCha20Poly1305Engine,
    generate_golden_canary_suite,
    pure_laplace_noise,
)


class TestUniversalAdapters(unittest.TestCase):
    """Test universal adapters for Series, Movie, Audio, and Publication, qBittorrent, Transmission, ffprobe."""

    def test_series_webhook_adapter_download_complete(self):
        payload = {
            "eventType": "Download",
            "series": {"title": "Cosmos Laundromat", "seriesCatalogId": 371980, "movieCatalogId": 97546},
            "episodes": [{"seasonNumber": 1, "episodeNumber": 1, "title": "Good News About Hell"}],
            "release": {
                "releaseTitle": "Cosmos.Laundromat.S01E01.1080p.WEB-DL.DDP5.1.Atmos.H.264-OPEN",
                "size": 2400000000,
            },
            "episodeFile": {
                "relativePath": "Season 01/Cosmos Laundromat - S01E01 - First Cycle.mkv",
                "size": 2400000000,
                "mediaInfo": {
                    "videoCodec": "h264",
                    "audioFormat": "EAC3",
                    "audioChannels": 5.1,
                    "audioLanguages": "eng",
                    "subtitles": "eng/fre",
                },
            },
        }
        obs = SeriesWebhookAdapter.parse_webhook(payload)
        self.assertEqual(obs.source_adapter, "series_webhook")
        self.assertEqual(obs.claimed_category, "tv_show")
        self.assertTrue(obs.download_completed)
        self.assertEqual(obs.catalog_id_type, "series_catalog")
        self.assertEqual(obs.catalog_id_value, 371980)
        self.assertEqual(obs.total_bytes, 2400000000)
        self.assertEqual(obs.stream_metadata["audio_channels"], 5.1)

    def test_series_webhook_adapter_grab_in_progress(self):
        payload = {
            "eventType": "Grab",
            "series": {"title": "Cosmos Laundromat", "seriesCatalogId": 371980},
            "release": {"releaseTitle": "Cosmos.Laundromat.S01E01.1080p.WEB-DL", "size": 2400000000},
        }
        obs = SeriesWebhookAdapter.parse_webhook(payload)
        self.assertFalse(obs.download_completed)
        self.assertEqual(obs.client_state, "grabbed")

    def test_movie_webhook_adapter_download_complete(self):
        payload = {
            "eventType": "Download",
            "movie": {"title": "Tears of Steel", "year": 2024, "movieCatalogId": 693134, "externalRefId": "tt15239678"},
            "release": {
                "releaseTitle": "Tears.of.Steel.2012.2160p.UHD.BluRay.x265.TrueHD.Atmos.7.1-OPEN",
                "size": 35000000000,
            },
            "movieFile": {
                "relativePath": "Tears of Steel (2012).mkv",
                "size": 35000000000,
                "mediaInfo": {
                    "videoCodec": "hevc",
                    "audioFormat": "TrueHD",
                    "audioChannels": 7.1,
                    "audioLanguages": "eng",
                    "subtitles": "eng/spa",
                },
            },
        }
        obs = MovieWebhookAdapter.parse_webhook(payload)
        self.assertEqual(obs.source_adapter, "movie_webhook")
        self.assertEqual(obs.claimed_category, "movie")
        self.assertTrue(obs.download_completed)
        self.assertEqual(obs.catalog_id_type, "movie_catalog")
        self.assertEqual(obs.catalog_id_value, 693134)

    def test_audio_webhook_adapter_import(self):
        payload = {
            "eventType": "AlbumImported",
            "artist": {"name": "Kimiko Ishizaka", "audioCatalogId": "a74b1b7f-71a5-4011-9441-d0b5e4122711"},
            "album": {"title": "Open Goldberg Variations", "audioCatalogId": "b1b7f-1234"},
            "trackFiles": [{"audioFormat": "FLAC", "size": 350000000, "bitrate": 920000}],
        }
        obs = AudioWebhookAdapter.parse_webhook(payload)
        self.assertEqual(obs.source_adapter, "audio_webhook")
        self.assertEqual(obs.claimed_category, "music")
        self.assertTrue(obs.download_completed)
        self.assertEqual(obs.catalog_id_type, "audio_catalog")

    def test_publication_webhook_adapter_ebook_and_audiobook(self):
        payload_ebook = {
            "eventType": "BookFileImported",
            "book": {"title": "Frankenstein", "isbn": "9780441569595"},
            "author": {"name": "Mary Shelley"},
            "bookFiles": [{"format": "epub", "size": 1500000}],
        }
        obs_ebook = PublicationWebhookAdapter.parse_webhook(payload_ebook)
        self.assertEqual(obs_ebook.claimed_category, "ebook")
        self.assertTrue(obs_ebook.download_completed)

        payload_audio = {
            "eventType": "BookFileImported",
            "book": {"title": "Frankenstein", "isbn": "9780441569595"},
            "author": {"name": "Mary Shelley"},
            "bookFiles": [{"format": "m4b", "size": 450000000}],
        }
        obs_audio = PublicationWebhookAdapter.parse_webhook(payload_audio)
        self.assertEqual(obs_audio.claimed_category, "audiobook")

    def test_qbittorrent_adapter(self):
        payload_done = {
            "hash": "4a73b4d45d38c642277d32bbd213f56b69cfc890",
            "name": "Ubuntu.24.04.Desktop.AMD64.iso",
            "size": 6000000000,
            "progress": 1.0,
            "state": "uploading",
        }
        obs_done = QBittorrentAdapter.parse_event(payload_done)
        self.assertTrue(obs_done.download_completed)
        self.assertEqual(obs_done.claimed_category, "software")

        payload_prog = {
            "hash": "4a73b4d45d38c642277d32bbd213f56b69cfc890",
            "name": "Ubuntu.24.04.Desktop.AMD64.iso",
            "size": 6000000000,
            "progress": 0.45,
            "state": "downloading",
        }
        obs_prog = QBittorrentAdapter.parse_event(payload_prog)
        self.assertFalse(obs_prog.download_completed)

    def test_transmission_adapter(self):
        env_dict = {
            "TR_TORRENT_NAME": "Big.Buck.Bunny.2008.1080p.BluRay.x264-OPEN",
            "TR_TORRENT_HASH": "abc1234567890abcdef1234567890abcdef12345",
            "TR_TORRENT_DIR": "/downloads/complete",
        }
        obs = TransmissionAdapter.parse_event(env_dict)
        self.assertTrue(obs.download_completed)
        self.assertEqual(obs.claimed_category, "movie")

    def test_media_stream_verifier_ffprobe(self):
        ffprobe_data = {
            "streams": [
                {"codec_type": "video", "codec_name": "hevc", "width": 3840, "height": 2160},
                {
                    "codec_type": "audio",
                    "codec_name": "truehd",
                    "channels": 8,
                    "tags": {"language": "eng"},
                },
                {"codec_type": "subtitle", "tags": {"language": "eng"}},
            ],
            "format": {
                "format_name": "matroska,webm",
                "duration": "7200.5",
                "size": "35000000000",
            },
        }
        res = MediaStreamVerifier.verify_ffprobe_json(ffprobe_data)
        self.assertTrue(res["is_valid_media"])
        self.assertEqual(res["video_codec"], "hevc")
        self.assertEqual(res["audio_codec"], "truehd")
        self.assertEqual(res["audio_channels"], 8)
        self.assertIn("eng", res["audio_languages"])
        self.assertIn("eng", res["sub_languages"])
        self.assertEqual(res["duration_seconds"], 7200.5)


class TestPhysicalGroundTruthVerifier(unittest.TestCase):
    """Enforces foolproof physical ground truth verification (never raw unverified predictions)."""

    def setUp(self):
        self.verifier = PhysicalGroundTruthVerifier()

    def test_reject_incomplete_download(self):
        cand = CandidateObservation(
            source_adapter="series_webhook",
            raw_title="Cosmos.Laundromat.S01E01.1080p",
            claimed_category="tv_show",
            download_completed=False,  # Still downloading!
            bytes_downloaded=500000,
            total_bytes=2400000000,
            catalog_id_type="series_catalog",
            catalog_id_value=371980,
        )
        res = self.verifier.verify(cand)
        self.assertFalse(res.is_verified)
        self.assertEqual(res.rejection_reason, "incomplete_download_rejected")

    def test_reject_zero_byte_content(self):
        cand = CandidateObservation(
            source_adapter="qbittorrent",
            raw_title="Fake.Movie.1080p.mkv",
            claimed_category="movie",
            download_completed=True,
            bytes_downloaded=0,
            total_bytes=0,
        )
        res = self.verifier.verify(cand)
        self.assertFalse(res.is_verified)
        self.assertEqual(res.rejection_reason, "zero_byte_content_rejected")

    def test_reject_unverified_provenance_without_streams_or_catalog(self):
        cand = CandidateObservation(
            source_adapter="qbittorrent",
            raw_title="Random.File.Without.Catalog.Or.Streams",
            claimed_category="movie",
            download_completed=True,
            bytes_downloaded=1000000,
            total_bytes=1000000,
            catalog_id_type=None,
            catalog_id_value=None,
            stream_metadata=None,
        )
        res = self.verifier.verify(cand)
        self.assertFalse(res.is_verified)
        self.assertEqual(res.rejection_reason, "unverified_provenance_rejected")

    def test_detect_malware_masquerade_as_physical_junk(self):
        cand = CandidateObservation(
            source_adapter="transmission",
            raw_title="OpenMovie.2024.1080p.WEBRip.mkv.exe",
            claimed_category="movie",
            download_completed=True,
            bytes_downloaded=1500000,
            total_bytes=1500000,
        )
        res = self.verifier.verify(cand)
        self.assertTrue(res.is_verified)
        self.assertIsNotNone(res.observation)
        self.assertTrue(res.observation.junk)  # Physical JUNK detected!
        self.assertEqual(res.observation.owner_scope, "excluded")

    def test_verified_physical_tv_import(self):
        cand = CandidateObservation(
            source_adapter="series_webhook",
            raw_title="Cosmos.Laundromat.S01E01.1080p.WEB-DL.DDP5.1.Atmos.H.264-OPEN",
            claimed_category="tv_show",
            download_completed=True,
            bytes_downloaded=2400000000,
            total_bytes=2400000000,
            catalog_id_type="series_catalog",
            catalog_id_value=371980,
            stream_metadata={
                "video_codec": "h264",
                "audio_codec": "eac3",
                "audio_channels": 5.1,
                "audio_languages": ["eng"],
                "subtitles": "eng",
                "duration_seconds": 3600.0,
            },
        )
        res = self.verifier.verify(cand)
        self.assertTrue(res.is_verified)
        obs = res.observation
        self.assertEqual(obs.category, "tv_show")
        self.assertFalse(obs.adult)
        self.assertFalse(obs.anime)
        self.assertEqual(obs.lang, "english")
        self.assertFalse(obs.junk)
        self.assertEqual(obs.owner_scope, "allowed_type")
        self.assertTrue(obs.english_subs)
        self.assertEqual(obs.title_language, "english_title")


class TestCompleteAnonymityAndPrivacy(unittest.TestCase):
    """Test PII, passkey, IP, and path stripping; blinded stems; Laplace DP; Poisson jitter."""

    def setUp(self):
        self.anonymizer = TelemetryAnonymizer(
            blinding_salt=b"test-blinding-salt-32bytes-secret",
            dp_epsilon=1.0,
            poisson_mean_seconds=1800.0,
        )

    def test_strip_pii_passkeys_ips_and_paths(self):
        text = (
            "Tracker url: http://mytracker.org/announce?passkey=testpass0000 "
            "Downloaded by user at /Users/alice/Downloads/Big_Buck_Bunny.mkv from peer 198.51.100.55 "
            "and IPv6 2001:0db8:85a3:0000:0000:8a2e:0370:7334 with authkey: 998877665544332211"
        )
        clean = self.anonymizer.strip_pii(text)
        self.assertNotIn("abcdef1234567890abcdef", clean)
        self.assertNotIn("998877665544332211", clean)
        self.assertNotIn("198.51.100.55", clean)
        self.assertNotIn("2001:0db8:85a3:0000:0000:8a2e:0370:7334", clean)
        self.assertNotIn("/Users/alice", clean)
        self.assertIn("[REDACTED_PASSKEY]", clean)
        self.assertIn("[IP_REDACTED]", clean)
        self.assertIn("[IPV6_REDACTED]", clean)
        self.assertIn("/home/[USER_REDACTED]", clean)

    def test_blind_title_stem(self):
        title = "Big.Buck.Bunny.2008.1080p.BluRay.x264.DTS-HD.MA.5.1-OPEN"
        blinded = self.anonymizer.blind_title_stem(title)
        self.assertEqual(len(blinded.blinded_hash), 16)
        self.assertIn("1080p", blinded.token_features)
        self.assertIn("x264", blinded.token_features)
        self.assertIn("bluray", blinded.token_features)
        self.assertNotIn("bunny", [t.lower() for t in blinded.token_features])

        # Same title produces deterministic blinded hash with same salt
        blinded2 = self.anonymizer.blind_title_stem(title)
        self.assertEqual(blinded.blinded_hash, blinded2.blinded_hash)

        # Different salt produces different hash
        anon_diff_salt = TelemetryAnonymizer(blinding_salt=b"different-salt-32bytes-secret!")
        blinded3 = anon_diff_salt.blind_title_stem(title)
        self.assertNotEqual(blinded.blinded_hash, blinded3.blinded_hash)

    def test_pure_laplace_noise_and_differential_privacy(self):
        # Test pure Laplace noise generator
        samples = [pure_laplace_noise(scale=2.0) for _ in range(5000)]
        mean = sum(samples) / len(samples)
        self.assertAlmostEqual(mean, 0.0, delta=0.2)

        # Test DP size quantization
        size_bytes = 4 * 1024 * 1024 * 1024  # 4096 MB
        dp_size, dp_dur = self.anonymizer.apply_differential_privacy(size_bytes, 7200.0)
        self.assertGreater(dp_size, 0.0)
        self.assertIsNotNone(dp_dur)
        self.assertGreater(dp_dur, 0.0)

    def test_poisson_timing_jitter(self):
        delays = [self.anonymizer.calculate_poisson_jitter() for _ in range(500)]
        mean_delay = sum(delays) / len(delays)
        # Expected mean is ~1800s, allow reasonable statistical bounds [1400, 2200]
        self.assertGreater(mean_delay, 1400.0)
        self.assertLess(mean_delay, 2200.0)


class TestX25519ChaCha20Poly1305AEAD(unittest.TestCase):
    """Test X25519 ECDH key exchange with ChaCha20-Poly1305 AEAD encryption."""

    def test_encrypt_and_decrypt_success(self):
        # Recipient generates keypair
        recip_priv, recip_pub = X25519ChaCha20Poly1305Engine.generate_keypair()

        msg = b'{"verified": true, "category": "movie", "title_blinded": "a1b2c3d4e5f60718"}'
        bundle = X25519ChaCha20Poly1305Engine.encrypt_bundle(
            payload_bytes=msg,
            recipient_pubkey_bytes=recip_pub,
            seq=42,
        )

        decrypted = X25519ChaCha20Poly1305Engine.decrypt_bundle(bundle, recip_priv)
        self.assertEqual(decrypted, msg)

    def test_tampered_ciphertext_fails_aead_authentication(self):
        recip_priv, recip_pub = X25519ChaCha20Poly1305Engine.generate_keypair()
        msg = b'{"secret": "data"}'
        bundle = X25519ChaCha20Poly1305Engine.encrypt_bundle(msg, recip_pub, seq=1)

        # Tamper ciphertext by flipping bits in the hex string
        tampered_hex = list(bundle.ciphertext_hex)
        tampered_hex[0] = "0" if tampered_hex[0] != "0" else "1"
        bad_bundle = EncryptedTelemetryBundle(
            ephemeral_pubkey_hex=bundle.ephemeral_pubkey_hex,
            nonce_hex=bundle.nonce_hex,
            ciphertext_hex="".join(tampered_hex),
            aad=bundle.aad,
            seq=bundle.seq,
        )

        with self.assertRaises(ValueError):
            X25519ChaCha20Poly1305Engine.decrypt_bundle(bad_bundle, recip_priv)

    def test_post_quantum_mlkem768_fips203_and_fujisaki_okamoto_rejection(self):
        """Verify NIST FIPS 203 ML-KEM-768 keygen/encaps/decaps and FO implicit rejection on tampered lattice ct."""
        ek, dk = MLKEM768.keygen()
        self.assertEqual(len(ek), 1184)
        self.assertEqual(len(dk), 2400)
        ct, ss_enc = MLKEM768.encaps(ek)
        self.assertEqual(len(ct), 1088)
        self.assertEqual(len(ss_enc), 32)
        ss_dec = MLKEM768.decaps(dk, ct)
        self.assertEqual(ss_enc, ss_dec)

        # Flip 1 bit in ML-KEM-768 ciphertext -> FO implicit rejection produces pseudorandom K_bar != ss_enc
        bad_ct = bytearray(ct)
        bad_ct[500] ^= 0x01
        ss_rejected = MLKEM768.decaps(dk, bytes(bad_ct))
        self.assertEqual(len(ss_rejected), 32)
        self.assertNotEqual(ss_rejected, ss_enc)

        # Verify Hybrid X25519 + ML-KEM-768 AEAD bundle rejects tampered lattice ciphertext even if AAD is recomputed
        recip_priv, recip_pub = HybridPQCAEADEngine.generate_keypair()
        self.assertEqual(len(recip_priv), 2432)
        self.assertEqual(len(recip_pub), 1216)
        bundle = HybridPQCAEADEngine.encrypt_bundle(b'{"pq": true}', recip_pub, seq=7)
        self.assertTrue(bundle.is_post_quantum)
        self.assertEqual(len(bytes.fromhex(bundle.pq_kem_ciphertext_hex)), 1088)

        ephem_bytes = bytearray(bytes.fromhex(bundle.ephemeral_pubkey_hex))
        ephem_bytes[100] ^= 0x01  # Flip bit inside ML-KEM-768 ciphertext portion (bytes 32..1120)
        forged_ephem_hex = ephem_bytes.hex()
        forged_bundle = EncryptedTelemetryBundle(
            ephemeral_pubkey_hex=forged_ephem_hex,
            nonce_hex=bundle.nonce_hex,
            ciphertext_hex=bundle.ciphertext_hex,
            aad=f"seq:{bundle.seq};ephem_pub:{forged_ephem_hex}",
            seq=bundle.seq,
        )
        with self.assertRaises(ValueError):
            HybridPQCAEADEngine.decrypt_bundle(forged_bundle, recip_priv)

    def test_tampered_aad_fails_aead_authentication(self):
        recip_priv, recip_pub = X25519ChaCha20Poly1305Engine.generate_keypair()
        msg = b'{"secret": "data"}'
        bundle = X25519ChaCha20Poly1305Engine.encrypt_bundle(msg, recip_pub, seq=1)

        bad_bundle = EncryptedTelemetryBundle(
            ephemeral_pubkey_hex=bundle.ephemeral_pubkey_hex,
            nonce_hex=bundle.nonce_hex,
            ciphertext_hex=bundle.ciphertext_hex,
            aad="tampered_aad_value",
            seq=bundle.seq,
        )

        with self.assertRaises(ValueError):
            X25519ChaCha20Poly1305Engine.decrypt_bundle(bad_bundle, recip_priv)


class TestSpoofProofProtocol(unittest.TestCase):
    """Test BEP-44 strictly monotonic sequence numbers and Ed25519 threshold multi-signatures."""

    def setUp(self):
        # Federation signers
        self.priv1 = ed25519.Ed25519PrivateKey.generate()
        self.pub1 = self.priv1.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()

        self.priv2 = ed25519.Ed25519PrivateKey.generate()
        self.pub2 = self.priv2.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()

        self.priv3 = ed25519.Ed25519PrivateKey.generate()
        self.pub3 = self.priv3.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()

        self.trusted_signers = {
            self.pub1: "norvi_root_1",
            self.pub2: "norvi_root_2",
            self.pub3: "federation_partner_1",
        }

    def _sign_manifest(self, manifest_body: Dict[str, Any], priv_keys: list) -> Dict[str, str]:
        body = {
            "bundle_format": manifest_body["bundle_format"].lower(),
            "bundle_sha256": manifest_body["bundle_sha256"].lower(),
            "infohash": manifest_body["infohash"].lower(),
            "min_asn_quorum": int(manifest_body["min_asn_quorum"]),
            "seq": int(manifest_body["seq"]),
            "timestamp": int(manifest_body["timestamp"]),
            "version": str(manifest_body["version"]),
        }
        canonical = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
        sigs = {}
        for priv in priv_keys:
            pub_hex = priv.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
            sig = priv.sign(canonical).hex()
            sigs[pub_hex] = sig
        return sigs

    def test_bep44_monotonic_sequence_enforcement(self):
        # seq 10 with current_seq = 10 -> REJECT (replay/stale)
        manifest = ThresholdDHTManifest(
            seq=10,
            version="2.1.0",
            infohash="1122334455667788990011223344556677889900",
            bundle_sha256="aabbccddeeff00112233445566778899aabbccddeeff00112233445566778899",
            bundle_format="safetensors",
            min_asn_quorum=5,
            timestamp=int(time.time()),
            signatures={},
        )
        ok, reason = ThresholdSignatureVerifier.verify(
            manifest, self.trusted_signers, threshold_m=1, current_seq=10
        )
        self.assertFalse(ok)
        self.assertIn("stale_or_replayed_seq", reason)

        # seq 9 with current_seq = 10 -> REJECT (downgrade)
        manifest_down = ThresholdDHTManifest(
            seq=9,
            version="2.0.9",
            infohash="1122334455667788990011223344556677889900",
            bundle_sha256="aabbccddeeff00112233445566778899aabbccddeeff00112233445566778899",
            bundle_format="safetensors",
            min_asn_quorum=5,
            timestamp=int(time.time()),
            signatures={},
        )
        ok, reason = ThresholdSignatureVerifier.verify(
            manifest_down, self.trusted_signers, threshold_m=1, current_seq=10
        )
        self.assertFalse(ok)

    def test_threshold_signatures_2_of_3(self):
        m_data = {
            "seq": 11,
            "version": "2.1.0",
            "infohash": "1122334455667788990011223344556677889900",
            "bundle_sha256": "aabbccddeeff00112233445566778899aabbccddeeff00112233445566778899",
            "bundle_format": "safetensors",
            "min_asn_quorum": 5,
            "timestamp": int(time.time()),
        }

        # 1-of-3 signatures when threshold is 2 -> REJECT
        sigs_1 = self._sign_manifest(m_data, [self.priv1])
        manifest_1 = ThresholdDHTManifest(**m_data, signatures=sigs_1)
        ok, reason = ThresholdSignatureVerifier.verify(
            manifest_1, self.trusted_signers, threshold_m=2, current_seq=10
        )
        self.assertFalse(ok)
        self.assertIn("insufficient_threshold_signatures:1<2", reason)

        # 2-of-3 signatures when threshold is 2 -> PASS!
        sigs_2 = self._sign_manifest(m_data, [self.priv1, self.priv2])
        manifest_2 = ThresholdDHTManifest(**m_data, signatures=sigs_2)
        ok, reason = ThresholdSignatureVerifier.verify(
            manifest_2, self.trusted_signers, threshold_m=2, current_seq=10
        )
        self.assertTrue(ok)
        self.assertIn("threshold_verified", reason)

    def test_post_quantum_wots_shake256_hybrid_threshold_signatures(self):
        """Verify Hybrid Ed25519 + WOTS+-SHAKE256 M-of-N threshold signatures block classical-only forgeries."""
        s1_ed, s1_ed_pub, s1_pq_priv, s1_pq_pub = ThresholdSignatureVerifier.generate_hybrid_signer()
        s2_ed, s2_ed_pub, s2_pq_priv, s2_pq_pub = ThresholdSignatureVerifier.generate_hybrid_signer()
        s3_ed, s3_ed_pub, s3_pq_priv, s3_pq_pub = ThresholdSignatureVerifier.generate_hybrid_signer()

        trusted_ed = {s1_ed_pub: "root1", s2_ed_pub: "root2", s3_ed_pub: "root3"}
        trusted_pq = {s1_pq_pub: "root1_pq", s2_pq_pub: "root2_pq", s3_pq_pub: "root3_pq"}

        m_data = {
            "seq": 25,
            "version": "2.1.0",
            "infohash": "1122334455667788990011223344556677889900",
            "bundle_sha256": "aabbccddeeff00112233445566778899aabbccddeeff00112233445566778899",
            "bundle_format": "safetensors",
            "min_asn_quorum": 5,
            "timestamp": int(time.time()),
        }
        ed_sigs = self._sign_manifest(m_data, [s1_ed, s2_ed])
        temp_m = ThresholdDHTManifest(**m_data, signatures=ed_sigs)
        payload = temp_m.canonical_payload()

        pq_sigs = {
            s1_pq_pub: WOTSPlusSHAKE256.sign(s1_pq_priv, payload).hex(),
            s2_pq_pub: WOTSPlusSHAKE256.sign(s2_pq_priv, payload).hex(),
        }
        hybrid_manifest = ThresholdDHTManifest(**m_data, signatures=ed_sigs, pq_signatures=pq_sigs)

        ok, reason = ThresholdSignatureVerifier.verify(
            hybrid_manifest, trusted_ed, threshold_m=2, current_seq=20, trusted_pq_signers=trusted_pq, require_pq=True
        )
        self.assertTrue(ok, f"Expected hybrid PQ manifest to verify, got: {reason}")

        # Simulate quantum adversary who broke Ed25519 (2 valid Ed25519 sigs) but only has 1 valid WOTS+-SHAKE256 sig
        forged_pq_sigs = {
            s1_pq_pub: pq_sigs[s1_pq_pub],
            s2_pq_pub: "00" * WOTSPlusSHAKE256.SIG_BYTES,
        }
        quantum_forged_manifest = ThresholdDHTManifest(**m_data, signatures=ed_sigs, pq_signatures=forged_pq_sigs)
        ok_bad, reason_bad = ThresholdSignatureVerifier.verify(
            quantum_forged_manifest, trusted_ed, threshold_m=2, current_seq=20, trusted_pq_signers=trusted_pq, require_pq=True
        )
        self.assertFalse(ok_bad)
        self.assertIn("insufficient_pq_threshold_signatures:1<2", reason_bad)


class TestSybilQuorumEngine(unittest.TestCase):
    """Test K-of-N distinct-ASN Sybil quorum verification."""

    def _create_attestation(self, asn: int, peer_idx: int, target_hash: str) -> PeerAttestation:
        priv = ed25519.Ed25519PrivateKey.generate()
        pub_hex = priv.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
        ts = int(time.time())
        peer_id = f"peer_{peer_idx}"
        body = {
            "asn": int(asn),
            "peer_id": peer_id,
            "target_hash": target_hash.lower(),
            "timestamp": int(ts),
        }
        canon = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
        sig_hex = priv.sign(canon).hex()
        return PeerAttestation(
            peer_id=peer_id,
            asn=asn,
            peer_pubkey_hex=pub_hex,
            timestamp=ts,
            signature_hex=sig_hex,
        )

    def test_distinct_asn_quorum_success(self):
        target = "1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"
        # 5 distinct ASNs: Cloudflare, Google, AWS, Comcast, AT&T
        asns = [13335, 15169, 16509, 7922, 7018]
        attestations = [self._create_attestation(asn, i, target) for i, asn in enumerate(asns)]

        ok, reason, verified_asns = SybilQuorumVerifier.verify_quorum(
            attestations, target, min_distinct_asns=5
        )
        self.assertTrue(ok)
        self.assertEqual(len(verified_asns), 5)

    def test_single_asn_sybil_attack_rejected(self):
        target = "1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"
        # 10 nodes, but all from the SAME ASN (e.g. AWS 16509)
        attestations = [self._create_attestation(16509, i, target) for i in range(10)]

        ok, reason, verified_asns = SybilQuorumVerifier.verify_quorum(
            attestations, target, min_distinct_asns=5
        )
        self.assertFalse(ok)
        self.assertEqual(len(verified_asns), 1)
        self.assertIn("insufficient_distinct_asns:1<5", reason)


class TestSafeFormatValidator(unittest.TestCase):
    """Test strict safe format validation (SafeTensors, ONNX, JSON only; rejects pickle)."""

    def test_valid_safetensors_accepted(self):
        header_dict = {
            "weight": {
                "dtype": "F32",
                "shape": [2, 2],
                "data_offsets": [0, 16],
            },
            "__metadata__": {"format": "pt_lora"},
        }
        header_bytes = json.dumps(header_dict).encode("utf-8")
        header_len = len(header_bytes)
        tensor_buffer = os.urandom(16)
        bundle_bytes = header_len.to_bytes(8, "little") + header_bytes + tensor_buffer

        ok, reason = SafeFormatValidator.validate(bundle_bytes, "safetensors")
        self.assertTrue(ok)
        self.assertEqual(reason, "safetensors_valid")

    def test_safetensors_offset_out_of_bounds_rejected(self):
        header_dict = {
            "weight": {
                "dtype": "F32",
                "shape": [2, 2],
                "data_offsets": [0, 9999],  # out of buffer!
            }
        }
        header_bytes = json.dumps(header_dict).encode("utf-8")
        header_len = len(header_bytes)
        bundle_bytes = header_len.to_bytes(8, "little") + header_bytes + b"\x00" * 16

        ok, reason = SafeFormatValidator.validate(bundle_bytes, "safetensors")
        self.assertFalse(ok)
        self.assertIn("safetensors_offset_out_of_bounds", reason)

    def test_reject_prohibited_pickle_bytecode(self):
        # File disguised as safetensors or json but containing pickle opcode
        poison_payload = b"\x80\x04\x95\x1e\x00\x00\x00\x00\x00\x00\x00cos\nsystem\n(X\x06\x00\x00\x00whoami\x85R."
        ok, reason = SafeFormatValidator.validate(poison_payload, "safetensors")
        self.assertFalse(ok)
        self.assertIn("poison_detected", reason)

    def test_valid_json_accepted(self):
        valid_json = json.dumps({"heads": {"category": {"temperature": 1.2}}}).encode("utf-8")
        ok, reason = SafeFormatValidator.validate(valid_json, "json")
        self.assertTrue(ok)
        self.assertEqual(reason, "json_valid")


class TestMandatory500ItemGoldenCanaryRegressionGate(unittest.TestCase):
    """Test the mandatory 500-item golden canary regression gate."""

    def setUp(self):
        self.canary_items = generate_golden_canary_suite()
        self.assertEqual(len(self.canary_items), 500)
        self.gate = GoldenCanaryRegressionGate(self.canary_items)

    def test_golden_canary_composition(self):
        counts = {}
        for item in self.canary_items:
            cat = item.ground_truth["category"]
            counts[cat] = counts.get(cat, 0) + 1

        self.assertEqual(counts["movie"], 100)
        self.assertEqual(counts["tv_show"], 100)
        self.assertEqual(counts["anime"], 70)  # 50 anime series + 20 anime movies
        self.assertEqual(counts["music"], 40)
        self.assertEqual(counts["comic"], 30)
        self.assertEqual(counts["audiobook"], 30)
        self.assertEqual(counts["ebook"], 30)
        self.assertEqual(counts["game"], 15)
        self.assertEqual(counts["software"], 15)
        self.assertEqual(counts["course"], 20)
        self.assertEqual(counts["xxx"], 25)
        self.assertEqual(counts["unknown"], 25)  # 25 junk/malware traps

    def test_perfect_baseline_predictor_passes_all_gates(self):
        def baseline_predictor(item: CanaryItem) -> Dict[str, Any]:
            return dict(item.ground_truth)

        report = self.gate.evaluate(baseline_predictor)
        self.assertTrue(report.passed)
        self.assertEqual(report.total_items, 500)
        self.assertEqual(report.junk_balanced_accuracy, 1.0)
        self.assertEqual(report.junk_false_negatives, 0)
        self.assertEqual(report.adult_accuracy, 1.0)
        self.assertEqual(report.category_accuracy, 1.0)

    def test_malware_false_negative_triggers_instant_rejection(self):
        # A candidate predictor that predicts clean on even 1 malware trap!
        def malicious_candidate(item: CanaryItem) -> Dict[str, Any]:
            pred = dict(item.ground_truth)
            if item.item_id == "canary_jnk_000":
                pred["junk"] = False  # Bypassed malware filter!
            return pred

        report = self.gate.evaluate(malicious_candidate)
        self.assertFalse(report.passed)
        self.assertEqual(report.junk_false_negatives, 1)
        self.assertIn("CRITICAL_SAFETY_FAIL", report.reason)

    def test_adult_accuracy_regression_triggers_rejection(self):
        # Predictor failing on adult boundary
        def regressed_candidate(item: CanaryItem) -> Dict[str, Any]:
            pred = dict(item.ground_truth)
            if item.ground_truth["adult"]:
                pred["adult"] = False  # fails adult detection
            return pred

        report = self.gate.evaluate(regressed_candidate)
        self.assertFalse(report.passed)
        self.assertIn("ADULT_ACCURACY_REGRESSION", report.reason)


class TestOptInOutModesAndMasterEngine(unittest.TestCase):
    """Test ArmigerDHTRSIEngine with opt-in/opt-out modes and end-to-end flows."""

    def setUp(self):
        self.priv_signer = ed25519.Ed25519PrivateKey.generate()
        self.pub_signer_hex = self.priv_signer.public_key().public_bytes(
            Encoding.Raw, PublicFormat.Raw
        ).hex()
        self.trusted_signers = {self.pub_signer_hex: "root_authority"}

        self.aggregator_priv, self.aggregator_pub = X25519ChaCha20Poly1305Engine.generate_keypair()
        self.aggregator_pub_hex = self.aggregator_pub.hex()

    def test_mode_off(self):
        engine = ArmigerDHTRSIEngine(
            mode="off",
            trusted_signers=self.trusted_signers,
            aggregator_x25519_pubkey_hex=self.aggregator_pub_hex,
        )
        cand = CandidateObservation(
            source_adapter="series_webhook",
            raw_title="Cosmos.Laundromat.S01E01.1080p",
            claimed_category="tv_show",
            download_completed=True,
            bytes_downloaded=1000000,
            total_bytes=1000000,
            catalog_id_type="series_catalog",
            catalog_id_value=371980,
            stream_metadata={"video_codec": "h264"},
        )
        ok, reason, rec = engine.ingest_candidate(cand)
        self.assertFalse(ok)
        self.assertEqual(reason, "plugin_mode_off")

    def test_mode_pull_only_blocks_telemetry_but_allows_updates(self):
        engine = ArmigerDHTRSIEngine(
            mode="pull_only",
            trusted_signers=self.trusted_signers,
            threshold_m=1,
            min_distinct_asns=1,
            aggregator_x25519_pubkey_hex=self.aggregator_pub_hex,
        )
        # 1. Ingestion is blocked
        cand = CandidateObservation(
            source_adapter="movie_webhook",
            raw_title="Tears.of.Steel.2012.1080p",
            claimed_category="movie",
            download_completed=True,
            bytes_downloaded=1000000,
            total_bytes=1000000,
            catalog_id_type="movie_catalog",
            catalog_id_value=693134,
            stream_metadata={"video_codec": "hevc"},
        )
        ok, reason, rec = engine.ingest_candidate(cand)
        self.assertFalse(ok)
        self.assertEqual(reason, "telemetry_disabled_in_pull_only_mode")

    def test_mode_anonymous_contribute_end_to_end(self):
        engine = ArmigerDHTRSIEngine(
            mode="anonymous_contribute",
            current_seq=100,
            trusted_signers=self.trusted_signers,
            threshold_m=1,
            min_distinct_asns=2,
            blinding_salt=b"test-salt-secret-32-bytes-here!!",
            dp_epsilon=1.0,
            poisson_mean_seconds=60.0,
            aggregator_x25519_pubkey_hex=self.aggregator_pub_hex,
        )

        cand = CandidateObservation(
            source_adapter="series_webhook",
            raw_title="Cosmos.Laundromat.S01E01.1080p.WEB-DL.DDP5.1-OPEN?passkey=testpass0000",
            claimed_category="tv_show",
            download_completed=True,
            bytes_downloaded=2400000000,
            total_bytes=2400000000,
            catalog_id_type="series_catalog",
            catalog_id_value=371980,
            stream_metadata={
                "video_codec": "h264",
                "audio_codec": "eac3",
                "audio_channels": 5.1,
                "audio_languages": ["eng"],
                "subtitles": "eng",
                "duration_seconds": 3600.0,
            },
        )

        now = 100000.0
        ok, reason, record = engine.ingest_candidate(cand, current_timestamp=now)
        self.assertTrue(ok)
        self.assertEqual(reason, "verified_anonymized_and_queued")
        self.assertIsNotNone(record)
        self.assertEqual(len(engine.telemetry_queue), 1)

        # Before Poisson jitter elapses -> no items ready
        bundles_early = engine.get_pending_telemetry_batch(now_timestamp=now + 1.0)
        self.assertEqual(len(bundles_early), 0)

        # After Poisson jitter elapses -> batch is dispatched and encrypted!
        bundles_ready = engine.get_pending_telemetry_batch(now_timestamp=now + 10000.0)
        self.assertEqual(len(bundles_ready), 1)
        bundle = bundles_ready[0]
        self.assertEqual(bundle.seq, 101)

        # Decrypt bundle at simulated aggregator
        decrypted_bytes = X25519ChaCha20Poly1305Engine.decrypt_bundle(
            bundle, self.aggregator_priv
        )
        decrypted_json = json.loads(decrypted_bytes.decode("utf-8"))
        self.assertEqual(decrypted_json["category"], "tv_show")
        self.assertEqual(decrypted_json["lang"], "english")
        self.assertFalse(decrypted_json["adult"])
        self.assertFalse(decrypted_json["junk"])
        # Verify passkey was never leaked!
        self.assertNotIn("secret1234567890", json.dumps(decrypted_json))

    def test_staged_update_verification_and_sequence_advancement(self):
        engine = ArmigerDHTRSIEngine(
            mode="anonymous_contribute",
            current_seq=50,
            trusted_signers=self.trusted_signers,
            threshold_m=1,
            min_distinct_asns=2,
            aggregator_x25519_pubkey_hex=self.aggregator_pub_hex,
        )

        bundle_content = json.dumps({"heads": {"category": {"temperature": 1.25}}}).encode("utf-8")
        bundle_sha256 = hashlib.sha256(bundle_content).hexdigest()

        # Build signed manifest
        manifest_body = {
            "seq": 51,
            "version": "2.1.0",
            "infohash": "0123456789abcdef0123456789abcdef01234567",
            "bundle_sha256": bundle_sha256,
            "bundle_format": "json",
            "min_asn_quorum": 2,
            "timestamp": int(time.time()),
        }
        canonical = json.dumps(manifest_body, sort_keys=True, separators=(",", ":")).encode("utf-8")
        sig = self.priv_signer.sign(canonical).hex()
        manifest = ThresholdDHTManifest(**manifest_body, signatures={self.pub_signer_hex: sig})

        # Build attestations from 2 distinct ASNs
        def create_attestation(asn: int, peer_id: str) -> PeerAttestation:
            priv = ed25519.Ed25519PrivateKey.generate()
            pub_hex = priv.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
            ts = int(time.time())
            body = {"asn": asn, "peer_id": peer_id, "target_hash": bundle_sha256.lower(), "timestamp": ts}
            canon_bytes = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
            s = priv.sign(canon_bytes).hex()
            return PeerAttestation(peer_id=peer_id, asn=asn, peer_pubkey_hex=pub_hex, timestamp=ts, signature_hex=s)

        attestations = [
            create_attestation(13335, "node_cloudflare"),
            create_attestation(15169, "node_google"),
        ]

        def dummy_predictor(item: CanaryItem) -> Dict[str, Any]:
            return dict(item.ground_truth)

        ok, reason, report = engine.verify_and_stage_update(
            manifest=manifest,
            bundle_bytes=bundle_content,
            attestations=attestations,
            candidate_predictor_fn=dummy_predictor,
        )
        self.assertTrue(ok)
        self.assertEqual(reason, "update_staged_and_verified")
        self.assertIsNotNone(report)
        self.assertTrue(report.passed)
        # Verify sequence was advanced from 50 to 51
        self.assertEqual(engine.current_seq, 51)


# =====================================================================
# 9. Phase 2: 1,000-Node Swarm Simulation, 200-Payload Red-Team & Latency Benchmarks
# =====================================================================

FIFTY_REAL_ASNS = [
    13335, 15169, 16509, 7922, 7018, 3320, 24940, 16276, 14061, 63949,
    20940, 8075, 32934, 2906, 46489, 54113, 396982, 1299, 3356, 174,
    2914, 6453, 6939, 9009, 20473, 36352, 46606, 51167, 60068, 62567,
    6830, 12322, 5410, 3215, 3209, 8881, 6805, 5089, 2856, 1257,
    2516, 4713, 4766, 9318, 4134, 4837, 9808, 4788, 7545, 1221,
]


def run_swarm_and_redteam_benchmark() -> Dict[str, Any]:
    """Execute the 1,000-node 50-ASN swarm simulation, 200 adversarial red-team payloads,
    and end-to-end latency/throughput benchmarks, returning a structured report.
    """
    # 1. Setup 5-key Hybrid Post-Quantum Federation (3-of-5 Ed25519 + WOTS+-SHAKE256) & Aggregator X25519+ML-KEM-768 keypair
    fed_signers = [ThresholdSignatureVerifier.generate_hybrid_signer() for _ in range(5)]
    fed_privs = [s[0] for s in fed_signers]
    fed_pubs = [s[1] for s in fed_signers]
    fed_pq_privs = [s[2] for s in fed_signers]
    fed_pq_pubs = [s[3] for s in fed_signers]
    trusted_signers = {pub: f"federation_signer_{i}" for i, pub in enumerate(fed_pubs)}
    trusted_pq_signers = {pub: f"federation_pq_signer_{i}" for i, pub in enumerate(fed_pq_pubs)}
    agg_priv, agg_pub = X25519ChaCha20Poly1305Engine.generate_keypair()

    engine = ArmigerDHTRSIEngine(
        mode="anonymous_contribute",
        current_seq=500,
        trusted_signers=trusted_signers,
        threshold_m=3,
        min_distinct_asns=5,
        blinding_salt=b"swarm-benchmark-salt-32-bytes!!",
        dp_epsilon=1.0,
        poisson_mean_seconds=1800.0,
        aggregator_x25519_pubkey_hex=agg_pub.hex(),
    )

    # -----------------------------------------------------------------
    # Part A: 1,000-Node Swarm Simulation across 50 ASNs
    # -----------------------------------------------------------------
    base_ts = 1792350000.0
    adapter_counts: Dict[str, int] = {
        "series_webhook": 0,
        "movie_webhook": 0,
        "audio_webhook": 0,
        "pub_ebook": 0,
        "pub_audiobook": 0,
        "qbittorrent_ffprobe": 0,
        "transmission_ffprobe": 0,
    }

    t0_ingest = time.perf_counter()
    ingested_records = 0

    for node_idx in range(1000):
        asn = FIFTY_REAL_ASNS[node_idx % 50]
        kind = node_idx % 7
        ffprobe_payload = None

        if kind == 0:
            cand = SeriesWebhookAdapter.parse_webhook({
                "eventType": "Download",
                "series": {"title": f"Series_{node_idx}", "seriesCatalogId": 300000 + node_idx},
                "episodes": [{"seasonNumber": 1, "episodeNumber": (node_idx % 12) + 1}],
                "release": {
                    "releaseTitle": f"Series.{node_idx}.S01E01.1080p.WEB-DL.DDP5.1.H.264-OPEN",
                    "size": 2200000000 + node_idx * 1024,
                },
                "episodeFile": {
                    "relativePath": f"Season 01/Series_{node_idx}_S01E01.mkv",
                    "size": 2200000000 + node_idx * 1024,
                    "duration": 3400.0,
                    "mediaInfo": {
                        "videoCodec": "h264",
                        "audioFormat": "eac3",
                        "audioChannels": 5.1,
                        "audioLanguages": "eng",
                        "subtitles": "eng",
                    },
                },
            })
            adapter_counts["series_webhook"] += 1
        elif kind == 1:
            cand = MovieWebhookAdapter.parse_webhook({
                "eventType": "Download",
                "movie": {"title": f"Film_{node_idx}", "year": 2024, "movieCatalogId": 600000 + node_idx},
                "release": {
                    "releaseTitle": f"Film.{node_idx}.2024.2160p.UHD.BluRay.x265.TrueHD.Atmos.7.1-OPEN",
                    "size": 32000000000 + node_idx * 2048,
                },
                "movieFile": {
                    "relativePath": f"Film_{node_idx} (2024).mkv",
                    "size": 32000000000 + node_idx * 2048,
                    "duration": 7200.0,
                    "mediaInfo": {
                        "videoCodec": "hevc",
                        "audioFormat": "truehd",
                        "audioChannels": 7.1,
                        "audioLanguages": "eng",
                        "subtitles": "eng/spa",
                    },
                },
            })
            adapter_counts["movie_webhook"] += 1
        elif kind == 2:
            cand = AudioWebhookAdapter.parse_webhook({
                "eventType": "AlbumImported",
                "artist": {"name": f"Artist_{node_idx}", "audioCatalogId": f"mb-artist-{node_idx}"},
                "album": {"title": f"Album_{node_idx}", "audioCatalogId": f"mb-album-{node_idx}"},
                "trackFiles": [{"audioFormat": "FLAC", "size": 420000000, "bitrate": 950000}],
            })
            adapter_counts["audio_webhook"] += 1
        elif kind == 3:
            cand = PublicationWebhookAdapter.parse_webhook({
                "eventType": "BookFileImported",
                "book": {"title": f"Book_{node_idx}", "isbn": f"978000{node_idx:07d}"},
                "author": {"name": f"Author_{node_idx}"},
                "bookFiles": [{"format": "epub", "size": 2500000}],
            })
            adapter_counts["pub_ebook"] += 1
        elif kind == 4:
            cand = PublicationWebhookAdapter.parse_webhook({
                "eventType": "BookFileImported",
                "book": {"title": f"Audiobook_{node_idx}", "isbn": f"978111{node_idx:07d}"},
                "author": {"name": f"Narrator_{node_idx}"},
                "bookFiles": [{"format": "m4b", "size": 520000000}],
            })
            adapter_counts["pub_audiobook"] += 1
        elif kind == 5:
            cand = QBittorrentAdapter.parse_event({
                "hash": hashlib.sha1(f"qb_{node_idx}_{asn}".encode()).hexdigest(),
                "name": f"Documentary.{node_idx}.2023.1080p.BluRay.x264-GROUP",
                "size": 8500000000,
                "progress": 1.0,
                "state": "uploading",
                "content_path": f"/Users/alice/node_{node_idx}/Downloads/Documentary.{node_idx}.mkv",
            })
            ffprobe_payload = {
                "streams": [
                    {"codec_type": "video", "codec_name": "h264", "width": 1920, "height": 1080},
                    {"codec_type": "audio", "codec_name": "dts", "channels": 6, "tags": {"language": "eng"}},
                    {"codec_type": "subtitle", "tags": {"language": "eng"}},
                ],
                "format": {"format_name": "matroska,webm", "duration": "5400.0", "size": "8500000000"},
            }
            adapter_counts["qbittorrent_ffprobe"] += 1
        else:
            cand = TransmissionAdapter.parse_event({
                "TR_TORRENT_NAME": f"Series.Pack.{node_idx}.S02E05.2160p.WEB-DL.HEVC",
                "TR_TORRENT_HASH": hashlib.sha1(f"tr_{node_idx}_{asn}".encode()).hexdigest(),
                "TR_TORRENT_DIR": f"/home/bob/node_{node_idx}/torrents",
                "totalSize": 4100000000,
                "percentDone": 1.0,
                "isFinished": True,
            })
            ffprobe_payload = {
                "streams": [
                    {"codec_type": "video", "codec_name": "hevc", "width": 3840, "height": 2160},
                    {"codec_type": "audio", "codec_name": "eac3", "channels": 6, "tags": {"language": "eng"}},
                ],
                "format": {"format_name": "matroska,webm", "duration": "3100.0", "size": "4100000000"},
            }
            adapter_counts["transmission_ffprobe"] += 1

        ok, reason, rec = engine.ingest_candidate(
            cand, ffprobe_output=ffprobe_payload, current_timestamp=base_ts
        )
        if ok and rec is not None:
            ingested_records += 1

    ingest_elapsed_sec = time.perf_counter() - t0_ingest
    obs_per_sec = ingested_records / max(ingest_elapsed_sec, 1e-9)

    # Flush all 1,000 queued items after Poisson jitter window and measure AEAD encryption + decryption
    t0_enc = time.perf_counter()
    bundles = engine.get_pending_telemetry_batch(now_timestamp=base_ts + 86400.0)
    enc_elapsed_sec = time.perf_counter() - t0_enc

    t0_dec = time.perf_counter()
    decrypted_count = 0
    for idx, b in enumerate(bundles):
        plain = X25519ChaCha20Poly1305Engine.decrypt_bundle(
            b, agg_priv, expected_seq=501 + idx
        )
        parsed = json.loads(plain.decode("utf-8"))
        if "blinded_hash" in parsed and "dp_size_mb" in parsed:
            decrypted_count += 1
    dec_elapsed_sec = time.perf_counter() - t0_dec

    # Measure Ed25519 3-of-5 threshold verification latency (200 iterations)
    bundle_sample = json.dumps({"heads": {"category": {"temperature": 1.15}}}).encode("utf-8")
    sample_sha = hashlib.sha256(bundle_sample).hexdigest()
    m_body = {
        "bundle_format": "json",
        "bundle_sha256": sample_sha,
        "infohash": "aa" * 20,
        "min_asn_quorum": 5,
        "seq": 501,
        "timestamp": int(base_ts),
        "version": "2.1.0",
    }
    canon_m = json.dumps(m_body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    sigs_3_of_5 = {
        fed_pubs[i]: fed_privs[i].sign(canon_m).hex() for i in range(3)
    }
    pq_sigs_3_of_5 = {
        fed_pq_pubs[i]: WOTSPlusSHAKE256.sign(fed_pq_privs[i], canon_m).hex() for i in range(3)
    }
    valid_manifest = ThresholdDHTManifest(**m_body, signatures=sigs_3_of_5, pq_signatures=pq_sigs_3_of_5)

    t0_sig = time.perf_counter()
    for _ in range(200):
        s_ok, _ = ThresholdSignatureVerifier.verify(
            valid_manifest,
            trusted_signers,
            threshold_m=3,
            current_seq=500,
            trusted_pq_signers=trusted_pq_signers,
            require_pq=True,
        )
        assert s_ok
    sig_elapsed_sec = time.perf_counter() - t0_sig

    # 1,000-node peer attestations across all 50 ASNs (20 nodes per ASN)
    swarm_attestations = []
    # Pre-generate 50 ASN representative node keys (reused across the 20 peers in each ASN for fast deterministic simulation)
    asn_keys = [ed25519.Ed25519PrivateKey.generate() for _ in range(50)]
    asn_pubs = [k.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex() for k in asn_keys]
    for node_idx in range(1000):
        asn_slot = node_idx % 50
        asn_num = FIFTY_REAL_ASNS[asn_slot]
        peer_id = f"swarm_node_{node_idx:04d}_as{asn_num}"
        att_body = {
            "asn": int(asn_num),
            "peer_id": peer_id,
            "target_hash": sample_sha.lower(),
            "timestamp": int(base_ts),
        }
        att_canon = json.dumps(att_body, sort_keys=True, separators=(",", ":")).encode("utf-8")
        att_sig = asn_keys[asn_slot].sign(att_canon).hex()
        swarm_attestations.append(
            PeerAttestation(
                peer_id=peer_id,
                asn=asn_num,
                peer_pubkey_hex=asn_pubs[asn_slot],
                timestamp=int(base_ts),
                signature_hex=att_sig,
            )
        )

    quorum_ok, quorum_reason, verified_swarm_asns = SybilQuorumVerifier.verify_quorum(
        swarm_attestations, sample_sha, min_distinct_asns=50, now_timestamp=int(base_ts)
    )
    assert quorum_ok and len(verified_swarm_asns) == 50

    # Measure 500-item Golden Canary evaluation latency (20 full 500-item evaluations = 10,000 item evals)
    canary_gate = engine.canary_gate
    t0_canary = time.perf_counter()
    for _ in range(20):
        rep = canary_gate.evaluate(lambda item: dict(item.ground_truth))
        assert rep.passed
    canary_elapsed_sec = time.perf_counter() - t0_canary

    # -----------------------------------------------------------------
    # Part B: 200 Adversarial Red-Team Attack Payloads
    # -----------------------------------------------------------------
    redteam_counts: Dict[str, Dict[str, int]] = {}

    def record_attack(vector: str, blocked: bool) -> None:
        if vector not in redteam_counts:
            redteam_counts[vector] = {"total": 0, "blocked": 0}
        redteam_counts[vector]["total"] += 1
        if blocked:
            redteam_counts[vector]["blocked"] += 1

    # Vector 1: 20 Sybil Flood Campaigns (including a 500-fake-node single-ASN flood)
    for i in range(20):
        if i == 0:
            # 500 fake nodes all inside single ASN 16509
            sybil_atts = []
            for n in range(500):
                pk = ed25519.Ed25519PrivateKey.generate()
                pub_h = pk.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
                body = {"asn": 16509, "peer_id": f"sybil_{n}", "target_hash": sample_sha, "timestamp": int(base_ts)}
                sig_h = pk.sign(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hex()
                sybil_atts.append(PeerAttestation(f"sybil_{n}", 16509, pub_h, int(base_ts), sig_h))
        elif i < 10:
            # Multi-node floods distributed across only 1 to 4 ASNs (< quorum of 5)
            num_asns = (i % 4) + 1
            sybil_atts = []
            for n in range(50):
                asn_val = FIFTY_REAL_ASNS[n % num_asns]
                pk = ed25519.Ed25519PrivateKey.generate()
                pub_h = pk.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
                body = {"asn": asn_val, "peer_id": f"subquorum_{i}_{n}", "target_hash": sample_sha, "timestamp": int(base_ts)}
                sig_h = pk.sign(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hex()
                sybil_atts.append(PeerAttestation(f"subquorum_{i}_{n}", asn_val, pub_h, int(base_ts), sig_h))
        elif i < 15:
            # 10 distinct ASNs but stale timestamps (> 86400s old)
            sybil_atts = []
            for n in range(10):
                asn_val = FIFTY_REAL_ASNS[n]
                pk = ed25519.Ed25519PrivateKey.generate()
                pub_h = pk.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
                stale_ts = int(base_ts) - 200000 - i * 1000
                body = {"asn": asn_val, "peer_id": f"stale_{i}_{n}", "target_hash": sample_sha, "timestamp": stale_ts}
                sig_h = pk.sign(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hex()
                sybil_atts.append(PeerAttestation(f"stale_{i}_{n}", asn_val, pub_h, stale_ts, sig_h))
        else:
            # 10 distinct ASNs claiming ASNSpoof (signed over ASN 16509, claiming FIFTY_REAL_ASNS[n])
            sybil_atts = []
            for n in range(10):
                pk = ed25519.Ed25519PrivateKey.generate()
                pub_h = pk.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
                body = {"asn": 16509, "peer_id": f"spoof_{i}_{n}", "target_hash": sample_sha, "timestamp": int(base_ts)}
                sig_h = pk.sign(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hex()
                sybil_atts.append(PeerAttestation(f"spoof_{i}_{n}", FIFTY_REAL_ASNS[n], pub_h, int(base_ts), sig_h))

        q_ok, _, _ = SybilQuorumVerifier.verify_quorum(
            sybil_atts, sample_sha, min_distinct_asns=5, now_timestamp=int(base_ts)
        )
        record_attack("sybil_asn_floods", not q_ok)

    # Vector 2: 25 Replay / Downgrade BEP-44 seq Attacks
    for i in range(25):
        bad_seq = 500 - i if i < 20 else -(i * 10)
        bad_m = ThresholdDHTManifest(
            seq=bad_seq,
            version="2.1.0",
            infohash="aa" * 20,
            bundle_sha256=sample_sha,
            bundle_format="json",
            min_asn_quorum=5,
            timestamp=int(base_ts),
            signatures=sigs_3_of_5,
        )
        m_ok, _ = ThresholdSignatureVerifier.verify(
            bad_m, trusted_signers, threshold_m=3, current_seq=500
        )
        record_attack("replay_downgrade_seq", not m_ok)

    # Vector 3: 25 Forged / Insufficient Threshold Signatures
    for i in range(25):
        if i < 8:
            # Only 0, 1, or 2 valid signatures when threshold_m=3
            partial_sigs = {fed_pubs[j]: sigs_3_of_5[fed_pubs[j]] for j in range(i % 3)}
            forged_m = ThresholdDHTManifest(**m_body, signatures=partial_sigs)
        elif i < 16:
            # 5 signatures from untrusted rogue Ed25519 keys
            rogue_sigs = {}
            for r in range(5):
                rk = ed25519.Ed25519PrivateKey.generate()
                rpub = rk.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
                rogue_sigs[rpub] = rk.sign(canon_m).hex()
            forged_m = ThresholdDHTManifest(**m_body, signatures=rogue_sigs)
        else:
            # Tampered manifest payload with original valid signatures
            tampered_body = dict(m_body)
            tampered_body["bundle_sha256"] = hashlib.sha256(f"evil_{i}".encode()).hexdigest()
            forged_m = ThresholdDHTManifest(**tampered_body, signatures=sigs_3_of_5)

        m_ok, _ = ThresholdSignatureVerifier.verify(
            forged_m, trusted_signers, threshold_m=3, current_seq=500
        )
        record_attack("forged_threshold_sigs", not m_ok)

    # Vector 4: 25 Tampered ChaCha20-Poly1305 Ciphertexts / Nonces / Keys
    for i in range(25):
        orig = bundles[i]
        if i < 15:
            # Flip nibble in ciphertext/tag at position i * 4
            ct_list = list(orig.ciphertext_hex)
            pos = (i * 4) % len(ct_list)
            ct_list[pos] = "f" if ct_list[pos] != "f" else "0"
            bad_b = EncryptedTelemetryBundle(
                ephemeral_pubkey_hex=orig.ephemeral_pubkey_hex,
                nonce_hex=orig.nonce_hex,
                ciphertext_hex="".join(ct_list),
                aad=orig.aad,
                seq=orig.seq,
            )
        elif i < 20:
            # Flip nibble in 96-bit nonce
            n_list = list(orig.nonce_hex)
            n_list[i % len(n_list)] = "a" if n_list[i % len(n_list)] != "a" else "1"
            bad_b = EncryptedTelemetryBundle(
                ephemeral_pubkey_hex=orig.ephemeral_pubkey_hex,
                nonce_hex="".join(n_list),
                ciphertext_hex=orig.ciphertext_hex,
                aad=orig.aad,
                seq=orig.seq,
            )
        else:
            # Truncated ciphertext/tag
            bad_b = EncryptedTelemetryBundle(
                ephemeral_pubkey_hex=orig.ephemeral_pubkey_hex,
                nonce_hex=orig.nonce_hex,
                ciphertext_hex=orig.ciphertext_hex[:16],
                aad=orig.aad,
                seq=orig.seq,
            )

        blocked = False
        try:
            X25519ChaCha20Poly1305Engine.decrypt_bundle(bad_b, agg_priv)
        except ValueError:
            blocked = True
        record_attack("tampered_aead_ciphertext", blocked)

    # Vector 5: 25 Cross-Epoch AAD & Sequence Transplants
    for i in range(25):
        orig = bundles[i]
        other = bundles[(i + 1) % len(bundles)]
        if i < 8:
            # Transplant AAD from another bundle epoch
            bad_b = EncryptedTelemetryBundle(
                ephemeral_pubkey_hex=orig.ephemeral_pubkey_hex,
                nonce_hex=orig.nonce_hex,
                ciphertext_hex=orig.ciphertext_hex,
                aad=other.aad,
                seq=orig.seq,
            )
            exp_seq = None
        elif i < 16:
            # Mutate bundle.seq to next epoch while keeping old AAD
            bad_b = EncryptedTelemetryBundle(
                ephemeral_pubkey_hex=orig.ephemeral_pubkey_hex,
                nonce_hex=orig.nonce_hex,
                ciphertext_hex=orig.ciphertext_hex,
                aad=orig.aad,
                seq=orig.seq + 100,
            )
            exp_seq = None
        elif i < 21:
            # Mutate both bundle.seq and bundle.aad to forged epoch without re-encrypting
            forged_seq = orig.seq + 500
            bad_b = EncryptedTelemetryBundle(
                ephemeral_pubkey_hex=orig.ephemeral_pubkey_hex,
                nonce_hex=orig.nonce_hex,
                ciphertext_hex=orig.ciphertext_hex,
                aad=f"seq:{forged_seq};ephem_pub:{orig.ephemeral_pubkey_hex}",
                seq=forged_seq,
            )
            exp_seq = None
        else:
            # Replay valid bundle from seq=orig.seq into expected_seq=orig.seq + 1
            bad_b = orig
            exp_seq = orig.seq + 1

        blocked = False
        try:
            X25519ChaCha20Poly1305Engine.decrypt_bundle(bad_b, agg_priv, expected_seq=exp_seq)
        except ValueError:
            blocked = True
        record_attack("cross_epoch_aad_transplant", blocked)

    # Vector 6: 20 Embedded Python Pickle RCE & Malformed Format Payloads
    pickle_rce_payloads = [
        (b"\x80\x02cos\nsystem\n(S'id'\ntR.", "safetensors"),
        (b"\x80\x03cposix\nsystem\n(X\x02\x00\x00\x00idq\x00\x85q\x01Rq\x02.", "safetensors"),
        (b"\x80\x04\x95\x1e\x00\x00\x00\x00\x00\x00\x00cos\nsystem\n(X\x06\x00\x00\x00whoami\x85R.", "onnx"),
        (b"\x80\x05\x95\x15\x00\x00\x00\x00\x00\x00\x00c__builtin__\neval\n.", "json"),
        (b'{"__reduce__": "cos\nsystem"}', "json"),
        (b'{"exploit": "c__builtin__"}', "json"),
        (b'{"exploit": "cposix\nsystem"}', "json"),
        # SafeTensors with pickle hidden inside tensor buffer
        (
            len(b'{"w":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}').to_bytes(8, "little")
            + b'{"w":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}'
            + b"\x80\x04evil12",
            "safetensors",
        ),
        # SafeTensors out-of-bounds offset
        (
            len(b'{"w":{"dtype":"F32","shape":[2],"data_offsets":[0,4096]}}').to_bytes(8, "little")
            + b'{"w":{"dtype":"F32","shape":[2],"data_offsets":[0,4096]}}'
            + b"\x00" * 8,
            "safetensors",
        ),
        # SafeTensors negative offset
        (
            len(b'{"w":{"dtype":"F32","shape":[2],"data_offsets":[-4,4]}}').to_bytes(8, "little")
            + b'{"w":{"dtype":"F32","shape":[2],"data_offsets":[-4,4]}}'
            + b"\x00" * 8,
            "safetensors",
        ),
        # SafeTensors inverted offsets
        (
            len(b'{"w":{"dtype":"F32","shape":[2],"data_offsets":[8,4]}}').to_bytes(8, "little")
            + b'{"w":{"dtype":"F32","shape":[2],"data_offsets":[8,4]}}'
            + b"\x00" * 8,
            "safetensors",
        ),
        # SafeTensors disallowed dtype
        (
            len(b'{"w":{"dtype":"PICKLE_OBJ","shape":[2],"data_offsets":[0,8]}}').to_bytes(8, "little")
            + b'{"w":{"dtype":"PICKLE_OBJ","shape":[2],"data_offsets":[0,8]}}'
            + b"\x00" * 8,
            "safetensors",
        ),
        # SafeTensors header overflow
        ((99999999).to_bytes(8, "little") + b"{}", "safetensors"),
        # SafeTensors too short
        (b"short", "safetensors"),
        # ONNX wrong magic
        (b"\xff\xff\xff\xff" + b"\x00" * 32, "onnx"),
        # ONNX too short
        (b"\x08\x01", "onnx"),
        # JSON array instead of dict
        (b"[1, 2, 3]", "json"),
        # Unsupported formats
        (b"raw_bytes", "pickle"),
        (b"raw_bytes", "pth"),
        (b"raw_bytes", "joblib"),
    ]
    for payload_bytes, fmt_name in pickle_rce_payloads:
        f_ok, _ = SafeFormatValidator.validate(payload_bytes, fmt_name)
        record_attack("pickle_rce_and_format_poison", not f_ok)

    # Vector 7: 20 Malware .mkv.exe / .scr / Password Archive Masquerades
    masquerade_titles = [
        ("Tears.of.Steel.2012.2160p.WEB-DL.mkv.exe", "movie"),
        ("Cosmos.Laundromat.S01E01.1080p.WEB-DL.mp4.scr", "tv_show"),
        ("Big.Buck.Bunny.2008.1080p.BluRay.avi.vbs", "movie"),
        ("Sintel.2010.1080p.mov.bat", "movie"),
        ("Open.Science.Lectures.S03E01.720p.mkv.cmd", "tv_show"),
        ("PublicArchive.2024.1080p.mp4.lnk", "movie"),
        ("Kimiko Ishizaka.Open.Goldberg.Variations.FLAC.flac.exe", "music"),
        ("Open.Goldberg.Variations.mp3.scr", "music"),
        ("Frankenstein.Mary.Shelley.epub.vbs", "ebook"),
        ("Pride.and.Prejudice.LibriVox.Audiobook.m4b.bat", "audiobook"),
        ("Viral.Clip.2024.wmv.pif", "movie"),
        ("Anime.Episode.01.1080p.webm.ps1", "tv_show"),
        ("Codec_Pack_Required_To_Watch_4K.scr", "movie"),
        ("VLC_HDR_Plugin_Updater.vbs", "movie"),
        ("Install_Dolby_Atmos_Decoder.bat", "movie"),
        ("OpenMovie.2025.Fake.Screener.1080p.exe", "movie"),
        ("Blockbuster.2024.1080p.password_in_description.mkv", "movie"),
        ("Premiere.S01E01.pass in desc.mp4", "tv_show"),
        ("Exclusive.Release.2024.pw txt.mkv", "movie"),
        ("Early.Access.Movie.2024.unlock_code.mkv", "movie"),
    ]
    for m_title, m_cat in masquerade_titles:
        m_cand = CandidateObservation(
            source_adapter="qbittorrent",
            raw_title=m_title,
            claimed_category=m_cat,
            download_completed=True,
            bytes_downloaded=15000000,
            total_bytes=15000000,
        )
        v_res = engine.verifier.verify(m_cand)
        isolated_as_junk = bool(
            v_res.is_verified
            and v_res.observation is not None
            and v_res.observation.junk is True
            and v_res.observation.owner_scope == "excluded"
        )
        record_attack("malware_masquerades", isolated_as_junk)

    # Vector 8: 20 Private Tracker Passkey / IP / Path Leak Attempts
    for i in range(20):
        secret_pk = f"SECRET_PASSKEY_{i:04d}_ deadbeef99887766".replace(" ", "")
        secret_ip4 = f"198.51.100.{(i % 250) + 1}"
        secret_ip6 = f"2001:0db8:85a3:0000:0000:8a2e:0370:{7300 + i:04x}"
        secret_user = f"privateuser_{i}"
        raw_leaky_title = (
            f"Movie.Title.{2000 + i}.1080p.BluRay.x264-GROUP "
            f"http://tracker.example.org/announce?passkey={secret_pk}&authkey={secret_pk} "
            f"peer={secret_ip4} ipv6={secret_ip6} path=/Users/{secret_user}/Downloads "
            f"win=C:\\Users\\{secret_user}\\Videos"
        )
        leaky_cand = CandidateObservation(
            source_adapter="movie_webhook",
            raw_title=raw_leaky_title,
            claimed_category="movie",
            download_completed=True,
            bytes_downloaded=4000000000,
            total_bytes=4000000000,
            file_path=f"/Users/{secret_user}/Movies/Movie.{2000 + i}.mkv",
            catalog_id_type="movie_catalog",
            catalog_id_value=10000 + i,
            stream_metadata={
                "video_codec": f"hevc?passkey={secret_pk}",
                "audio_codec": f"truehd from {secret_ip4} /home/{secret_user}",
                "audio_channels": 7.1,
                "audio_languages": "eng",
                "subtitles": "eng",
                "duration_seconds": 7200.0,
            },
        )
        ok_l, _, rec_l = engine.ingest_candidate(leaky_cand, current_timestamp=base_ts)
        assert ok_l and rec_l is not None
        serialized = rec_l.canonical_json_bytes().decode("utf-8")
        no_leak = (
            secret_pk not in serialized
            and secret_ip4 not in serialized
            and secret_ip6 not in serialized
            and secret_user not in serialized
        )
        record_attack("passkey_ip_path_leak_attempts", no_leak)

    # Vector 9: 20 Poisoned Weight Updates Targeting the 500-Item Golden Canary Gate
    for i in range(20):
        if i < 8:
            # Malware trap false-negative backdoor on trap index i
            target_trap = f"canary_jnk_{i:03d}"
            def poisoned_fn(item: CanaryItem, trap_id: str = target_trap) -> Dict[str, Any]:
                pred = dict(item.ground_truth)
                if item.item_id == trap_id:
                    pred["junk"] = False
                return pred
        elif i < 12:
            # Clean items falsely marked as junk, dropping junk_balanced_accuracy < 0.999
            def poisoned_fn(item: CanaryItem) -> Dict[str, Any]:
                pred = dict(item.ground_truth)
                if item.item_id.startswith("canary_mov_00"):
                    pred["junk"] = True
                return pred
        elif i < 16:
            # Adult boundary degradation (< 0.985 accuracy)
            def poisoned_fn(item: CanaryItem) -> Dict[str, Any]:
                pred = dict(item.ground_truth)
                if item.is_adult_boundary:
                    pred["adult"] = False
                return pred
        else:
            # Category accuracy regression (> 0.005 drop from baseline 1.0)
            def poisoned_fn(item: CanaryItem) -> Dict[str, Any]:
                pred = dict(item.ground_truth)
                if item.item_id.startswith("canary_tv_00"):
                    pred["category"] = "unknown"
                return pred

        rep = canary_gate.evaluate(
            poisoned_fn,
            baseline_metrics={"category_accuracy": 1.0},
        )
        record_attack("poisoned_canary_updates", not rep.passed)

    total_attacks = sum(v["total"] for v in redteam_counts.values())
    total_blocked = sum(v["blocked"] for v in redteam_counts.values())

    return {
        "swarm_simulation": {
            "total_nodes": 1000,
            "distinct_asns": 50,
            "verified_quorum_asns": len(verified_swarm_asns),
            "ingested_verified_records": ingested_records,
            "decrypted_verified_bundles": decrypted_count,
            "adapter_breakdown": adapter_counts,
        },
        "latency_and_throughput": {
            "ingest_1000_nodes_total_ms": round(ingest_elapsed_sec * 1000.0, 3),
            "ingest_throughput_obs_per_sec": round(obs_per_sec, 1),
            "aead_encrypt_per_bundle_ms": round((enc_elapsed_sec / 1000.0) * 1000.0, 4),
            "aead_decrypt_per_bundle_ms": round((dec_elapsed_sec / 1000.0) * 1000.0, 4),
            "ed25519_3of5_verify_ms": round((sig_elapsed_sec / 200.0) * 1000.0, 4),
            "golden_canary_500_eval_ms": round((canary_elapsed_sec / 20.0) * 1000.0, 4),
        },
        "adversarial_redteam": {
            "total_attack_payloads": total_attacks,
            "total_blocked": total_blocked,
            "block_rate_pct": round((total_blocked / max(total_attacks, 1)) * 100.0, 2),
            "vectors": redteam_counts,
        },
    }


class TestThousandNodeSwarmAndAdversarialRedTeam(unittest.TestCase):
    """Phase 2: Large-Scale 1,000-Node 50-ASN Swarm Simulation & 200-Payload Red-Team Stress Suite."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.report = run_swarm_and_redteam_benchmark()

    def test_1000_node_swarm_across_50_asns_full_fidelity(self):
        swarm = self.report["swarm_simulation"]
        self.assertEqual(swarm["total_nodes"], 1000)
        self.assertEqual(swarm["distinct_asns"], 50)
        self.assertEqual(swarm["verified_quorum_asns"], 50)
        self.assertEqual(swarm["ingested_verified_records"], 1000)
        self.assertEqual(swarm["decrypted_verified_bundles"], 1000)
        self.assertEqual(sum(swarm["adapter_breakdown"].values()), 1000)

    def test_redteam_vector_1_sybil_asn_floods_including_500_fake_nodes(self):
        v = self.report["adversarial_redteam"]["vectors"]["sybil_asn_floods"]
        self.assertEqual(v["total"], 20)
        self.assertEqual(v["blocked"], 20)

    def test_redteam_vector_2_replay_and_downgrade_seq_attacks(self):
        v = self.report["adversarial_redteam"]["vectors"]["replay_downgrade_seq"]
        self.assertEqual(v["total"], 25)
        self.assertEqual(v["blocked"], 25)

    def test_redteam_vector_3_forged_threshold_signatures(self):
        v = self.report["adversarial_redteam"]["vectors"]["forged_threshold_sigs"]
        self.assertEqual(v["total"], 25)
        self.assertEqual(v["blocked"], 25)

    def test_redteam_vector_4_tampered_aead_ciphertexts(self):
        v = self.report["adversarial_redteam"]["vectors"]["tampered_aead_ciphertext"]
        self.assertEqual(v["total"], 25)
        self.assertEqual(v["blocked"], 25)

    def test_redteam_vector_5_cross_epoch_aad_transplants(self):
        v = self.report["adversarial_redteam"]["vectors"]["cross_epoch_aad_transplant"]
        self.assertEqual(v["total"], 25)
        self.assertEqual(v["blocked"], 25)

    def test_redteam_vector_6_pickle_rce_and_format_poison(self):
        v = self.report["adversarial_redteam"]["vectors"]["pickle_rce_and_format_poison"]
        self.assertEqual(v["total"], 20)
        self.assertEqual(v["blocked"], 20)

    def test_redteam_vector_7_malware_mkv_exe_scr_masquerades(self):
        v = self.report["adversarial_redteam"]["vectors"]["malware_masquerades"]
        self.assertEqual(v["total"], 20)
        self.assertEqual(v["blocked"], 20)

    def test_redteam_vector_8_passkey_ip_and_path_leak_attempts(self):
        v = self.report["adversarial_redteam"]["vectors"]["passkey_ip_path_leak_attempts"]
        self.assertEqual(v["total"], 20)
        self.assertEqual(v["blocked"], 20)

    def test_redteam_vector_9_poisoned_canary_weight_updates(self):
        v = self.report["adversarial_redteam"]["vectors"]["poisoned_canary_updates"]
        self.assertEqual(v["total"], 20)
        self.assertEqual(v["blocked"], 20)

    def test_unified_200_adversarial_payloads_and_latency_slas(self):
        rt = self.report["adversarial_redteam"]
        self.assertEqual(rt["total_attack_payloads"], 200)
        self.assertEqual(rt["total_blocked"], 200)
        self.assertEqual(rt["block_rate_pct"], 100.0)

        perf = self.report["latency_and_throughput"]
        self.assertGreater(perf["ingest_throughput_obs_per_sec"], 1000.0)
        self.assertLess(perf["aead_encrypt_per_bundle_ms"], 5.0)
        self.assertLess(perf["aead_decrypt_per_bundle_ms"], 5.0)
        self.assertLess(perf["ed25519_3of5_verify_ms"], 10.0)
        self.assertLess(perf["golden_canary_500_eval_ms"], 50.0)


if __name__ == "__main__":
    import sys
    if "--benchmark" in sys.argv:
        print(json.dumps(run_swarm_and_redteam_benchmark(), indent=2))
    else:
        unittest.main()

