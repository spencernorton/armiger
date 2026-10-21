<h1 align="center">Armiger Plugin — Modular, CSAM-Protected, Encrypted DHT Database Sync (<code>armiger-dht-db-sync</code>)</h1>

<p align="center">
  <strong>Decentralized <code>BEP-44</code> / <code>BEP-46</code> database synchronization engine featuring multi-layer CSAM &amp; abuse blocking, <code>Ed25519</code>-signed blinded <code>SHA-256</code> tombstone purging, modular fine-grained category &amp; language subscriptions with isolated <code>xxx</code> containment, <code>X25519</code> + <code>ChaCha20-Poly1305</code> AEAD encryption, and $K$-of-$N$ distinct-ASN Sybil quorum verification.</strong><br>
  Designed for decentralized indexers (<code>Bitmagnet</code> and self-hosted library fleets) powered by <a href="https://huggingface.co/NorviTech/armiger-ultra"><code>NorviTech/armiger-ultra</code></a>.
</p>

<p align="center">
  <a href="https://huggingface.co/NorviTech/armiger-ultra"><img alt="Compatible with Armiger Ultra" src="https://img.shields.io/badge/Bundled_with-Armiger_Ultra-FD8024.svg"></a>
  <a href="https://huggingface.co/NorviTech/armiger"><img alt="Compatible with Armiger" src="https://img.shields.io/badge/Compatible_with-Armiger_CPU-0284C7.svg"></a>
  <img alt="Licence Apache-2.0" src="https://img.shields.io/badge/licence-Apache--2.0-blue.svg">
  <img alt="Tests Passing" src="https://img.shields.io/badge/tests-27%2F27%20passing-brightgreen.svg">
</p>

---

## Architecture & Security Invariants

### 1. Multi-Layer CSAM & Harmful-Content Protection
- **Multi-Stage ContentFilter Parity:** Full unicode NFKD decomposition, diacritic stripping, zero-width / bidirectional formatting removal, and leetspeak deobfuscation (`0`→`o`, `1`→`i`, `3`→`e`, `4`/`@`→`a`, `$`→`s`, `7`/`+`→`t`).
- **Deterministic & Structural Pre-Ingest Blocker:** Inspects titles, file lists, and tags before shard generation. Blocks child sexual abuse material (CSAM / CSAE), non-consensual sexual exploitation, and extreme violence / snuff. Disguised executable droppers (`.mp4.exe`, `.scr`, `.bat`, `.vbs`, `.ps1`) in media categories are unconditionally rejected.
- **Ed25519-Signed Blinded SHA-256 Tombstone Purge Engine:** Revocations use `SHA-256("armiger-tombstone-v1:" || infohash)`. Manifests require strictly increasing sequence counters (`seq > current_seq`) and valid Ed25519 root signatures. Nodes automatically sweep and purge matching entries from local databases and ban re-ingestion.
- **Mandatory Armiger Conformal Safety Gate:** Mondrian conformal gates (`junk < 0.1010`, `scope < 0.4770`, `adult >= 0.5451` routed strictly to adult containment).

### 2. Modular Sharded Subscription Architecture
- **Partition Dimensions:** Partitioned across Category (10 modules) $\times$ Language (14 modules) $\times$ Quality Tier (7 tiers) $\times$ Time Epoch.
- **10 Category Modules:** `movie`, `tv_show`, `music`, `game`, `software`, `anime`, `book`, `comic`, `course`, and `xxx`.
- **Strict `xxx` Isolation & Opt-In:** Adult content is **strictly disabled by default** (`enable_xxx=False`). Adult shards use a segregated namespace and storage table (`adult_torrents`). Mainstream searches never query or return adult records, and mainstream shards never carry adult entries.
- **Selective Shard Download:** Subscribers configure exact subsets (e.g. `categories={"movie", "tv_show"}`, `languages={"en", "ja"}`). DHT lookups only fetch subscribed shards—zero network bandwidth or storage is spent on unwanted modules.

### 3. X25519 + ChaCha20-Poly1305 AEAD Encryption
- **Key Derivation:** HKDF-SHA256 derives 32-byte ChaCha20-Poly1305 keys salted by `shard_id` and bounded by `epoch`.
- **Authenticated Associated Data (AAD):** `armiger-shard-aad-v1:{shard_id}:{epoch}:{seq}:{merkle_root}` is bound to every ciphertext block.
- **Anti-Transplant & Replay Defense:** Any tampering, sequence alteration, or cross-shard ciphertext transplant immediately fails Poly1305 authentication.

### 4. BEP-44 / BEP-46 Root-of-Trust + K-of-N Distinct-ASN Quorum
- **BEP-44 Mutable DHT Pointers:** Target key calculated via `SHA1(pubkey || salt)`.
- **BEP-46 Infohash Pointers:** Canonical infohash references encrypted micro-shards.
- **Sybil-Resistant $K$-of-$N$ Quorum:** Requires $K \ge 5$ cryptographically valid Ed25519 attestations from $\ge 5$ **distinct Autonomous System Numbers (ASNs)** (e.g., AS15169 Google, AS13335 Cloudflare, AS16509 Amazon, AS7922 Comcast, AS3356 Lumen). Multi-node Sybil attacks within the same ASN or /24 subnet are automatically rejected.
- **Binary Merkle Tree Verification:** Each shard payload forms a binary Merkle tree of canonical record hashes. Manifest carries signed `merkle_root`. Supports individual inclusion proofs for auditability.

### 5. Anonymity & Dynamic Opt-In/Opt-Out
- **Zero PII:** No IP addresses, user tokens, passkeys, or client headers are stored or broadcast.
- **Sanitized Metadata:** Private tracker passkeys (`[passkey=...]`, `?passkey=...`) are stripped prior to hashing.
- **Sync Modes:** `OFF`, `PULL_ONLY` (download only), `SUBSCRIBE_AND_SEED` (distribute subscribed shards), `PUBLISHER`.

---

## Quickstart

```python
from armiger_dht_db_sync import (
    ArmigerDHTDatabaseSync,
    SubscriptionConfig,
    SyncMode,
    create_mock_quorum_attestations,
    generate_ed25519_keypair,
)
from csam_guard import sign_tombstone_list

# 1. Initialize root keys and subscription
root_priv, root_pub = generate_ed25519_keypair()
sub_cfg = SubscriptionConfig(
    categories={"movie", "anime"},
    languages={"en", "ja"},
    quality_tiers={"uhd_4k", "hd_1080p"},
    epochs={"2026_q4"},
    enable_xxx=False,  # Adult content strictly excluded
    sync_mode=SyncMode.PULL_ONLY,
)

sync_engine = ArmigerDHTDatabaseSync(
    subscription=sub_cfg,
    trusted_root_pubkeys_hex=[root_pub.hex()],
)

# 2. Ingest candidate record with CSAM & safety checks
ok, msg, record = sync_engine.ingest_candidate_record(
    infohash="0123456789abcdef0123456789abcdef01234567",
    title="Spring.2019.1080p.WEBM.AV1-OPEN",
    category="anime",
    language="ja",
    quality_tier="hd_1080p",
    epoch="2026_q4",
    p_adult=0.01,
    p_junk=0.001,
    p_excluded_scope=0.01,
)
print("Ingestion result:", ok, msg)

# 3. Search local database
results = sync_engine.db.search(query="Spring", category="anime")
print("Found torrents:", len(results))

# 4. Apply Ed25519-signed blinded tombstone purge
tombstone_manifest = sign_tombstone_list(
    seq=1,
    infohashes_or_blinded=["0123456789abcdef0123456789abcdef01234567"],
    private_key_bytes=root_priv,
)
ok, msg, new_count, purged = sync_engine.apply_signed_tombstones(tombstone_manifest)
print("Tombstone purge executed. Purged records:", purged)
```

---

## Test Suite Execution

Run the comprehensive 27-point test suite:

```bash
python3 test_dht_db_sync.py -v
```

All 27 test cases pass cleanly:
```text
test_ciphertext_tampering_defense ... ok
test_cross_shard_transplant_defense ... ok
test_key_derivation_deterministic_and_unique ... ok
test_round_trip_encryption_decryption ... ok
test_merkle_tree_construction_and_inclusion_proofs ... ok
test_quorum_verification_success ... ok
test_sybil_attack_rejected_duplicate_asns ... ok
test_armiger_mondrian_safety_gate ... ok
test_direct_csam_detection ... ok
test_disguised_malware_dropper_in_media_torrents ... ok
test_nonconsensual_and_extreme_violence_detection ... ok
test_private_tracker_passkey_sanitization ... ok
test_signed_tombstone_purge_lifecycle ... ok
test_text_normalization_and_leetspeak ... ok
test_bep44_target_derivation ... ok
test_ciphertext_sha256_mismatch_rejected ... ok
test_full_publish_subscribe_and_shard_application ... ok
test_merkle_root_mismatch_rejected ... ok
test_pre_ingest_reinspection_drops_bad_records_in_shard ... ok
test_stale_or_downgrade_sequence_rejected ... ok
test_sync_mode_off_disables_all_operations ... ok
test_untrusted_root_key_rejected ... ok
test_x25519_ecdh_key_exchange ... ok
test_database_table_segregation ... ok
test_selective_shard_filtering ... ok
test_strict_xxx_isolation_in_subscription ... ok
test_taxonomy_definitions ... ok

Ran 27 tests in 0.011s - OK
```

---

## Part of the Armiger Modular Plugin Suite

| Plugin Repository | Bundled in `Armiger Ultra`? | Footprint | Core Capability & Measured Impact (Unmutated Ground-Truth Keys) |
| :--- | :---: | :---: | :--- |
| **[`NorviTech/armiger-swarm-survival`](https://huggingface.co/NorviTech/armiger-swarm-survival)** | **Yes** (`swarm_adapter.pt`, `swarm_survival.py`) | `167 KB` (`132 KB` `.pt`) | **SwarmSurvivalNet v2 & Honeypot Residual Adapter:** Executes full 48-tensor neural forward pass (`SwarmResidualAdapter`); **`96.83%`** Multi-Source Corpus AUROC (`N=10,278`, vs. `53.02%` raw seeders), **`92.42%`** Public-Tracker AUROC (`N=757`, vs. `85.17%` raw seeders), disguised-executable extension filtering, seeder-independent content categorization. |
| **[`NorviTech/armiger-swarm-priors`](https://huggingface.co/NorviTech/armiger-swarm-priors)** | **Yes** (`swarm_priors.py`) | `29.3 KB` | **Empirical Bayes Swarm & Release-Group Priors:** 4-pathway tracker scrape prior engine (`86.64%` formula AUROC), mandatory modality dropout (`p=0.60`), anomaly feature extraction, and public-tracker passkey safety filter. |
| **[`NorviTech/armiger-zero-cue-catalog`](https://huggingface.co/NorviTech/armiger-zero-cue-catalog)** | **Yes** (`zero_cue_plugin.py`, `validated_zero_cue_catalog.json`) | `26.0 MB` (`BLAKE2b-64`) | **Zero-Cue Catalog & Structural Disambiguator:** Offline `931,142`-entry `BLAKE2b-64` cryptographically hashed catalog (`472,445` `(title, year)`, `256,963` anime stems, `201,734` long titles) + deterministic structural rules; lifts 5-split (`N=10,262`) weighted category accuracy from **`93.00%`** (Standalone Neural) to **`94.63%`** (Fair Blind Hybrid), `beyond_confirmation` (`N=1,025`) from **`91.61%`** to **`93.07%`**, and `zero_cue_2000` (`N=2,000`) from **`85.55%`** to **`91.00%`** (`+5.45 pp`). |
| **[`NorviTech/armiger-audio-guard`](https://huggingface.co/NorviTech/armiger-audio-guard)** | **Yes** (`audio_language_guard.py`) | `3.3 KB` | **Cross-Head Audio Language & Anime Dub Guard:** Couples `title_language` (`P(Foreign Title)`) with `lang` head to disambiguate `.en.` prepositions, `<Foreign>.ENG.Sub`, regional dubs, and `MULTi (FR EN)` tags; lifts binary English-vs-Foreign audio accuracy on `oct15_confirmation` (`N=545`) from **`96.33%`** (Standalone INT8 Neural) to **`96.70%`**. |
| **[`NorviTech/armiger-calibration-dht`](https://huggingface.co/NorviTech/armiger-calibration-dht)** | **Yes** (`calibration.json`, `calibration_and_dht.py`, `armiger_dht_rsi.py`) | `181 KB` | **Conformal Calibration, BEP-44 DHT Updater & Distributed RSI Engine:** Per-head temperature scaling ($T_h$), Mondrian safety gates (**`99.23%`** balanced junk/sample accuracy / **`99.90%`** overall accuracy on `beyond_confirmation` `N=977`, `65` positive items), Ed25519-signed `BEP-44`/`BEP-46` DHT hot-reload, `X25519`+`ChaCha20-Poly1305` AEAD, and 500-canary regression gate (repository BEP/DHT test battery). |
| **[`NorviTech/armiger-dht-db-sync`](https://huggingface.co/NorviTech/armiger-dht-db-sync)** | **Yes** (`csam_guard.py`, `armiger_dht_db_sync.py`) | `155 KB` | **Modular, CSAM-Protected, Encrypted BEP-44/BEP-46 DHT Database Sync:** Multi-stage CSAM & abuse pre-ingest blocker, Ed25519-signed blinded `SHA-256` tombstone purge engine, sharded subscriptions with default-off `xxx` isolation, and `X25519`+`ChaCha20-Poly1305` AEAD (repository BEP/DHT test battery). |

---

## Intended Use & Legal Compliance

`Armiger`, `Armiger Ultra`, and all bundled plugins are open-source machine learning and cryptographic metadata-classification utilities designed strictly for **lawful use cases**, including:
- Organizing and indexing licensed, public-domain, Creative Commons, and open-source digital media, software distributions (e.g., Linux ISOs, open-source games), and educational archives (e.g., OpenCourseWare, LibriVox, Project Gutenberg).
- Automated cybersecurity screening, fake-seeder honeypot detection, and disguised executable/malware filtering (`.mkv.exe`, `.scr`, `.vbs`) in peer-to-peer and decentralized storage research.
- Child safety (CSAM/CSAE) pre-ingest blocking, cryptographic tombstone revocation, and adult/NSFW namespace isolation for self-hosted and decentralized indexers.

**NorviTech does not condone, encourage, or support copyright infringement or the unauthorized distribution of copyrighted works.** Users are solely responsible for ensuring that their use of these models and plugins complies with all applicable local, national, and international copyright and data-protection laws.

---

<p align="center">
  <a href="https://norvitech.com"><img alt="Part of the NorviTech Suite — open-source apps for the Linux desktop and the self-hosted stack" src="https://norvitech.com/assets/banner.svg" width="640"></a>
</p>

<p align="center">
  <a href="https://huggingface.co/NorviTech/armiger-ultra">Armiger Ultra</a> ·
  <a href="https://huggingface.co/NorviTech/armiger">Armiger</a> ·
  <a href="https://huggingface.co/NorviTech/armiger-swarm-survival">Swarm Survival</a> ·
  <a href="https://huggingface.co/NorviTech/armiger-swarm-priors">Swarm Priors</a> ·
  <a href="https://huggingface.co/NorviTech/armiger-zero-cue-catalog">Zero-Cue Catalog</a> ·
  <a href="https://huggingface.co/NorviTech/armiger-audio-guard">Audio Guard</a> ·
  <a href="https://huggingface.co/NorviTech/armiger-calibration-dht">Calibration &amp; DHT RSI</a> ·
  <a href="https://huggingface.co/NorviTech/armiger-dht-db-sync">DHT DB Sync</a> ·
  <a href="https://norvitech.com">norvitech.com</a>
</p>
