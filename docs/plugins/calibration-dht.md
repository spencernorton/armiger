<h1 align="center">Armiger Plugin #5 — Conformal Calibration &amp; BEP-44 Decentralized DHT Updater (<code>armiger-calibration-dht</code>)</h1>

<p align="center">
  <strong>Per-head temperature scaling, Menon prior adjustments, Mondrian conformal safety gates (<code>99.95%</code> junk balanced accuracy), and an <code>Ed25519</code>-signed <code>BEP-44</code> / <code>BEP-46</code> BitTorrent DHT update engine.</strong><br>
  Available standalone here and pre-bundled inside <a href="https://huggingface.co/NorviTech/armiger-ultra"><code>NorviTech/armiger-ultra</code></a>.
</p>

<p align="center">
  <a href="https://huggingface.co/NorviTech/armiger-ultra"><img alt="Bundled in Armiger Ultra" src="https://img.shields.io/badge/Bundled_in-Armiger_Ultra-FD8024.svg"></a>
  <a href="https://huggingface.co/NorviTech/armiger"><img alt="Compatible with Armiger" src="https://img.shields.io/badge/Compatible_with-Armiger_CPU-0284C7.svg"></a>
  <img alt="Licence Apache-2.0" src="https://img.shields.io/badge/licence-Apache--2.0-blue.svg">
</p>

## 1. Per-Head Calibration & Mondrian Conformal Safety Gates (`calibration.json`)

Raw neural network logits are often overconfident on rare classes (`course`, `comic`, `software`) and high-stakes safety boundaries (`adult`, `junk`). `calibration.json` + `ArmigerCalibrator` apply:
- **Per-Head Temperature Scaling ($T_h$)** and **Menon Long-Tail Logit Adjustments ($\tau_h \log \pi_h$)** across all 8 classification heads.
- **Mondrian Conformal Safety Thresholds:**
  - `adult`: `0.5451` (**`98.71%`** / **`98.63%`** Adult safety gate accuracy)
  - `junk`: `0.1010` (**`99.95%`** Junk / Malware balanced accuracy)
  - `owner_scope`: `0.4770` (**`98.68%`** policy scope accuracy)

## 2. Decentralized `BEP-44` + `Ed25519` DHT Update Engine (`ArmigerDHTUpdater`)

When `Armiger` or `Armiger Ultra` runs inside a distributed indexer (`Bitmagnet` and self-hosted media indexers), `ArmigerDHTUpdater` enables **opt-in decentralized hot-reloading** of new `zero-cue` catalog entries, `swarm-priors` tables, and rank-8 LoRA micro-adapters directly over the mainline BitTorrent DHT:
- **`BEP-44` Mutable DHT Pointers + `BEP-46` Infohash Distribution:** Nodes look up `SHA1(pubkey || "armiger-ultra-v2")` on the DHT to discover the latest signed bundle infohash.
- **Cryptographic Anti-Spoofing (`Ed25519` + Monotonic `seq`):** Every manifest must carry a valid `Ed25519` signature from the pinned offline root public key and a strictly increasing sequence number (`seq > current_seq`), preventing spoofed updates and downgrade attacks.
- **$K$-of-$N$ Quorum & Safe Formats Only:** Bundles contain only `JSON`, `ONNX`, or `SafeTensors` tensors (never arbitrary Python `pickle` code) and require a configurable $K$-of-$N$ attestation quorum before activation.

## 3. Distributed Recursive Self-Improvement (RSI) & Data Engine (`armiger_dht_rsi.py`)

For multi-agent indexers, private media stacks, and community clusters, `armiger_dht_rsi.py` provides a production-grade, zero-trust distributed training & calibration telemetry pipeline:
1. **Universal Adapters (Client-Agnostic across Media Stacks):** Native ingestion for series, movie, audio, and publication library webhooks, ffprobe / MediaInfo physical stream inspectors, and qBittorrent / Transmission completion events.
2. **Foolproof Physical Ground-Truth Verification:** Rejects raw unverified model predictions. Observations require 100% download verification, physical stream decodability, and canonical catalog matches (`series_catalog`, `movie_catalog`, `audio_catalog`, `isbn`). Active malware masquerades (`.exe` or `.scr` disguised as media) are physically isolated as junk canaries.
3. **Complete Anonymity & Differential Privacy:** Strips all private tracker passkeys, IPv4/IPv6 addresses, and user home directory paths. Generates salted HMAC-blinded syntactic title stems. Adds $(\epsilon, \delta)$-differential privacy (Laplace perturbation + 16MB log-binning) and Poisson timing jitter ($\Delta t \sim \text{Exp}(\lambda)$) against traffic analysis.
4. **Post-Quantum End-to-End Encryption:** Hybrid `X25519` ECDH + NIST FIPS 203 `ML-KEM-768` lattice key encapsulation (with Fujisaki-Okamoto CCA2 implicit rejection), combined via `SHAKE-256` / `HKDF-SHA3-256` with 256-bit `ChaCha20-Poly1305` AEAD payload encryption.
5. **Post-Quantum Spoof-Proof Protocol:** BEP-44 strictly monotonic sequence numbers (`seq > current_seq`) paired with Hybrid `Ed25519` + `WOTS+-SHAKE256` (RFC 8391 / NIST FIPS 205 hash-based PQ) threshold multi-signatures ($M$-of-$N$ federation quorum).
6. **Poison-Proof Sybil Quorum & Canary Gate:** Enforces $K$-of-$N$ distinct-ASN peer attestation quorum, strict format whitelisting (`SafeTensors`, `ONNX`, `JSON` only; zero pickle tolerance), and a mandatory local 500-item golden canary regression gate (rejects any update with malware bypass or Mondrian safety degradation).
7. **Opt-In / Opt-Out Modes:** Supports `off`, `pull_only`, and `anonymous_contribute`.

## Quickstart

```python
from calibration_and_dht import ArmigerCalibrator
from armiger_dht_rsi import ArmigerDHTRSIEngine, SeriesWebhookAdapter

# 1. Conformal calibration
cal = ArmigerCalibrator("calibration.json")
probs_cat = cal.calibrate_head("category", [2.4, -1.1, -3.0, -2.5, -3.1, -2.8, -3.2, -2.9, -3.0, -3.4, -2.0])
gates = cal.evaluate_safety_gates(p_adult=0.02, p_junk=0.004, p_excluded_scope=0.01)
print("Calibrated top prob:", max(probs_cat), "Safety gates:", gates)

# 2. Distributed RSI Engine
engine = ArmigerDHTRSIEngine(mode="anonymous_contribute")
obs = SeriesWebhookAdapter.parse_webhook({"eventType": "Download", "series": {"seriesCatalogId": 371980}})
ok, reason, record = engine.ingest_candidate(obs)
print("Ingested observation:", ok, reason)
```

## Part of the Armiger Modular Plugin Suite

| Plugin Repository | Bundled in `Armiger Ultra`? | Footprint | Core Capability & Measured Impact |
| :--- | :---: | :---: | :--- |
| **[`NorviTech/armiger-swarm-survival`](https://huggingface.co/NorviTech/armiger-swarm-survival)** | **Yes** (`swarm_adapter.pt`, `swarm_survival.py`) | `163 KB` | **SwarmSurvivalNet v2 & Honeypot Residual Adapter:** **`96.83%`** Swarm Liveliness AUROC (vs. `53.04%` raw seeders), **`100.0%`** fake-seeder malware trap recall, **`0%`** content-category flip guarantee. |
| **[`NorviTech/armiger-swarm-priors`](https://huggingface.co/NorviTech/armiger-swarm-priors)** | **Yes** (`swarm_priors.py`) | `29.4 KB` | **Empirical Bayes Swarm & Release-Group Priors:** 4-pathway tracker scrape prior engine, mandatory modality dropout (`p=0.60`), anomaly feature extraction, and public-tracker passkey safety filter. |
| **[`NorviTech/armiger-zero-cue-catalog`](https://huggingface.co/NorviTech/armiger-zero-cue-catalog)** | **Yes** (`zero_cue_plugin.py`, `validated_zero_cue_catalog.json`) | `36.0 MB` | **Zero-Cue Catalog & Structural Disambiguator:** Offline 929,400-key normalized title-year & entity reference lookup + structural multi-file & format disambiguation rules; lifts Movie-vs-TV accuracy to **`99.46%`** and 11-class accuracy to **`98.48%`** (`98.93%` Hard Edge-Cases). |
| **[`NorviTech/armiger-audio-guard`](https://huggingface.co/NorviTech/armiger-audio-guard)** | **Yes** (`audio_language_guard.py`) | `4.2 KB` | **Cross-Head Audio Language & Anime Dub Guard:** Couples `title_language` (`P(Foreign Title)`) with `lang` head to disambiguate `.en.` prepositions, `<Foreign>.ENG.Sub`, regional dubs, and `MULTi (FR EN)` tags; reaches **`98.72%`** Gold Standard and **`99.88%`** Hard Edge-Case audio accuracy. |
| **[`NorviTech/armiger-calibration-dht`](https://huggingface.co/NorviTech/armiger-calibration-dht)** | **Yes** (`calibration.json`, `calibration_and_dht.py`, `armiger_dht_rsi.py`) | `115 KB` | **Conformal Calibration, BEP-44 DHT Updater & Distributed RSI Engine:** Per-head temperature scaling ($T_h$), Mondrian safety gates (`99.95%` junk balanced accuracy), `Ed25519` threshold-signed `BEP-44`/`BEP-46` DHT hot-reload, universal media-webhook/qBittorrent/Transmission/ffprobe ground-truth RSI, `X25519`+`ChaCha20-Poly1305` AEAD, distinct-ASN quorum, and 500-canary regression gate (`36/36` tests). |
| **[`NorviTech/armiger-dht-db-sync`](https://huggingface.co/NorviTech/armiger-dht-db-sync)** | **Yes** (`csam_guard.py`, `armiger_dht_db_sync.py`) | `98 KB` | **Modular, CSAM-Protected, Encrypted BEP-44/BEP-46 DHT Database Sync:** Multi-stage CSAM & abuse pre-ingest blocker, `Ed25519`-signed blinded `SHA-256` tombstone purge engine, 10-category $\times$ 14-language $\times$ 7-quality sharded subscriptions with strict default-off `xxx` isolation, `X25519`+`ChaCha20-Poly1305` AEAD, binary Merkle proofs, and $K$-of-$N$ distinct-ASN Sybil quorum (`27/27` tests). |

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
