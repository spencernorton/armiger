<h1 align="center">Armiger Plugin #5 — Conformal Calibration &amp; BEP-44 Decentralized DHT Updater (<code>armiger-calibration-dht</code>)</h1>

<p align="center"><strong>Release:</strong> Armiger <code>v3.0.0</code> (<code>32.5 MB</code> FP32 + <code>15.7 MB</code> INT8) · Apache-2.0</p>

<p align="center">
  <strong>Per-head temperature scaling, Menon prior adjustments, Mondrian conformal safety gates (<code>99.23%</code> balanced junk/sample accuracy / <code>99.90%</code> overall accuracy on <code>beyond_confirmation</code> <code>N=977</code>), and an optional <code>Ed25519</code>-signed <code>BEP-44</code> / <code>BEP-46</code> BitTorrent DHT calibration update engine.</strong><br>
  Available standalone here and pre-bundled inside <a href="https://huggingface.co/NorviTech/armiger-ultra"><code>NorviTech/armiger-ultra</code></a>.
</p>

<p align="center">
  <a href="https://huggingface.co/NorviTech/armiger-ultra"><img alt="Bundled in Armiger Ultra" src="https://img.shields.io/badge/Bundled_in-Armiger_Ultra-FD8024.svg"></a>
  <a href="https://huggingface.co/NorviTech/armiger"><img alt="Compatible with Armiger" src="https://img.shields.io/badge/Compatible_with-Armiger_CPU-0284C7.svg"></a>
  <img alt="Licence Apache-2.0" src="https://img.shields.io/badge/licence-Apache--2.0-blue.svg">
</p>

## 1. Per-Head Calibration & Mondrian Conformal Safety Gates (`calibration.json`)

Raw neural network logits are often overconfident on rare classes (`course`, `comic`, `software`) and high-stakes safety boundaries (`adult`, `junk`). `calibration.json` + `ArmigerCalibrator` apply:
- **Guarded Per-Head Temperature Scaling ($T_h$)** and **Menon Long-Tail Logit Adjustments ($\tau_h \log \pi_h$)** across the 8 calibrated classification heads (the 9th `span_bio` head is CRF/Viterbi-decoded and is not temperature-scaled). A fitted temperature ships only when it improves validation NLL over `T=1`.
- **Mondrian Conformal Safety Thresholds (Unmutated Ground-Truth Keys):** v3.0.0 retains the inherited production gates below; calibration-split FPR-fitted alternates are recorded in `calibration.json` (`mondrian_fitted_calibration_split`) but are not shipped as gates because they cost adult/owner recall on confirmation splits.
  - `adult`: `0.5451` (**`98.49%`** on `oct15_confirmation` `N=2,781`, **`98.43%`** on `beyond_confirmation` `N=1,016`; `zero_cue_2000` / `multilang_3000` have no labeled adult positives)
  - `junk`: `0.1010` (**`99.95%`** overall accuracy on `oct15_confirmation` `N=2,170` with 1 positive, **`100.00%`** on `fresh_confirmation` `N=1,291` with 3 positives, **`99.90%`** overall / **`99.23%`** balanced on `beyond_confirmation` `976/977` with 65 positives, plus deterministic disguised-executable extension blocking)
  - `owner_scope`: `0.4770` (**`98.31%`** on `oct15_confirmation` `2,624/2,669`, **`98.25%`** on `beyond_confirmation`, **`97.46%`** on `fresh_confirmation`; temperature-scaled neural head)

## 2. Decentralized `BEP-44` + `Ed25519` DHT Update Engine (`ArmigerDHTUpdater`)

When `Armiger` or `Armiger Ultra` runs inside a distributed indexer (`Bitmagnet` and self-hosted media indexers), `ArmigerDHTUpdater` enables **opt-in decentralized hot-reloading** of new `zero-cue` catalog entries, `swarm-priors` tables, and rank-8 LoRA micro-adapters directly over the mainline BitTorrent DHT:
- **`BEP-44` Mutable DHT Pointers + `BEP-46` Infohash Distribution:** Nodes look up `SHA1(pubkey || "armiger-ultra-v2")` on the DHT to discover the latest signed bundle infohash.
- **Cryptographic Anti-Spoofing (`Ed25519` + Monotonic `seq`):** Every manifest must carry a valid `Ed25519` signature from the pinned offline root public key and a strictly increasing sequence number (`seq > current_seq`), preventing spoofed updates and downgrade attacks.
- **$K$-of-$N$ Quorum & Safe Formats Only:** Bundles contain only `JSON`, `ONNX`, or `SafeTensors` tensors (never arbitrary Python `pickle` code) and require a configurable $K$-of-$N$ attestation quorum before activation.

## 3. Distributed Recursive Self-Improvement (RSI) & Data Engine (`armiger_dht_rsi.py`)

For multi-agent indexers, private media stacks, and community clusters, `armiger_dht_rsi.py` provides a production-grade, zero-trust distributed training & calibration telemetry pipeline:
1. **Universal Adapters (Client-Agnostic across Media Stacks):** Native ingestion for series, movie, audio, and publication library webhooks, ffprobe / MediaInfo physical stream inspectors, and qBittorrent / Transmission completion events.
2. **Multi-Stage Physical Ground-Truth Verification:** Rejects raw unverified model predictions. Observations require 100% download verification, physical stream decodability, and canonical catalog matches (`series_catalog`, `movie_catalog`, `audio_catalog`, `isbn`). Active malware masquerades (`.exe` or `.scr` disguised as media) are physically isolated as junk canaries.
3. **Cryptographic Anonymization & Differential Privacy:** Strips all private tracker passkeys, IPv4/IPv6 addresses, and user home directory paths. Generates salted HMAC-blinded syntactic title stems. Adds $(\epsilon, \delta)$-differential privacy (Laplace perturbation + 16MB log-binning) and Poisson timing jitter ($\Delta t \sim \text{Exp}(\lambda)$) against traffic analysis.
4. **Authenticated Payload Encryption:** `X25519` ECDH key agreement combined with 256-bit `ChaCha20-Poly1305` AEAD payload encryption.
5. **Signed Anti-Rollback Manifests:** `BEP-44` strictly monotonic sequence numbers (`seq > current_seq`) paired with `Ed25519` threshold signatures.
6. **Sybil-Resistant ASN Quorum & Canary Gate:** Enforces $K$-of-$N$ distinct-ASN peer attestation quorum, strict format whitelisting (`SafeTensors`, `ONNX`, `JSON` only; zero pickle tolerance), and a mandatory local 500-item golden canary regression gate (rejects any update with malware bypass or Mondrian safety degradation).
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

| Plugin Repository | Bundled in `Armiger Ultra`? | Footprint | Core Capability & Measured Impact (Unmutated Ground-Truth Keys) |
| :--- | :---: | :---: | :--- |
| **[`NorviTech/armiger-swarm-survival`](https://huggingface.co/NorviTech/armiger-swarm-survival)** | **Yes** (`swarm_adapter.pt`, `swarm_survival.py`) | `149 KB` (`132 KB` `.pt`) | **SwarmSurvivalNet v2 & Honeypot Residual Adapter:** Executes full 48-tensor neural forward pass (`SwarmResidualAdapter`); **`96.83%`** Multi-Source Corpus AUROC (`N=10,278`, vs. `53.02%` raw seeders), **`92.42%`** Public-Tracker AUROC (`N=757`, vs. `85.17%` raw seeders), disguised-executable extension filtering, seeder-independent content categorization. |
| **[`NorviTech/armiger-swarm-priors`](https://huggingface.co/NorviTech/armiger-swarm-priors)** | **Yes** (`swarm_priors.py`) | `37 KB` | **Empirical Bayes Swarm & Release-Group Priors:** 4-pathway tracker scrape prior engine (`86.64%` formula AUROC), mandatory modality dropout (`p=0.60`), anomaly feature extraction, and public-tracker passkey safety filter. |
| **[`NorviTech/armiger-zero-cue-catalog`](https://huggingface.co/NorviTech/armiger-zero-cue-catalog)** | **Yes** (`zero_cue_plugin.py`, `validated_zero_cue_catalog.json`) | `26.0 MB` (`BLAKE2b-64`) | **Zero-Cue Catalog & Structural Disambiguator:** Offline `931,142`-entry `BLAKE2b-64` cryptographically hashed catalog (`472,445` `(title, year)`, `256,963` anime stems, `201,734` long titles) + deterministic structural rules; lifts 5-split (`N=10,262`) weighted category accuracy from **`93.00%`** (Standalone Neural) to **`94.63%`** (Fair Blind Hybrid), `beyond_confirmation` (`N=1,025`) from **`91.61%`** to **`93.07%`**, and `zero_cue_2000` (`N=2,000`) from **`85.55%`** to **`91.00%`** (`+5.45 pp`). |
| **[`NorviTech/armiger-audio-guard`](https://huggingface.co/NorviTech/armiger-audio-guard)** | **Yes** (`audio_language_guard.py`) | `32 KB` | **Cross-Head Audio Language & Anime Dub Guard:** Couples `title_language` (`P(Foreign Title)`) with `lang` head to disambiguate `.en.` prepositions, `<Foreign>.ENG.Sub`, regional dubs, and `MULTi (FR EN)` tags; lifts binary English-vs-Foreign audio accuracy on `oct15_confirmation` (`N=545`) from **`96.33%`** (Standalone INT8 Neural) to **`96.70%`**. |
| **[`NorviTech/armiger-calibration-dht`](https://huggingface.co/NorviTech/armiger-calibration-dht)** | **Yes** (`calibration.json`, `calibration_and_dht.py`, `armiger_dht_rsi.py`) | `81 KB` | **Conformal Calibration, BEP-44 DHT Updater & Distributed RSI Engine:** Per-head temperature scaling ($T_h$), Mondrian safety gates (**`99.23%`** balanced junk/sample accuracy / **`99.90%`** overall accuracy on `beyond_confirmation` `N=977`, `65` positive items), Ed25519-signed `BEP-44`/`BEP-46` DHT hot-reload, `X25519`+`ChaCha20-Poly1305` AEAD, and 500-canary regression gate (repository BEP/DHT test battery). |
| **[`NorviTech/armiger-dht-db-sync`](https://huggingface.co/NorviTech/armiger-dht-db-sync)** | **Yes** (`csam_guard.py`, `armiger_dht_db_sync.py`) | `209 KB` | **Modular, CSAM-Protected, Encrypted BEP-44/BEP-46 DHT Database Sync:** Multi-stage CSAM & abuse pre-ingest blocker, Ed25519-signed blinded `SHA-256` tombstone purge engine, sharded subscriptions with default-off `xxx` isolation, and `X25519`+`ChaCha20-Poly1305` AEAD (repository BEP/DHT test battery). |

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
