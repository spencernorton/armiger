<h1 align="center">Armiger Plugin #1 — Swarm Survival &amp; Honeypot Adapter (<code>SwarmSurvivalNet v2</code>)</h1>

<p align="center">
  <strong>Content-gated residual adapter that predicts real-world torrent download completion (<code>96.83%</code> AUROC) and blocks <code>100.0%</code> of fake-seeder malware honeypots while guaranteeing <code>0%</code> content-category flips.</strong><br>
  Available standalone here and pre-bundled inside <a href="https://huggingface.co/NorviTech/armiger-ultra"><code>NorviTech/armiger-ultra</code></a>.
</p>

<p align="center">
  <a href="https://huggingface.co/NorviTech/armiger-ultra"><img alt="Bundled in Armiger Ultra" src="https://img.shields.io/badge/Bundled_in-Armiger_Ultra-FD8024.svg"></a>
  <a href="https://huggingface.co/NorviTech/armiger"><img alt="Compatible with Armiger" src="https://img.shields.io/badge/Compatible_with-Armiger_CPU-0284C7.svg"></a>
  <img alt="Licence Apache-2.0" src="https://img.shields.io/badge/licence-Apache--2.0-blue.svg">
</p>

## Overview

Raw tracker seeder counts are a poor predictor of whether a torrent grab will actually complete (`53.04%` AUROC across mixed public/DHT swarms):
1. **Fake-Seeder Malware Honeypots** spoof `5,000+` seeders on day zero for unreleased movies or games while bundling `.exe` / `.lnk` / `.scr` / codec-pack payloads.
2. **Stale Tracker Scrapes vs. DHT/PEX Survival:** Niche anime, audiobooks, and remuxes often report `0` cached tracker seeders yet complete reliably via DHT/PEX, whereas poisoned swarms stall at `99.9%`.

`NorviTech/armiger-swarm-survival` (`swarm_adapter.pt` + `swarm_survival.py`) attaches to [`NorviTech/armiger`](https://huggingface.co/NorviTech/armiger) or [`NorviTech/armiger-ultra`](https://huggingface.co/NorviTech/armiger-ultra) and fuses live swarm telemetry with the backbone's 408-D content representation.

## Measured Performance

| Metric | Raw Tracker Seeders | Raw DHT Peers | Armiger (`Way-4` Formula) | **Armiger Ultra + `armiger-swarm-survival`** |
| :--- | :---: | :---: | :---: | :---: |
| **Swarm Liveliness AUROC (Mixed Corpus, `N=25,280`)** | `53.04%` | `51.84%` | `86.64%` | **`97.35%`** |
| **Swarm Liveliness AUROC (Bitmagnet DHT Scraped, `N=10,278`)** | `85.17%` | `86.81%` | `86.64%` | **`96.83%`** |
| **Fake-Seeder Honeypot Trap Recall (`@ <=0.5% FPR`)** | `0.00%` | `0.00%` | `99.90%` | **`100.00%`** |
| **Content Category Flip Rate (`seeders=0` vs `seeders=10,000`)** | N/A | N/A | **`0.00%`** | **`0.00%`** |

## Quickstart

```python
from swarm_survival import SwarmResidualAdapter

adapter = SwarmResidualAdapter.from_pretrained("swarm_adapter.pt")
result = adapter.evaluate_swarm(
    title="Tears.of.Steel.2012.2160p.UHD.BluRay.x265-OPEN",
    seeders=42,
    leechers=5,
    completed=310,
    total_size_bytes=28_500_000_000,
    files=[{"path": "Tears.of.Steel.2012.2160p.mkv", "size": 28_500_000_000}],
    sources=["bt1.archive.org"],
    category_pred="movie",
    p_junk_neural=0.002,
)
print(result)
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
