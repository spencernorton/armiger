<h1 align="center">Armiger Plugin #3 — Zero-Cue Catalog &amp; Structural Disambiguator (<code>armiger-zero-cue-catalog</code>)</h1>

<p align="center">
  <strong>Resolves bare-title ("zero-cue") torrents lacking season/episode syntax via an offline 931,142-entry <code>BLAKE2b-64</code> cryptographically hashed media catalog (<code>26.0 MB</code>) and deterministic structural rules, lifting 5-split (<code>N=10,262</code>) category accuracy from <code>93.00%</code> to <code>94.63%</code> and specialist zero-cue accuracy from <code>85.55%</code> to <code>91.00%</code> (<code>+5.45 pp</code>).</strong><br>
  Available standalone here and pre-bundled inside <a href="https://huggingface.co/NorviTech/armiger-ultra"><code>NorviTech/armiger-ultra</code></a>.
</p>

<p align="center">
  <a href="https://huggingface.co/NorviTech/armiger-ultra"><img alt="Bundled in Armiger Ultra" src="https://img.shields.io/badge/Bundled_in-Armiger_Ultra-FD8024.svg"></a>
  <a href="https://huggingface.co/NorviTech/armiger"><img alt="Compatible with Armiger" src="https://img.shields.io/badge/Compatible_with-Armiger_CPU-0284C7.svg"></a>
  <img alt="Licence Apache-2.0" src="https://img.shields.io/badge/licence-Apache--2.0-blue.svg">
</p>

## Why the Zero-Cue Catalog Plugin Exists

In real-world BitTorrent/DHT indexing, **2% to 4%** of movie, TV, and anime torrents contain **zero structural cues** in either their title or file tree:
- `Sintel (2010) [1080p] [WEB-DL].mkv` (a standalone film vs. a 1-file TV special)
- `[SubGroup] Bare Japanese/English Title [1080p].mkv` (a theatrical anime film vs. a TV OVA/series without an episode number)
- `Director Name - Movies 1-6 Collection (1979-1998) 1080p` (a multi-file movie pack whose 6 `.mkv` files look identical in file-count to a 6-episode TV mini-series)

Even a 70B+ parameter cloud LLM must guess when given only a bare title unless it can consult a media catalog. `NorviTech/armiger-zero-cue-catalog` pairs deterministic structural guards with an offline, O(1) hash-table lookup across **931,142 `BLAKE2b-64` (`person=b"ArmigZCC"`) cryptographically hashed keys** (`472,445` `(title, year)` pairs, `256,963` anime title stems, and `201,734` unambiguous long titles — zero plain-text media titles stored on disk, zero `info_hash` memorization).

## Measured Accuracy Lift Across All 5 Evaluation Cohorts (Unmutated Ground-Truth Keys)

| Evaluation Cohort (Unmutated `rows.y`) | Valid `N` | Standalone Neural (`FP32` / `INT8`) | **Fair Blind Hybrid (`+ armiger-zero-cue-catalog` FP32 / INT8)** | Net Lift |
| :--- | :---: | :---: | :---: | :---: |
| **`oct15_confirmation` (Gold Standard)** | `2,835` | `94.46%` / `94.43%` | **`94.85%`** / **`94.81%`** | `+0.39 pp` |
| **`fresh_confirmation` (Fresh Holdout)** | `1,402` | `92.30%` / `92.15%` | **`92.58%`** / **`92.44%`** | `+0.29 pp` |
| **`beyond_confirmation` (Hard Edge-Cases)** | `1,025` | `91.61%` / `91.61%` | **`93.07%`** / **`92.98%`** | **`+1.46 pp`** |
| **`zero_cue_2000` (Specialist Zero-Cue)** | `2,000` | `85.55%` / `85.55%` | **`91.00%`** / **`90.95%`** | **`+5.45 pp`** |
| **`multilang_3000` (Multilingual)** | `3,000` | `97.40%` / `97.27%` | **`98.33%`** / **`98.27%`** | **`+0.93 pp`** |
| **5-Split Weighted Mean (`N=10,262`)** | **`10,262`** | **`93.00%`** / **`92.94%`** | **`94.63%`** / **`94.56%`** | **`+1.63 pp`** |

## Quickstart

```python
from zero_cue_plugin import ZeroCueCatalogPlugin

plugin = ZeroCueCatalogPlugin("validated_zero_cue_catalog.json")
res = plugin.resolve(
    raw_name="Sintel (2010) [1080p] [WEB-DL]",
    probs_category=[0.46, 0.51, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.03],
    p_adult=0.01,
    p_anime=0.02,
)
print(res)
# -> {'category_index': 0, 'category': 'movie', 'resolution_source': 'zero_cue_title_year'}
```

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
