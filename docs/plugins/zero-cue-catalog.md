<h1 align="center">Armiger Plugin #3 — Zero-Cue Catalog &amp; Structural Disambiguator (<code>armiger-zero-cue-catalog</code>)</h1>

<p align="center">
  <strong>Resolves bare-title ("zero-cue") torrents lacking season/episode syntax via an offline 929,400-key normalized media catalog and structural release rules, lifting Movie-vs-TV accuracy to <code>99.46%</code> and 11-class accuracy to <code>98.48%</code>.</strong><br>
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

Even a 70B+ parameter cloud baseline classifiers must guess when given only a bare title unless it can consult a media catalog. `NorviTech/armiger-zero-cue-catalog` pairs deterministic structural guards (for multi-item media collections, interactive software archives, and structured educational materials) with an offline, O(1) hash-table lookup across **929,400 normalized keys** (`472,017` `(title, year)` pairs, `257,016` anime title stems, and `200,367` unambiguous long titles).

## Measured Accuracy Lift Across All 6 Evaluation Cohorts

| Evaluation Cohort | Neural Backbone Only (`Armiger Ultra`) | + Structural Guards | **+ `armiger-zero-cue-catalog` (`Armiger Ultra`)** | **+ `armiger-zero-cue-catalog` (`Armiger 8.3 MB`)** |
| :--- | :---: | :---: | :---: | :---: |
| **Gold Standard (`N=2,835`)** | `97.45%` | `97.81%` | **`98.48%`** (`99.46%` Movie-vs-TV) | **`97.92%`** (`99.15%` Movie-vs-TV) |
| **Fresh Torrents (`N=1,354`)** | `95.79%` | `95.94%` | **`96.60%`** | **`96.38%`** |
| **Hard Edge-Cases (`N=1,025`)** | `95.71%` | `96.10%` | **`97.17%`** | **`96.20%`** |
| **Validation Holdout (`N=2,237`)** | `96.92%` | `97.05%` | **`97.32%`** | **`96.51%`** |
| **Specialist Torrents (`N=1,837`)** | `93.85%` | `94.18%` | **`98.26%`** | **`97.82%`** |
| **Multilingual Torrents (`N=2,828`)** | `95.44%` | `95.69%` | **`98.02%`** | **`97.81%`** |

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
