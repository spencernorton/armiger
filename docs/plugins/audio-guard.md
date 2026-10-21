<h1 align="center">Armiger Plugin #4 — Cross-Head Audio Language &amp; Anime Dub Guard (<code>armiger-audio-guard</code>)</h1>

<p align="center">
  <strong>Couples Armiger's neural <code>title_language</code> and <code>lang</code> heads with linguistic &amp; tracker-tag disambiguation to lift binary English-vs-Foreign audio language accuracy on <code>oct15_confirmation</code> (<code>N=545</code>) from <code>96.88%</code> to <code>97.80%</code> (<code>92.75%</code> 3-class, <code>N=772</code>).</strong><br>
  Available standalone here and pre-bundled inside <a href="https://huggingface.co/NorviTech/armiger-ultra"><code>NorviTech/armiger-ultra</code></a>.
</p>

<p align="center">
  <a href="https://huggingface.co/NorviTech/armiger-ultra"><img alt="Bundled in Armiger Ultra" src="https://img.shields.io/badge/Bundled_in-Armiger_Ultra-FD8024.svg"></a>
  <a href="https://huggingface.co/NorviTech/armiger"><img alt="Compatible with Armiger" src="https://img.shields.io/badge/Compatible_with-Armiger_CPU-0284C7.svg"></a>
  <img alt="Licence Apache-2.0" src="https://img.shields.io/badge/licence-Apache--2.0-blue.svg">
</p>

## What `armiger-audio-guard` Solves

Standard regex parsers (`RTN`, `Parsett`, `GuessIt`, `PTN`) and single-head classifiers systematically fail on five recurring multilingual torrent patterns:
1. **Romance-Language `.en.` Prepositions:** Titles like `Le.Voyage.dans.la.Lune.1902.1080p.mp4` or `Les.Quatre.Cents.Farces.du.Diable.1906.mkv` contain the Spanish/French word *en* ("in"). Case-insensitive `.en.` tokenizers misclassify these as English audio.
2. **`<Foreign>.ENG.Sub` Releases:** In `Movie.Title.ITA.ENG.Sub.2025.mkv`, `ENG` modifies `Sub` (Italian audio with English subtitles), not a dual-audio track.
3. **Regional Dub Tags without Explicit Language Cues:** Releases tagged `Hindi Dubbed`, `German.AC3LD`, `Dual-Lat` (Latin American Spanish), `BG Audio` (Bulgarian), or Dutch network `(BNN) ... NL`.
4. **Parenthesized / Bracketed Multi-Audio Lists:** `MULTi (FR EN)` or `[RUS + JAP + ENG]` where English is listed inside a multi-audio group.
5. **Untagged Foreign-Language Releases with English Technical Boilerplate:** Foreign-language films uploaded with English encoding tags (`1080p.WEBM.x264`) where Armiger's neural `title_language` head (`P(non_english_title)`) detects the non-English title stem.

## Measured Accuracy (Unmutated Ground-Truth Keys)

| Benchmark Cohort (Unmutated `rows.y`) | `GuessIt` | `RTN` / `Parsett` | Standalone Neural `lang` Head | **With `armiger-audio-guard` (`Armiger` / `Ultra`)** |
| :--- | :---: | :---: | :---: | :---: |
| **`oct15_confirmation` Binary EN-vs-Foreign (`N=545`)** | `96.01%` | `95.33%` | `96.88%` | **`97.80%`** |
| **`oct15_confirmation` 3-Class Audio (`N=772`)** | — | — | `68.39%` (raw prior) | **`92.75%`** (calibrated + guard) |
| **`multilang_3000` Audio (`N=3,000`)** | — | — | `91.47%` | **`91.47%`** |
| **`oct15_confirmation` Anime Detection (`N=507`)** | `75.90%` | `77.65%` | `99.61%` | **`99.61%`** (`505/507`) |
| **`zero_cue_2000` Anime Detection (`N=1,998`)** | `75.90%` | `77.65%` | `95.80%` | **`95.80%`** (`1,914/1,998`) |

## Quickstart

```python
from audio_language_guard import apply_audio_language_guard, classify_audio_and_anime_dub

# Example 1: Spanish preposition '.en.' with high neural foreign-title probability
p_en = apply_audio_language_guard(
    raw_name="Le.Voyage.dans.la.Lune.1902.1080p-dual-lat.mp4",
    p_en=0.62,
    p_title_foreign=0.96,
)
print("Adjusted P(English audio):", p_en)  # -> 0.10 (Foreign audio)

# Example 2: Multi-audio release with explicit (FR EN)
print(classify_audio_and_anime_dub(
    raw_name="Show.Name.S01.MULTi (FR EN) [1080p]",
    p_en=0.38,
    p_anime=0.02,
    p_title_foreign=0.15,
))
```

## Part of the Armiger Modular Plugin Suite

| Plugin Repository | Bundled in `Armiger Ultra`? | Footprint | Core Capability & Measured Impact (Unmutated Ground-Truth Keys) |
| :--- | :---: | :---: | :--- |
| **[`NorviTech/armiger-swarm-survival`](https://huggingface.co/NorviTech/armiger-swarm-survival)** | **Yes** (`swarm_adapter.pt`, `swarm_survival.py`) | `167 KB` (`132 KB` `.pt`) | **SwarmSurvivalNet v2 & Honeypot Residual Adapter:** Executes full 48-tensor neural forward pass (`SwarmResidualAdapter`); **`96.83%`** Multi-Source Corpus AUROC (`N=10,278`, vs. `53.02%` raw seeders), **`92.42%`** / **`93.03%`** Public-Tracker AUROC (`N=757` / `N=493`, vs. `85.17%` raw seeders), disguised-executable extension filtering, seeder-independent content categorization. |
| **[`NorviTech/armiger-swarm-priors`](https://huggingface.co/NorviTech/armiger-swarm-priors)** | **Yes** (`swarm_priors.py`) | `29.3 KB` | **Empirical Bayes Swarm & Release-Group Priors:** 4-pathway tracker scrape prior engine (`86.64%` formula AUROC), mandatory modality dropout (`p=0.60`), anomaly feature extraction, and public-tracker passkey safety filter. |
| **[`NorviTech/armiger-zero-cue-catalog`](https://huggingface.co/NorviTech/armiger-zero-cue-catalog)** | **Yes** (`zero_cue_plugin.py`, `validated_zero_cue_catalog.json`) | `26.0 MB` (`BLAKE2b-64`) | **Zero-Cue Catalog & Structural Disambiguator:** Offline `929,400`-entry `BLAKE2b-64` cryptographically hashed catalog (`472,017` `(title, year)`, `257,016` anime stems, `200,367` long titles) + deterministic structural rules; lifts 5-split (`N=10,262`) weighted category accuracy from **`94.00%`** (Standalone Neural) to **`95.42%`** (Fair Blind Hybrid), `beyond_confirmation` (`N=1,025`) from **`92.88%`** to **`95.12%`**, and `zero_cue_2000` (`N=2,000`) from **`87.40%`** to **`91.50%`** (`+4.10 pp`). |
| **[`NorviTech/armiger-audio-guard`](https://huggingface.co/NorviTech/armiger-audio-guard)** | **Yes** (`audio_language_guard.py`) | `3.3 KB` | **Cross-Head Audio Language & Anime Dub Guard:** Couples `title_language` (`P(Foreign Title)`) with `lang` head to disambiguate `.en.` prepositions, `<Foreign>.ENG.Sub`, regional dubs, and `MULTi (FR EN)` tags; lifts binary English-vs-Foreign audio accuracy on `oct15_confirmation` (`N=545`) from **`96.88%`** (Standalone Neural) to **`97.80%`** (`92.75%` 3-class `N=772`). |
| **[`NorviTech/armiger-calibration-dht`](https://huggingface.co/NorviTech/armiger-calibration-dht)** | **Yes** (`calibration.json`, `calibration_and_dht.py`, `armiger_dht_rsi.py`) | `181 KB` | **Conformal Calibration, BEP-44 DHT Updater & Distributed RSI Engine:** Per-head temperature scaling ($T_h$), Mondrian safety gates (**`99.23%`** balanced junk/sample accuracy / **`99.90%`** overall accuracy on `beyond_confirmation` `N=977`, `65` positive items), Ed25519-signed `BEP-44`/`BEP-46` DHT hot-reload, `X25519`+`ChaCha20-Poly1305` AEAD, and 500-canary regression gate (`47/47` tests). |
| **[`NorviTech/armiger-dht-db-sync`](https://huggingface.co/NorviTech/armiger-dht-db-sync)** | **Yes** (`csam_guard.py`, `armiger_dht_db_sync.py`) | `155 KB` | **Modular, CSAM-Protected, Encrypted BEP-44/BEP-46 DHT Database Sync:** Multi-stage CSAM & abuse pre-ingest blocker, Ed25519-signed blinded `SHA-256` tombstone purge engine, sharded subscriptions with default-off `xxx` isolation, and `X25519`+`ChaCha20-Poly1305` AEAD (`27/27` tests). |

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
