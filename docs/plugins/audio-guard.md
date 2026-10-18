<h1 align="center">Armiger Plugin #4 — Cross-Head Audio Language &amp; Anime Dub Guard (<code>armiger-audio-guard</code>)</h1>

<p align="center">
  <strong>Couples Armiger's neural <code>title_language</code> and <code>lang</code> heads with linguistic &amp; tracker-tag disambiguation to achieve <code>98.72%</code> Gold Standard and <code>99.88%</code> Hard Edge-Case audio language accuracy.</strong><br>
  Available standalone here and pre-bundled inside <a href="https://huggingface.co/NorviTech/armiger-ultra"><code>NorviTech/armiger-ultra</code></a>.
</p>

<p align="center">
  <a href="https://huggingface.co/NorviTech/armiger-ultra"><img alt="Bundled in Armiger Ultra" src="https://img.shields.io/badge/Bundled_in-Armiger_Ultra-FD8024.svg"></a>
  <a href="https://huggingface.co/NorviTech/armiger"><img alt="Compatible with Armiger" src="https://img.shields.io/badge/Compatible_with-Armiger_CPU-0284C7.svg"></a>
  <img alt="Licence Apache-2.0" src="https://img.shields.io/badge/licence-Apache--2.0-blue.svg">
</p>

## What `armiger-audio-guard` Solves

Standard regex parsers (`RTN`, `Parsett`, `GuessIt`, `PTN`) and single-head classifiers systematically fail on five recurring multilingual torrent patterns:
1. **Romance-Language `.en.` Prepositions:** Titles like `la.casa.en.llamas.2024.1080p.mp4` or `En.attendant.la.nuit.2024.mkv` contain the Spanish/French word *en* ("in"). Case-insensitive `.en.` tokenizers misclassify these as English audio.
2. **`<Foreign>.ENG.Sub` Releases:** In `Movie.Title.ITA.ENG.Sub.2025.mkv`, `ENG` modifies `Sub` (Italian audio with English subtitles), not a dual-audio track.
3. **Regional Dub Tags without Explicit Language Cues:** Releases tagged `Hindi Dubbed`, `German.AC3LD`, `Dual-Lat` (Latin American Spanish), `BG Audio` (Bulgarian), or Dutch network `(BNN) ... NL`.
4. **Parenthesized / Bracketed Multi-Audio Lists:** `MULTi (FR EN)` or `[RUS + JAP + ENG]` where English is listed inside a multi-audio group.
5. **Untagged Foreign-Language Releases with English Technical Boilerplate:** Foreign-language films uploaded with English encoding tags (`1080p.BluRay.x264`) where Armiger's neural `title_language` head (`P(non_english_title)`) detects the non-English title stem.

## Measured Accuracy

| Benchmark Cohort | `GuessIt` | `RTN` / `Parsett` | Neural `lang` Head Alone | **With `armiger-audio-guard` (`Armiger`)** | **With `armiger-audio-guard` (`Armiger Ultra`)** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Gold Standard Audio Language (`N=545`)** | `96.01%` | `95.33%` | `96.88%` | **`98.17%`** | **`98.72%`** |
| **Hard Edge-Cases Audio Language (`N=861`)** | `96.01%` | `95.33%` | `99.65%` | **`99.88%`** | **`99.88%`** |
| **Anime & Anime Dub Detection (`N=2,835`)** | `75.90%` | `77.65%` | `97.60%` | **`97.82%`** | **`97.92%`** |

## Quickstart

```python
from audio_language_guard import apply_audio_language_guard, classify_audio_and_anime_dub

# Example 1: Spanish preposition '.en.' with high neural foreign-title probability
p_en = apply_audio_language_guard(
    raw_name="la.casa.en.llamas.2024.1080p-dual-lat.mp4",
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
