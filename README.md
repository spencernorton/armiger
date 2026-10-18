<h1 align="center">Armiger Ultra</h1>

<p align="center">
  <strong>Unified torrent intelligence suite bundling Armiger (8.3 MB INT8 CPU), Armiger Ultra (32.3 MB FP32 GPU/CPU), the 48-tensor Swarm Health &amp; Honeypot Adapter (<code>swarm_adapter.pt</code>), and all 6 official plugins.</strong><br>
  Delivers 97.55% 6-dimension composite accuracy on unmutated ground-truth keys, 0.21 ms/item batched on NVIDIA RTX GPU (0.29 ms on Apple M4 Pro MPS GPU, 2.44 ms single-item on M4 Pro CPU INT8), 100% fake-seeder malware trap recall, and 96.83% warehouse / 92.42% public-tracker swarm liveliness AUROC.
</p>

<p align="center">
  <a href="https://norvitech.com"><img alt="NorviTech Suite" src="https://img.shields.io/badge/NorviTech-Suite-FD8024.svg"></a>
  <a href="https://huggingface.co/NorviTech/armiger"><img alt="Armiger CPU" src="https://img.shields.io/badge/Armiger-8.3_MB_CPU-0284C7.svg"></a>
  <img alt="Licence Apache-2.0" src="https://img.shields.io/badge/licence-Apache--2.0-blue.svg">
</p>

**Armiger Ultra** (`NorviTech/armiger-ultra`) is the complete, all-in-one release of the Armiger neural classification and swarm-intelligence stack for `Bitmagnet` and self-hosted media library stacks.

**One Download, Everything Included:** You do **not** need to download [`NorviTech/armiger`](https://huggingface.co/NorviTech/armiger) and `Armiger Ultra` separately. This repository bundles:
1. **`armiger_ultra.fp32.onnx` (`32.3 MB`) & `armiger_ultra.pt` (`32.2 MB`):** Full-precision GPU/CPU 8-head neural classifier from the hardened `H350000_Curated_Wave2-seed0` checkpoint (`0.21 ms/item` batch-64 on NVIDIA RTX GPU, `0.29 ms/item` batch-64 on Apple M4 Pro MPS GPU).
2. **`armiger.int8.onnx` (`8.3 MB`):** Dynamically quantized CPU fast-path engine from the identical checkpoint (`2.44 ms` single-item p50 / `3.07 ms/item` batch-64 on Apple M4 Pro CPU ONNX Runtime).
3. **`armiger_inference.py` + `vocab.json`:** Self-contained, zero-tokenizer 11-input featurizer (`title_bytes`, `ext_ids`, `file_sizes_log`, `file_depths`, `file_flags`, `file_mask`, `struct_feats`, `tracker_ids`, `cat_hint_id`, `tag_ids`, `swarm_feats`) for direct ONNX Runtime inference.
4. **`swarm_adapter.pt` (`132 KB`) & `swarm_survival.py`:** **48-Tensor PyTorch Swarm Health, Download Viability & Fake-Seeder Honeypot Residual Adapter (`SwarmResidualAdapter`)**, fusing tracker/DHT seeder/leecher telemetry with content features to achieve **`96.83%` Multi-Source Warehouse AUROC** (`N=10,278`, vs. `53.02%` raw seeders), **`92.42%` / `93.03%` Public-Tracker AUROC** (`N=757` / `N=493`, vs. `85.17%` raw seeders), **`100.0%` fake-seeder honeypot recall**, and **`0.00%` category flip rate**.
5. **`validated_zero_cue_catalog.json` (`26.0 MB`) & `zero_cue_plugin.py`:** Privacy-hardened `BLAKE2b-64` (`person=b"ArmigZCC"`) 929,400-entry offline title/year/anime catalog + deterministic structural rules.
6. **`calibration.json`:** Post-hoc temperature scaling, Menon prior logit adjustments, and Mondrian conformal safety thresholds (`100.00%` junk accuracy on `oct15_confirmation` `N=2,170`).

---

## Modular 6-Plugin Architecture (All 6 Rolled Into `Armiger Ultra` + Available Standalone)

All **6 Armiger plugins** are rolled directly into this `NorviTech/armiger-ultra` repository for one-command deployment, and are also published as standalone modular repositories under [`NorviTech`](https://huggingface.co/NorviTech) for custom pipelines:

| Plugin Repository | Bundled in `Armiger Ultra`? | Footprint | Core Capability & Measured Impact (Unmutated Ground-Truth Keys) |
| :--- | :---: | :---: | :--- |
| **[`NorviTech/armiger-swarm-survival`](https://huggingface.co/NorviTech/armiger-swarm-survival)** | **Yes** (`swarm_adapter.pt`, `swarm_survival.py`) | `167 KB` (`132 KB` `.pt`) | **SwarmSurvivalNet v2 & Honeypot Residual Adapter:** Executes full 48-tensor neural forward pass (`SwarmResidualAdapter`); **`96.83%`** Multi-Source Warehouse AUROC (`N=10,278`, vs. `53.02%` raw seeders), **`92.42%`** / **`93.03%`** Public-Tracker AUROC (`N=757` / `N=493`, vs. `85.17%` raw seeders), **`100.0%`** fake-seeder malware trap recall, **`0.00%`** content-category flip guarantee. |
| **[`NorviTech/armiger-swarm-priors`](https://huggingface.co/NorviTech/armiger-swarm-priors)** | **Yes** (`swarm_priors.py`) | `29.3 KB` | **Empirical Bayes Swarm & Release-Group Priors:** 4-pathway tracker scrape prior engine (`86.64%` formula AUROC), mandatory modality dropout (`p=0.60`), anomaly feature extraction, and public-tracker passkey safety filter. |
| **[`NorviTech/armiger-zero-cue-catalog`](https://huggingface.co/NorviTech/armiger-zero-cue-catalog)** | **Yes** (`zero_cue_plugin.py`, `validated_zero_cue_catalog.json`) | `26.0 MB` (`BLAKE2b-64`) | **Zero-Cue Catalog & Structural Disambiguator:** Offline `929,400`-entry `BLAKE2b-64` cryptographically hashed catalog (`472,017` `(title, year)`, `257,016` anime stems, `200,367` long titles) + deterministic structural rules; lifts 5-split (`N=10,262`) weighted category accuracy from **`94.00%`** (Standalone Neural) to **`95.42%`** (Fair Blind Hybrid), `beyond_confirmation` (`N=1,025`) from **`92.88%`** to **`95.12%`**, and `tc_truespec_2000` (`N=2,000`) from **`87.40%`** to **`91.50%`** (`+4.10 pp`). |
| **[`NorviTech/armiger-audio-guard`](https://huggingface.co/NorviTech/armiger-audio-guard)** | **Yes** (`audio_language_guard.py`) | `3.3 KB` | **Cross-Head Audio Language & Anime Dub Guard:** Couples `title_language` (`P(Foreign Title)`) with `lang` head to disambiguate `.en.` prepositions, `<Foreign>.ENG.Sub`, regional dubs, and `MULTi (FR EN)` tags; lifts binary English-vs-Foreign audio accuracy on `oct15_confirmation` (`N=545`) from **`96.88%`** (Standalone Neural) to **`97.80%`** (`92.75%` 3-class `N=772`). |
| **[`NorviTech/armiger-calibration-dht`](https://huggingface.co/NorviTech/armiger-calibration-dht)** | **Yes** (`calibration.json`, `calibration_and_dht.py`, `armiger_dht_rsi.py`) | `181 KB` | **Conformal Calibration, BEP-44 DHT Updater & Distributed RSI Engine:** Per-head temperature scaling ($T_h$), Mondrian safety gates (**`100.00%`** junk accuracy on `oct15_confirmation` `N=2,170` & `fresh_confirmation` `N=1,291`, **`99.90%`** on `beyond_confirmation` `N=977`), Hybrid `Ed25519`+`WOTS+-SHAKE256` signed `BEP-44`/`BEP-46` DHT hot-reload, Hybrid `X25519`+`ML-KEM-768`+`ChaCha20-Poly1305` AEAD, distinct-ASN quorum, and 500-canary regression gate (`47/47` tests). |
| **[`NorviTech/armiger-dht-db-sync`](https://huggingface.co/NorviTech/armiger-dht-db-sync)** | **Yes** (`csam_guard.py`, `armiger_dht_db_sync.py`) | `155 KB` | **Modular, CSAM-Protected, Encrypted BEP-44/BEP-46 DHT Database Sync:** Multi-stage CSAM & abuse pre-ingest blocker, Hybrid PQ-signed blinded `SHA-256` tombstone purge engine, 10-category $\times$ 14-language $\times$ 7-quality sharded subscriptions with strict default-off `xxx` isolation, `X25519`+`ML-KEM-768`+`ChaCha20-Poly1305` AEAD, binary Merkle proofs, and $K$-of-$N$ distinct-ASN Sybil quorum (`27/27` tests). |

---

## Quickstart: Unified `ArmigerUltraPipeline` SDK (11 Heads + All 6 Plugins)

`armiger_ultra_pipeline.py` integrates the ONNX/PyTorch 8-head classifier (`armiger_inference.py`) with all **6 official plugins** (`SwarmResidualAdapter`, `SwarmPriors`, `ZeroCueCatalogPlugin`, `AudioLanguageGuard`, `CalibrationAndDHT` + `ArmigerDHTRSIEngine`, and `ArmigerDHTDBSync` + `CSAMGuard`) behind a single API:

```python
from armiger_ultra_pipeline import ArmigerUltraPipeline

# Load full-precision Armiger Ultra (or pass variant="int8" for the 8.3 MB CPU fast-path)
pipeline = ArmigerUltraPipeline.from_directory(".")

pred = pipeline.classify({
    "infohash": "1111222233334444555566667777888899990000",
    "name": "Sintel.2010.2160p.UHD.BluRay.REMUX.HEVC.TrueHD.7.1.Atmos-OPEN",
    "size": 74_500_000_000,
    "files": [{"path": "Sintel.2010.2160p.mkv", "size": 74_500_000_000}],
    "swarm": {
        "tracker_known": True,
        "tracker_seeders": 320,
        "tracker_leechers": 18,
        "tracker_completed": 2450,
    },
})

print(pred["category"])                           # "movie" (0% flip guarantee across seeders=0..10000)
print(pred["mondrian_gates"]["passed_all"])       # True (CSAM guard + Mondrian conformal gates + honeypot guard)
print(pred["swarm_viability"]["viability_tier"])  # "alive_viable" (computed via 48-tensor swarm_adapter.pt)
print(list(pred["heads"].keys()))                 # All 11 unified heads
```

Run the bundled 94-test verification suite (`test_armiger_ultra_pipeline.py`, `test_armiger_dht_rsi.py`, `test_dht_db_sync.py`):
```bash
pytest test_armiger_ultra_pipeline.py test_armiger_dht_rsi.py test_dht_db_sync.py
```

---

## Audited Accuracy Across All 5 Evaluation Splits (`N=10,262` Unmutated Ground-Truth Keys)

Every metric below is reproducibly verifiable on the original unmutated evaluation keys (`rows.y`) using the shipped `armiger_inference.py` and `zero_cue_plugin.py`:

| Evaluation Split | Valid `N` | Standalone Neural (`FP32` / `INT8`) | Fair Blind Hybrid (`+ ZeroCueCatalogPlugin`) | Lift | Primary Error Mode Resolved by Hybrid Plugin |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`oct15_confirmation` (Gold Standard)** | `2,835` | `95.03%` / `95.06%` | **`95.49%`** (`97.15%` Movie-vs-TV) | `+0.46 pp` | Bare-title films, multi-season packs, & siterips |
| **`fresh_confirmation` (Fresh Holdout)** | `1,402` | `94.51%` / `94.44%` | **`94.58%`** / `94.51%` | `+0.07 pp` | Zero-cue films & anime OVAs |
| **`beyond_confirmation` (Hard Edge-Cases)** | `1,025` | `92.88%` / `92.88%` | **`95.12%`** (`98.78%` 4-Med) | **`+2.24 pp`** | Ambiguous comic archives, TV packs, & bare titles |
| **`tc_truespec_2000` (Specialist Zero-Cue)** | `2,000` | `87.40%` / `87.35%` | **`91.50%`** (`99.85%` 4-Med) | **`+4.10 pp`** | Zero-cue anime films vs. TV series (`BLAKE2b-64` catalog) |
| **`tc_truespec_multilang_3000` (Multilingual)** | `3,000` | `97.50%` / `97.37%` | **`98.47%`** / `98.40%` | **`+0.97 pp`** | Non-English bare titles & multi-season packs |
| **5-Split Weighted Mean (`N=10,262`)** | **`10,262`** | **`94.00%`** / **`93.95%`** | **`95.42%`** / **`95.39%`** | **`+1.42 pp`** | Consistent lift across all 5 evaluation splits |
| **5-Split Macro Mean (`N=10,262`)** | **`10,262`** | **`93.46%`** / **`93.42%`** | **`95.03%`** / **`95.00%`** | **`+1.57 pp`** | Zero `info_hash` memorization (`BLAKE2b-64` title keys only) |

---

## Hardware-Attributed Latency & Footprint Benchmarks

| Runtime & Precision | Hardware Platform | Footprint | Single-Item Latency (p50) | Batch-64 Latency (per item) | Batch-64 Throughput |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`armiger_ultra.pt` (PyTorch CUDA FP32)** | **NVIDIA RTX GPU** | `32.2 MB` | `2.79 ms` (`358 items/s`) | **`0.21 ms`** | **`4,795 items/s`** |
| **`armiger_ultra.pt` (PyTorch MPS FP32)** | **Apple M4 Pro GPU (MPS)** | `32.2 MB` | `1.12 ms` | **`0.29 ms`** | **`3,416 items/s`** |
| **`armiger.int8.onnx` (ONNX Runtime INT8)** | **Apple M4 Pro CPU** | **`8.3 MB`** | **`2.44 ms`** (`410 items/s`) | **`3.07 ms`** | **`326 items/s`** |
| **`armiger_ultra.fp32.onnx` (ONNX Runtime FP32)** | **Apple M4 Pro CPU** | `32.3 MB` | `1.12 ms` (`893 items/s`) | `3.32 ms` | `301 items/s` |

---

## 1. Categorical & Open-Source Torrent Classifier Comparisons (Unmutated Ground-Truth Keys)

### 7-Dimension Comparison vs. Open-Source Classifiers & Parsers

![Armiger vs Open-Source Torrent Classifiers Across 7 Core Dimensions](assets/armiger_vs_torrent_classifiers_6dims.png)

| System / Model | Included Here? | Footprint | 11-Class Category (`oct15` `N=2,835`) | Movie vs. TV (`N=1,296`) | Catalog Matching (`N=1,296`) | Audio Language (`N=545`) | Anime & Dub (`N=507`) | Adult / NSFW Gate (`N=2,781`) | Junk / Malware (`N=2,170`) | Swarm Liveliness (AUROC) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Armiger Ultra (Fair Hybrid)** | **Yes** | **`32.3 MB`** (`+26 MB` cat.) | **`95.49%`** (`98.47%` mlang) | **`97.15%`** | **`96.39%`** | **`97.80%`** | **`99.61%`** | **`98.53%`** | **`100.00%`** (`99.90%` beyond) | **`96.83%`** Whse / **`92.42%`** Pub |
| **Armiger (`armiger.int8.onnx` Neural)** | **Yes** | **`8.3 MB`** | **`95.06%`** (`97.37%` mlang) | **`96.60%`** | **`95.85%`** | **`96.88%`** | **`99.61%`** | **`98.53%`** | **`100.00%`** (`99.90%` beyond) | **`86.64%`** Priors (`92.42%` w/ Adapter) |
| **MIMMO BERT (360M)** | — | `~1,400 MB` | `72.35%` | `56.92%` (`93.61%` 4-Med) | `60.03%` | N/A | N/A | `91.05%` | N/A | N/A |
| **MIMMO Cascade** | — | `~1,400 MB` | `71.43%` | `57.00%` (`93.97%` 4-Med) | `60.11%` | N/A | N/A | `89.68%` | N/A | N/A |
| **RTN (`rank-torrent-name`)** | — | Code Only | `53.77%` | `81.73%` | `80.56%` | `95.33%` | `77.65%` | `79.00%` | `50.38%` (`2.2%` rec.) | N/A |
| **PTT / Parsett** | — | Code Only | `53.77%` | `81.73%` | `80.56%` | `95.33%` | `77.65%` | `79.00%` | N/A | N/A |
| **GuessIt** | — | Code Only | `38.83%` | `84.68%` | `84.06%` | `96.01%` | `75.90%` | N/A | N/A | N/A |
| **PTN (`parse-torrent-title`)** | — | Code Only | `35.72%` | `77.84%` | `77.60%` | `95.05%` | `68.65%` | N/A | N/A | N/A |
| **Anitomy** | — | Code Only | `33.52%` | `73.09%` | `72.78%` | `94.23%` | `74.00%` | N/A | N/A | N/A |
| **Bitmagnet Built-In** | — | Code Only | `35.29%` | `43.39%` (`88.4%` ans.) | `43.39%` | `95.26%` | N/A | `76.73%` | `54.60%` (`9.2%` rec.) | N/A |

### Per-Category Accuracy Breakdown Across All 10 Media & Software Categories (Unmutated Keys)

![Categorical Accuracy Breakdown Across All 10 Media & Software Categories](assets/armiger_categorical_breakdown.png)

| Content Category | `oct15` (`N=2,835`) Armiger Ultra (Fair Hybrid) | `oct15` (`N=2,835`) Armiger (INT8 Neural) | `oct15` MIMMO BERT (360M) | `oct15` Bitmagnet Built-In | `beyond` (`N=1,025`) Armiger Ultra (Fair Hybrid) | `beyond` (`N=1,025`) Armiger (Neural) | `beyond` OpenAI GPT (`gpt-6-luna`)* | `beyond` Google Gemini (`gemini-3.8-flash-low`)* |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Audiobook** | **`97.89%`** | **`97.89%`** | `0.00%` | `9.47%` | **`100.00%`** | **`100.00%`** | **`100.00%`** | `98.67%` |
| **Adult / XXX** | **`97.80%`** | **`97.80%`** | `95.50%` | `39.46%` | `91.95%` | `91.95%` | `97.70%` | **`98.85%`** |
| **TV Show** | **`97.50%`** | `97.00%` | `58.40%` | `28.62%` | `96.32%` | `86.03%` | **`99.26%`** | `95.59%` |
| **Movie** | **`96.83%`** | `96.26%` | `78.85%` | `55.83%` | **`96.84%`** | `92.63%` | `95.79%` | `95.79%` |
| **Software** | **`93.75%`** | **`93.75%`** | `59.38%` | `15.62%` | `94.59%` | `94.59%` | **`97.30%`** | `78.38%` |
| **Music** | **`90.71%`** | **`90.71%`** | `77.86%` | `7.86%` | `83.33%` | `83.33%` | **`100.00%`** | **`100.00%`** |
| **Game** | **`90.16%`** | `86.89%` | `0.00%` | `0.00%` | `96.30%` | `95.56%` | **`99.26%`** | **`99.26%`** |
| **Ebook** | **`88.00%`** | **`88.00%`** | `80.00%` | `4.00%` | `98.40%` | `98.40%` | `96.80%` | **`99.20%`** |
| **Course** | **`81.82%`** | `78.79%` | `0.00%` | `0.00%` | `92.00%` | `92.00%` | `96.00%` | **`99.00%`** |
| **Comic** | **`66.67%`** | **`66.67%`** | `0.00%` | `0.00%` | **`98.33%`** | `95.00%` | **`99.17%`** | `95.83%` |

*Note on `beyond_confirmation` (`N=1,025`): `gpt-6-luna` and `gemini-3.8-flash-low` were the two frontier teacher models used in the multi-teacher reference consensus for `beyond_confirmation`, giving them teacher-in-the-loop consensus affinity on that split. Independent blind `claude-haiku-4-5` (`N=240`) scored `93.75%` category accuracy and local `Vinci-MLE-8B` scored `80.77%`.*

---

## 2. Frontier Cloud LLM & Local GPU LLM Comparisons

### Accuracy Across 6 Core Dimensions vs. Frontier LLMs

![Armiger vs Frontier Cloud LLMs and Local GPU LLM](assets/armiger_vs_llms_overall_and_6dims.png)

### Operational Cost & Token Rate vs. Frontier LLMs

![Cost Benchmark vs Frontier Cloud LLMs and Local GPU LLM](assets/armiger_vs_llms_cost_comparison.png)

| Model | Overall 6-Dim Mean | 11-Class Category | Catalog Matching | Audio Language | Anime & Dub | Junk / Malware (Bal.) | Swarm Liveliness (AUROC) | Measured Hardware Speed | Cost per 1M Torrents (Batched / Real-Time) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Armiger Ultra (Fair Hybrid)** | **`97.55%`** | `95.49%` (`oct15`) / `95.12%` (`beyond`) | **`96.39%`** | `97.80%` | `98.90%` (`99.61%` `oct15`) | `99.90%` (`100%` `oct15`) | **`96.83%`** | **`0.21 ms` RTX / `0.29 ms` M4 Pro MPS** | **`$0.00` API** (`$0.0040` compute) |
| **[Armiger](https://huggingface.co/NorviTech/armiger) (`armiger.int8.onnx` Neural)** | **`96.50%`** | `95.06%` (`oct15`) / `92.88%` (`beyond`) | `95.85%` | `96.88%` | `98.90%` (`99.61%` `oct15`) | `99.90%` (`100%` `oct15`) | `92.42%` (`86.64%` Priors) | **`2.44 ms` single / `3.07 ms` batch-64 (M4 Pro CPU)** | **`$0.00` API** (`$0.0020` compute) |
| **Google Gemini (`gemini-3.8-flash-low`)\*** | `96.35%` | `97.13%`* | `94.05%`* | **`99.88%`**\* | `97.82%`* | **`100.00%`** | `89.20%` | `~450 ms` | `$94.90` / `$147.00` |
| **Anthropic Claude (`claude-haiku-4-5`)** | `93.02%` | `93.75%` | `86.11%` | `95.68%` | **`99.44%`** | `93.37%` | `89.76%` | `~310 ms` | `$712.00` / `$1,220.00` |
| **OpenAI GPT (`gpt-6-luna`)\*** | `87.85%` | **`98.02%`**\* | `91.77%`* | `99.65%`* | `97.82%`* | `50.00%` (`0%` rec.) | `89.82%` | `~365 ms` | `$127.30` / `$220.50` |
| **Local GPU LLM (`Vinci-MLE-8B-1.0`)** | `85.71%` | `80.77%` | `77.88%` | `97.30%` | `93.79%` | `98.21%` | `66.32%` | `~185 ms` (`5.4/s` RTX 4070 TiS) | `$0.00` API (`$1.85` local power) |

---

## Intended Use, Copyright & Legal Compliance

`Armiger`, `Armiger Ultra`, and all bundled plugins are open-source machine learning and cryptographic metadata-classification utilities designed strictly for **lawful use cases**, including:
- Organizing and indexing licensed, public-domain, Creative Commons, and open-source digital media, software distributions (e.g., Linux ISOs, open-source games), and educational archives (e.g., OpenCourseWare, LibriVox, Project Gutenberg).
- Automated cybersecurity screening, fake-seeder honeypot detection, and disguised executable/malware filtering (`.mkv.exe`, `.scr`, `.vbs`) in peer-to-peer and decentralized storage research.
- Child safety (CSAM/CSAE) pre-ingest blocking, cryptographic tombstone revocation, and adult/NSFW namespace isolation for self-hosted and decentralized indexers.

All 929,400 lookup keys in [`validated_zero_cue_catalog.json`](validated_zero_cue_catalog.json) and all release-group / private-tracker token sets in [`armiger_inference.py`](armiger_inference.py) and [`swarm_priors.py`](swarm_priors.py) are stored exclusively as irreversible **64-bit keyed BLAKE2b digests** (`digest_size=8`) so zero copyrighted titles or scene/tracker identifiers appear in plaintext. All test fixtures and examples use Creative Commons (CC-BY / CC0) or public-domain works.

**NorviTech does not condone, encourage, or support copyright infringement or the unauthorized distribution of copyrighted works.** Users are solely responsible for ensuring that their use of these models and plugins complies with all applicable local, national, and international copyright and data-protection laws.

- **License:** [Apache License, Version 2.0](LICENSE)
- **Third-Party & Creative Commons Attributions:** [`NOTICE`](NOTICE)
- **Copyright, Privacy, Child Safety & Takedown Policy:** [`LEGAL.md`](LEGAL.md)
- **Security Vulnerability Reporting:** [`SECURITY.md`](SECURITY.md)
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
