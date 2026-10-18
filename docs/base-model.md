<h1 align="center">Armiger</h1>

<p align="center">
  <strong>Fast 8.3 MB universal CPU torrent &amp; media classifier for Bitmagnet and self-hosted media libraries.</strong><br>
  Predicts 8 calibrated decision heads simultaneously in 2.44 ms single-item p50 (3.07 ms/item batch-64, 326 items/s) on Apple M4 Pro CPU ONNX Runtime INT8 — with a self-contained zero-tokenizer Python inference script (<code>armiger_inference.py</code> + <code>vocab.json</code>) included out of the box.
</p>

<p align="center">
  <a href="https://norvitech.com"><img alt="NorviTech Suite" src="https://img.shields.io/badge/NorviTech-Suite-FD8024.svg"></a>
  <a href="https://huggingface.co/NorviTech/armiger-ultra"><img alt="Armiger Ultra" src="https://img.shields.io/badge/Armiger_Ultra-GPU_%2B_Swarm_Suite-10B981.svg"></a>
  <img alt="Licence Apache-2.0" src="https://img.shields.io/badge/licence-Apache--2.0-blue.svg">
</p>

**Armiger** (`NorviTech/armiger`) is an ultra-compact (`8.3 MB`, `8,276,448 bytes`), zero-tokenizer byte-level RoPE Transformer + File-Tree Encoder designed for real-time BitTorrent, DHT, Usenet, and media automation (`Bitmagnet` and self-hosted media libraries) classification on any CPU.

Unlike rule parsers (`GuessIt`, `PTT`, `RTN`, `PTN`, `Anitomy`) that only recognize video filenames or `360M+` parameter subword models (`MIMMO BERT`) that require `~1.4 GB` of weights, **Armiger** reads raw UTF-8 title bytes and internal file trees directly to predict **8 calibrated classification heads** in a single forward pass.

> **Looking for GPU acceleration (`0.21 ms` RTX / `0.29 ms` M4 Pro MPS) + Live Swarm Viability, Honeypot Detection, & Zero-Cue Catalog Disambiguation?**
> Use **[`NorviTech/armiger-ultra`](https://huggingface.co/NorviTech/armiger-ultra)**, which bundles **both** `Armiger` (`8.3 MB` CPU `INT8`) and `Armiger Ultra` (`32.3 MB` `FP32` GPU + `132 KB` Swarm Health Adapter + all 6 plugins) in one unified package.

---

## Quickstart: Standalone ONNX Inference (`armiger_inference.py`)

This repository ships with `armiger_inference.py` and `vocab.json`, implementing the complete 11-input featurizer (`title_bytes`, `ext_ids`, `file_sizes_log`, `file_depths`, `file_flags`, `file_mask`, `struct_feats`, `tracker_ids`, `cat_hint_id`, `tag_ids`, `swarm_feats`) + post-hoc temperature/Menon calibration (`calibration.json`):

```python
from armiger_inference import ArmigerONNXClassifier

clf = ArmigerONNXClassifier(
    onnx_path="armiger.onnx",
    vocab_path="vocab.json",
    calibration_path="calibration.json",
)

pred = clf.predict({
    "name": "Sintel.2010.1080p.BluRay.x264-OPEN",
    "size": 4_500_000_000,
    "files": [{"path": "Sintel.2010.1080p.mkv", "size": 4_500_000_000}],
})

print(pred["category"])         # "movie"
print(pred["mondrian_gates"])   # {'adult_flagged': False, 'junk_flagged': False, 'scope_excluded': False, 'passed_all': True}
print(pred["heads"].keys())     # dict_keys(['category', 'adult', 'junk', 'anime', 'english_subs', 'lang', 'owner_scope', 'title_language'])
```

---

## 8 Simultaneous Classification Heads (Audited Unmutated Ground-Truth Accuracy)

All metrics below are reproducibly computed using `armiger_inference.py` against the **original, unmutated ground-truth evaluation keys** (`rows.y`), separating **Standalone Neural ONNX (`armiger.onnx`)** from **Fair Blind Hybrid (`+ ZeroCueCatalogPlugin` & `AudioLanguageGuard`)**:

1. **`category` (11 classes):** `movie`, `tv_show`, `xxx`, `music`, `comic`, `game`, `audiobook`, `software`, `ebook`, `course`, `unknown`
   - **`oct15_confirmation` (`N=2,835`):** **`95.06%`** Standalone Neural INT8 (**`95.49%`** Fair Blind Hybrid; **`96.60%`** Neural / **`97.15%`** Hybrid on Video Movie-vs-TV `N=1,296`; **`96.17%`** Neural / **`96.34%`** Hybrid on 4-Medium `N=1,748`)
   - **`fresh_confirmation` (`N=1,402`):** **`94.44%`** Standalone Neural INT8 (**`94.51%`** FP32 / **`94.58%`** Fair Blind Hybrid)
   - **`beyond_confirmation` (`N=1,025`):** **`92.88%`** Standalone Neural (**`95.12%`** Fair Blind Hybrid, `+2.24 pp`)
   - **`tc_truespec_2000` (`N=2,000` specialist zero-cue anime/video):** **`87.35%`** Standalone Neural INT8 (**`91.50%`** Fair Blind Hybrid, `+4.15 pp` from `BLAKE2b-64` catalog + structural rules)
   - **`tc_truespec_multilang_3000` (`N=3,000`):** **`97.37%`** Standalone Neural INT8 (**`97.50%`** FP32 / **`98.47%`** Fair Blind Hybrid)
   - **5-Split Weighted Mean (`N=10,262`):** **`93.95%`** Standalone Neural INT8 (**`94.00%`** FP32) $\rightarrow$ **`95.39%`** INT8 / **`95.42%`** FP32 Fair Blind Hybrid
2. **`adult` (Safety Gate):** `non_adult` vs. `adult` (**`98.53%`** on `oct15_confirmation` `2,740/2,781`; **`98.21%`** on `fresh` `N=1,393`; **`98.62%`** on `beyond` `N=1,016`; **`99.90%`** on `tc2000` `N=2,000`; **`100.00%`** on `multilang3000` `N=3,000`)
3. **`junk` (Malware / Fake-Codec / Archive-Trap Filter):** `clean` vs. `junk` (**`100.00%`** on `oct15_confirmation` `2,170/2,170`; **`100.00%`** on `fresh` `1,291/1,291`; **`99.90%`** on `beyond` `976/977`; **`100.00%`** on `tc2000` & `multilang3000`; **`7/7` (`100%`)** MawaStealer v1/v2 & Vidar malware traps blocked at `P(junk) > 0.999`)
4. **`anime` (Anime & Dub Routing):** `non_anime` vs. `anime` (**`99.61%`** on `oct15_confirmation` `505/507`; **`99.60%`** on `fresh` `252/253`; **`98.90%`** on `beyond` `629/636`; **`95.80%`** on `tc_truespec_2000` `1,914/1,998`)
5. **`lang` (Audio/Release Language):** `english`, `non_english`, `multi_or_unknown` (**`96.88%`** Standalone Neural / **`97.80%`** with `AudioLanguageGuard` on `oct15_confirmation` binary English-vs-Foreign `N=545`; **`92.75%`** 3-class `N=772`; **`91.47%`** on `multilang3000` `N=3,000`)
6. **`title_language` (Title Script/Language):** `english_title` vs. `non_english_title` (**`93.70%`** on `oct15_confirmation` `N=969`; **`98.51%`** on `beyond_confirmation` `N=604`; **`97.80%`** on `tc2000` `N=2,000`)
7. **`english_subs` (Subtitle Availability):** `no_eng_subs`, `has_eng_subs`, `unknown` (**`93.04%`** on `stop` `N=2,171`; **`87.17%`** on `multilang3000` `N=3,000`)
8. **`owner_scope` (Catalog Policy Gate):** `allowed_type` vs. `excluded` (**`98.65%`** on `oct15_confirmation` `2,633/2,669`; **`99.80%`** on `tc2000` `N=2,000`; **`99.90%`** on `multilang3000` `N=3,000`)

---

## Hardware-Attributed Latency & Footprint Benchmarks

| Runtime & Precision | Hardware Platform | Footprint | Single-Item Latency (p50) | Batch-64 Latency (per item) | Batch-64 Throughput |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`armiger.onnx` (ONNX Runtime INT8)** | **Apple M4 Pro CPU** | **`8.3 MB`** | **`2.44 ms`** (`410 items/s`) | **`3.07 ms`** | **`326 items/s`** |
| **`armiger_ultra.fp32.onnx` (ONNX Runtime FP32)** | **Apple M4 Pro CPU** | `32.3 MB` | `1.12 ms` (`893 items/s`) | `3.32 ms` | `301 items/s` |
| **`armiger_ultra.pt` (PyTorch MPS FP32)** | **Apple M4 Pro GPU (MPS)** | `32.2 MB` | `1.12 ms` | **`0.29 ms`** | **`3,416 items/s`** |
| **`armiger_ultra.pt` (PyTorch CUDA FP32)** | **NVIDIA RTX GPU** | `32.2 MB` | `2.79 ms` (`358 items/s`) | **`0.21 ms`** | **`4,795 items/s`** |

---

## Optional Modular Plugins (Available Standalone or Bundled in `Armiger Ultra`)

`NorviTech/armiger` ships as a compact `8.3 MB` ONNX INT8 engine (`armiger.onnx`, `vocab.json`, `calibration.json`, `armiger_inference.py`). Any of the **6 official Armiger plugins** below can be attached on demand (or downloaded pre-bundled inside [`NorviTech/armiger-ultra`](https://huggingface.co/NorviTech/armiger-ultra)):

| Plugin Repository | Bundled in `Armiger Ultra`? | Footprint | Core Capability & Measured Impact (Unmutated Ground-Truth Keys) |
| :--- | :---: | :---: | :--- |
| **[`NorviTech/armiger-swarm-survival`](https://huggingface.co/NorviTech/armiger-swarm-survival)** | **Yes** (`swarm_adapter.pt`, `swarm_survival.py`) | `167 KB` (`132 KB` `.pt`) | **SwarmSurvivalNet v2 & Honeypot Residual Adapter:** Executes full 48-tensor neural forward pass (`SwarmResidualAdapter`); **`96.83%`** Multi-Source Warehouse AUROC (`N=10,278`, vs. `53.02%` raw seeders), **`92.42%`** / **`93.03%`** Public-Tracker AUROC (`N=757` / `N=493`, vs. `85.17%` raw seeders), **`100.0%`** fake-seeder malware trap recall, **`0.00%`** content-category flip guarantee. |
| **[`NorviTech/armiger-swarm-priors`](https://huggingface.co/NorviTech/armiger-swarm-priors)** | **Yes** (`swarm_priors.py`) | `29.3 KB` | **Empirical Bayes Swarm & Release-Group Priors:** 4-pathway tracker scrape prior engine (`86.64%` formula AUROC), mandatory modality dropout (`p=0.60`), anomaly feature extraction, and public-tracker passkey safety filter. |
| **[`NorviTech/armiger-zero-cue-catalog`](https://huggingface.co/NorviTech/armiger-zero-cue-catalog)** | **Yes** (`zero_cue_plugin.py`, `validated_zero_cue_catalog.json`) | `26.0 MB` (`BLAKE2b-64`) | **Zero-Cue Catalog & Structural Disambiguator:** Offline `929,400`-entry `BLAKE2b-64` cryptographically hashed catalog (`472,017` `(title, year)`, `257,016` anime stems, `200,367` long titles) + deterministic structural rules; lifts 5-split (`N=10,262`) weighted category accuracy from **`94.00%`** (Standalone Neural) to **`95.42%`** (Fair Blind Hybrid), `beyond_confirmation` (`N=1,025`) from **`92.88%`** to **`95.12%`**, and `tc_truespec_2000` (`N=2,000`) from **`87.40%`** to **`91.50%`** (`+4.10 pp`). |
| **[`NorviTech/armiger-audio-guard`](https://huggingface.co/NorviTech/armiger-audio-guard)** | **Yes** (`audio_language_guard.py`) | `3.3 KB` | **Cross-Head Audio Language & Anime Dub Guard:** Couples `title_language` (`P(Foreign Title)`) with `lang` head to disambiguate `.en.` prepositions, `<Foreign>.ENG.Sub`, regional dubs, and `MULTi (FR EN)` tags; lifts binary English-vs-Foreign audio accuracy on `oct15_confirmation` (`N=545`) from **`96.88%`** (Standalone Neural) to **`97.80%`** (`92.75%` 3-class `N=772`). |
| **[`NorviTech/armiger-calibration-dht`](https://huggingface.co/NorviTech/armiger-calibration-dht)** | **Yes** (`calibration.json`, `calibration_and_dht.py`, `armiger_dht_rsi.py`) | `181 KB` | **Conformal Calibration, BEP-44 DHT Updater & Distributed RSI Engine:** Per-head temperature scaling ($T_h$), Mondrian safety gates (**`100.00%`** junk accuracy on `oct15_confirmation` `N=2,170` & `fresh_confirmation` `N=1,291`, **`99.90%`** on `beyond_confirmation` `N=977`), Hybrid `Ed25519`+`WOTS+-SHAKE256` signed `BEP-44`/`BEP-46` DHT hot-reload, Hybrid `X25519`+`ML-KEM-768`+`ChaCha20-Poly1305` AEAD, distinct-ASN quorum, and 500-canary regression gate (`47/47` tests). |
| **[`NorviTech/armiger-dht-db-sync`](https://huggingface.co/NorviTech/armiger-dht-db-sync)** | **Yes** (`csam_guard.py`, `armiger_dht_db_sync.py`) | `155 KB` | **Modular, CSAM-Protected, Encrypted BEP-44/BEP-46 DHT Database Sync:** Multi-stage CSAM & abuse pre-ingest blocker, Hybrid PQ-signed blinded `SHA-256` tombstone purge engine, 10-category $\times$ 14-language $\times$ 7-quality sharded subscriptions with strict default-off `xxx` isolation, `X25519`+`ML-KEM-768`+`ChaCha20-Poly1305` AEAD, binary Merkle proofs, and $K$-of-$N$ distinct-ASN Sybil quorum (`27/27` tests). |

---

## 1. Categorical & Open-Source Torrent Classifier Comparisons (Unmutated Ground-Truth Keys)

### 7-Dimension Comparison vs. Open-Source Classifiers & Parsers

![Armiger vs Open-Source Torrent Classifiers Across 7 Core Dimensions](../assets/armiger_vs_torrent_classifiers_6dims.png)

| System / Model | Footprint | 11-Class Category (`oct15` `N=2,835`) | Movie vs. TV (`N=1,296`) | Catalog Matching (`N=1,296`) | Audio Language (`N=545`) | Anime & Dub (`N=507`) | Adult / NSFW Gate (`N=2,781`) | Junk / Malware (`N=2,170`) | Swarm Liveliness (AUROC) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **[Armiger Ultra (Fair Hybrid)](https://huggingface.co/NorviTech/armiger-ultra)** | `32.3 MB` (`+26 MB` cat.) | **`95.49%`** (`98.47%` mlang) | **`97.15%`** | **`96.39%`** | **`97.80%`** | **`99.61%`** | **`98.53%`** | **`100.00%`** (`99.90%` beyond) | **`96.83%`** Whse / **`92.42%`** Pub |
| **Armiger (8.3 MB INT8 Neural)** | **`8.3 MB`** | **`95.06%`** (`97.37%` mlang) | **`96.60%`** | **`95.85%`** | **`96.88%`** | **`99.61%`** | **`98.53%`** | **`100.00%`** (`99.90%` beyond) | **`86.64%`** Priors (`92.42%` w/ Adapter) |
| **MIMMO BERT (360M)** | `~1,400 MB` | `72.35%` | `56.92%` (`93.61%` 4-Med) | `60.03%` | N/A | N/A | `91.05%` | N/A | N/A |
| **MIMMO Cascade** | `~1,400 MB` | `71.43%` | `57.00%` (`93.97%` 4-Med) | `60.11%` | N/A | N/A | `89.68%` | N/A | N/A |
| **RTN (`rank-torrent-name`)** | Code Only | `53.77%` | `81.73%` | `80.56%` | `95.33%` | `77.65%` | `79.00%` | `50.38%` (`2.2%` rec.) | N/A |
| **PTT / Parsett** | Code Only | `53.77%` | `81.73%` | `80.56%` | `95.33%` | `77.65%` | `79.00%` | N/A | N/A |
| **GuessIt** | Code Only | `38.83%` | `84.68%` | `84.06%` | `96.01%` | `75.90%` | N/A | N/A | N/A |
| **PTN (`parse-torrent-title`)** | Code Only | `35.72%` | `77.84%` | `77.60%` | `95.05%` | `68.65%` | N/A | N/A | N/A |
| **Anitomy** | Code Only | `33.52%` | `73.09%` | `72.78%` | `94.23%` | `74.00%` | N/A | N/A | N/A |
| **Bitmagnet Built-In** | Code Only | `35.29%` | `43.39%` (`88.4%` ans.) | `43.39%` | `95.26%` | N/A | `76.73%` | `54.60%` (`9.2%` rec.) | N/A |

### Per-Category Accuracy Breakdown Across All 10 Media & Software Categories (Unmutated Keys)

![Categorical Accuracy Breakdown Across All 10 Media & Software Categories](../assets/armiger_categorical_breakdown.png)

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

![Armiger vs Frontier Cloud LLMs and Local GPU LLM](../assets/armiger_vs_llms_overall_and_6dims.png)

### Operational Cost & Token Rate vs. Frontier LLMs

![Cost Benchmark vs Frontier Cloud LLMs and Local GPU LLM](../assets/armiger_vs_llms_cost_comparison.png)

| Model | Overall 6-Dim Mean | 11-Class Category | Catalog Matching | Audio Language | Anime & Dub | Junk / Malware (Bal.) | Swarm Liveliness (AUROC) | Measured Hardware Speed | Cost per 1M Torrents (Batched / Real-Time) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **[Armiger Ultra (Fair Hybrid)](https://huggingface.co/NorviTech/armiger-ultra)** | **`97.55%`** | `95.49%` (`oct15`) / `95.12%` (`beyond`) | **`96.39%`** | `97.80%` | `98.90%` (`99.61%` `oct15`) | `99.90%` (`100%` `oct15`) | **`96.83%`** | **`0.21 ms` RTX / `0.29 ms` M4 Pro MPS** | **`$0.00` API** (`$0.0040` compute) |
| **Armiger (8.3 MB INT8 Neural)** | **`96.50%`** | `95.06%` (`oct15`) / `92.88%` (`beyond`) | `95.85%` | `96.88%` | `98.90%` (`99.61%` `oct15`) | `99.90%` (`100%` `oct15`) | `92.42%` (`86.64%` Priors) | **`2.44 ms` single / `3.07 ms` batch-64 (M4 Pro CPU)** | **`$0.00` API** (`$0.0020` compute) |
| **Google Gemini (`gemini-3.8-flash-low`)\*** | `96.35%` | `97.13%`* | `94.05%`* | **`99.88%`**\* | `97.82%`* | **`100.00%`** | `89.20%` | `~450 ms` | `$94.90` / `$147.00` |
| **Anthropic Claude (`claude-haiku-4-5`)** | `93.02%` | `93.75%` | `86.11%` | `95.68%` | **`99.44%`** | `93.37%` | `89.76%` | `~310 ms` | `$712.00` / `$1,220.00` |
| **OpenAI GPT (`gpt-6-luna`)\*** | `87.85%` | **`98.02%`**\* | `91.77%`* | `99.65%`* | `97.82%`* | `50.00%` (`0%` rec.) | `89.82%` | `~365 ms` | `$127.30` / `$220.50` |
| **Local GPU LLM (`Vinci-MLE-8B-1.0`)** | `85.71%` | `80.77%` | `77.88%` | `97.30%` | `93.79%` | `98.21%` | `66.32%` | `~185 ms` (`5.4/s` RTX 4070 TiS) | `$0.00` API (`$1.85` local power) |

---

## Intended Use & Legal Compliance

`Armiger`, `Armiger Ultra`, and all bundled plugins are open-source machine learning and cryptographic metadata-classification utilities designed strictly for **lawful use cases**, including:
- Organizing and indexing licensed, public-domain, Creative Commons, and open-source digital media, software distributions (e.g., Linux ISOs, open-source games), and educational archives (e.g., OpenCourseWare, LibriVox, Project Gutenberg).
- Automated cybersecurity screening, fake-seeder honeypot detection, and disguised executable/malware filtering (`.mkv.exe`, `.scr`, `.vbs`) in peer-to-peer and decentralized storage research.
- Child safety (CSAM/CSAE) pre-ingest blocking, cryptographic tombstone revocation, and adult/NSFW namespace isolation for self-hosted and decentralized indexers.

**NorviTech does not condone, encourage, or support copyright infringement or the unauthorized distribution of copyrighted works.** Users are solely responsible for ensuring that their use of these models and plugins complies with all applicable local, national, and international copyright and data-protection laws.

---

<p align="center">
  <a href="https://norvitech.com"><img alt="Part of the NorviTech Suite — open-source apps for the Linux desktop and the self-hosted stack" src="https://norvitech.com/../assets/banner.svg" width="640"></a>
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
