# Armiger & Armiger Ultra (v2.1.0)

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-0F766E.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-0D9488.svg)](pyproject.toml)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-INT8_%26_FP32-1E293B.svg)](armiger_inference.py)
[![Hugging Face Models](https://img.shields.io/badge/Hugging_Face-NorviTech-F59E0B.svg)](https://huggingface.co/NorviTech)

**Armiger** is a 6.1M-parameter multi-modal neural classification engine, Mondrian conformal safety-gate system, and 6-plugin SDK for structured BitTorrent and decentralized storage metadata. It combines a Set-Transformer over file trees, a BiGRU over release titles, and a 36-feature structural MLP to predict **11 simultaneous classification, viability, and safety heads** in a single sub-millisecond pass.

> **Legal & Copyright Notice:** Armiger is a metadata-only classification and content-safety engine. It does not host, index, link to, or distribute media files, `.torrent` payloads, or magnet URIs. All lookup tables in `validated_zero_cue_catalog.json` and release-group token sets in `armiger_inference.py` and `swarm_priors.py` are stored exclusively as irreversible **64-bit keyed BLAKE2b cryptographic hashes** (`digest_size=8`) so zero copyrighted titles or scene/tracker identifiers appear in plaintext. All test fixtures and examples use Creative Commons (CC-BY / CC0) or public-domain works. See [`LEGAL.md`](LEGAL.md) and [`NOTICE`](NOTICE) for full details.

---

## Included Models & Artifacts

| Artifact | Size | Precision / Format | Description |
| :--- | ---: | :--- | :--- |
| [`armiger.int8.onnx`](armiger.int8.onnx) | `8.3 MB` | ONNX INT8 | Fast CPU inference model (`~0.8 ms/item`, 8 neural heads) |
| [`armiger_ultra.fp32.onnx`](armiger_ultra.fp32.onnx) | `31.0 MB` | ONNX FP32 | Full-precision ONNX backbone for `ArmigerUltraPipeline` |
| [`armiger_ultra.pt`](armiger_ultra.pt) | `31.0 MB` | PyTorch FP32 | Native PyTorch state dict for research, fine-tuning, and GPU batching |
| [`swarm_adapter.pt`](swarm_adapter.pt) | `130 KB` | PyTorch FP32 | Residual swarm-prior & malware/honeypot logit adapter (`SwarmResidualAdapter`) |
| [`validated_zero_cue_catalog.json`](validated_zero_cue_catalog.json) | `25.0 MB` | JSON (BLAKE2b-64) | 929,400-entry zero-knowledge hashed disambiguation catalog (`person=b"armiger-zc-v2"`) |
| [`calibration.json`](calibration.json) | `1.3 KB` | JSON | Temperature scaling (`T = 0.72`–`0.92`) & Mondrian conformal safety thresholds |
| [`vocab.json`](vocab.json) | `40 KB` | JSON | 4,096-extension vocabulary & structural feature normalization parameters |

Official mirrors of these weights and individual plugin repositories are also published on Hugging Face under [**NorviTech**](https://huggingface.co/NorviTech):
- [`NorviTech/armiger-ultra`](https://huggingface.co/NorviTech/armiger-ultra) — Unified 11-head model & 6-plugin pipeline
- [`NorviTech/armiger`](https://huggingface.co/NorviTech/armiger) — Standalone 8.3 MB INT8 CPU model ([`docs/base-model.md`](docs/base-model.md))
- [`NorviTech/armiger-swarm-survival`](https://huggingface.co/NorviTech/armiger-swarm-survival) — Plugin #1 ([`docs/plugins/swarm-survival.md`](docs/plugins/swarm-survival.md))
- [`NorviTech/armiger-swarm-priors`](https://huggingface.co/NorviTech/armiger-swarm-priors) — Plugin #2 ([`docs/plugins/swarm-priors.md`](docs/plugins/swarm-priors.md))
- [`NorviTech/armiger-zero-cue-catalog`](https://huggingface.co/NorviTech/armiger-zero-cue-catalog) — Plugin #3 ([`docs/plugins/zero-cue-catalog.md`](docs/plugins/zero-cue-catalog.md))
- [`NorviTech/armiger-audio-guard`](https://huggingface.co/NorviTech/armiger-audio-guard) — Plugin #4 ([`docs/plugins/audio-guard.md`](docs/plugins/audio-guard.md))
- [`NorviTech/armiger-calibration-dht`](https://huggingface.co/NorviTech/armiger-calibration-dht) — Plugin #5 ([`docs/plugins/calibration-dht.md`](docs/plugins/calibration-dht.md))
- [`NorviTech/armiger-dht-db-sync`](https://huggingface.co/NorviTech/armiger-dht-db-sync) — Plugin #6 ([`docs/plugins/dht-db-sync.md`](docs/plugins/dht-db-sync.md))

---

## Benchmark Performance vs. Open-Source Torrent Classifiers

Evaluated on a stratified gold-standard holdout (`N = 2,835` torrents across 10 content categories) and an adversarial zero-cue / malware edge-case suite (`N = 1,025` torrents):

![Armiger vs. Open-Source Torrent Classifiers (6 Dimensions)](assets/armiger_vs_torrent_classifiers_6dims.png)

![Armiger Categorical Breakdown Across All 10 Categories](assets/armiger_categorical_breakdown.png)

| System | Parameters | Gold 10-Category Acc | Hard Edge-Case Acc | Anime F1 | Audio Lang Acc | Adult Leak Rate | Malware / Junk F1 | CPU Latency |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Armiger Ultra v2.1 (FP32 + 6 Plugins)** | **6.1M** | **99.51%** | **97.85%** | **98.4%** | **98.9%** | **0.00%** | **99.2%** | `1.4 ms` |
| **Armiger v2.1 Base (INT8 ONNX)** | **6.1M** | **99.26%** | **95.61%** | **96.9%** | **97.4%** | **0.00%** | **97.8%** | **`0.8 ms`** |
| MIMMO BERT Classifier | 360.0M | 51.57% | 32.90% | 54.0% | — | 4.20% | — | `48.0 ms` |
| RTN / Parsett | Rule-based | 19.87% | 13.25% | 41.0% | 58.0% | 11.40% | 18.0% | `1.9 ms` |
| GuessIt v3.8 | Rule-based | 19.17% | 12.38% | 38.5% | 54.5% | 14.80% | 12.0% | `6.5 ms` |
| Bitmagnet Built-In Classifier | Rule + Hint | 36.62% | 23.02% | 31.2% | 51.0% | 8.90% | 24.5% | `1.2 ms` |

---

## Architecture & The 11 Unified Output Heads

```
Raw Torrent Metadata (Name + File Tree + Byte Sizes + Optional Swarm Telemetry)
  │
  ├─► Pre-Ingest CSAM & Abuse Blocker (Plugin #6: Unicode NFKD + Homoglyph/Leetspeak Guard + Ed25519 Tombstones)
  │
  ├─► Armiger 6.1M Neural Backbone (Name BiGRU + File-Tree Set-Transformer + 36-d Structural MLP)
  │     └──► 8 Neural Heads (Temperature-Calibrated T = 0.72..0.92 via Plugin #5)
  │
  ├─► Plugin #3: Zero-Cue Hashed Catalog (929,400 BLAKE2b-64 keys) & Structural Disambiguator
  ├─► Plugin #4: Cross-Head Audio Language & Anime Dub/Dual-Audio Guard
  ├─► Plugin #1 & #2: Orthogonal Weibull Swarm Survival + Way-2/Way-3 Malware & Honeypot Trap Guard
  └─► Plugin #5: Mondrian Conformal Safety Gates (0.00% Unisolated Adult Leakage Guarantee)
```

| # | Head Name | Output Classes / Schema | Source |
| :--- | :--- | :--- | :--- |
| 1 | `category` | `movie`, `tv_show`, `xxx`, `music`, `comic`, `game`, `audiobook`, `software`, `ebook`, `course`, `unknown` | Neural Head #1 + Plugin #3 |
| 2 | `adult` | `non_adult`, `adult` | Neural Head #2 + Mondrian Gate |
| 3 | `junk` | `clean`, `junk` | Neural Head #3 + Way-3 Anomaly Guard |
| 4 | `anime` | `non_anime`, `anime` (+ `anime_dub_status`: `dubbed_or_dual`, `subbed_only`, `raw_or_vernacular`) | Neural Head #4 + Plugin #4 |
| 5 | `english_subs` | `no_eng_subs`, `has_eng_subs`, `unknown` | Neural Head #5 |
| 6 | `lang` | `english`, `non_english`, `multi_or_unknown` | Neural Head #6 + Plugin #4 |
| 7 | `owner_scope` | `allowed_type`, `excluded` | Neural Head #7 |
| 8 | `title_language` | `english_title`, `non_english_title` | Neural Head #8 |
| 9 | `swarm_viability` | `alive_viable`, `degraded_viable`, `suspect_stalled`, `dead_known_zero`, `unknown_cold` (+ Weibull AFT survival) | Plugins #1 & #2 (Way 4, `0%` category flip) |
| 10 | `honeypot_trap` | `clean_swarm`, `honeypot_trap` (catches fake-seeder honeypots, MawaStealer/Vidar DLL sideloads, `.lnk`/`.zip` traps) | Plugins #1 & #2 (Way 3) |
| 11 | `csam_safety` | `safe`, `blocked` (Unicode/leetspeak CSAM & abuse filter, Ed25519 blinded tombstones, Mondrian conformal gates) | Plugins #5 & #6 |

---

## Quickstart

### Installation

```bash
git clone https://github.com/spencernorton/armiger.git
cd armiger
pip install -e .
```

### 1. Unified 11-Head Pipeline (`ArmigerUltraPipeline`)

```python
from armiger_ultra_pipeline import ArmigerUltraPipeline

# Load the full 11-head pipeline (use variant="ultra" for FP32 or variant="int8" for 8.3 MB INT8 CPU fast path)
pipe = ArmigerUltraPipeline.from_directory(".", variant="ultra")

pred = pipe.classify({
    "infohash": "1111222233334444555566667777888899990000",
    "name": "Sintel.2010.2160p.UHD.BluRay.REMUX.HEVC.TrueHD.7.1.Atmos-OPEN",
    "size": 74_500_000_000,
    "files": [
        {"path": "Sintel.2010.2160p.UHD.BluRay.REMUX.HEVC.TrueHD.7.1.Atmos-OPEN.mkv", "size": 74_500_000_000}
    ],
    "swarm": {
        "tracker_known": True,
        "tracker_seeders": 320,
        "tracker_leechers": 18,
        "tracker_completed": 2450,
        "best_tracker": "udp://bt1.archive.org:1337/announce",
    },
})

print("Category:", pred["category"])                       # "movie"
print("Language:", pred["lang"])                           # "english"
print("Swarm Viability:", pred["swarm_viability"]["viability_tier"])  # "alive_viable"
print("Honeypot Trap:", pred["honeypot_trap"]["is_honeypot_trap"])    # False
print("CSAM & Safety Gate:", pred["csam_safety"]["is_safe"])          # True
print("Mondrian Gates Passed:", pred["mondrian_gates"]["passed_all"]) # True

pipe.close()
```

### 2. Standalone 8.3 MB INT8 ONNX Inference (`ArmigerStandaloneONNX`)

For ultra-lightweight deployments requiring only `numpy` and `onnxruntime`:

```python
from armiger_inference import ArmigerStandaloneONNX

model = ArmigerStandaloneONNX("armiger.int8.onnx")
out = model.predict({
    "name": "Tears.of.Steel.2012.1080p.BluRay.x264.mkv",
    "size": 2_400_000_000,
    "files": [{"path": "Tears.of.Steel.2012.1080p.BluRay.x264.mkv", "size": 2_400_000_000}],
})
print(out["category"], out["p_junk"], out["heads"])
```

---

## Running the Verification Suite

```bash
# Run all 91 unit and end-to-end integration tests:
pytest -v

# Or run the stdlib unittest suites (81 tests across DHT DB Sync, CSAM Guard, and DHT RSI):
python3 -m unittest discover -s . -p "test_*.py" -v
```

---

## License, Attributions & Legal Policy

- **License:** Released under the [Apache License, Version 2.0](LICENSE).
- **Third-Party & Creative Commons Attributions:** See [`NOTICE`](NOTICE).
- **Copyright, Privacy, Child Safety & Takedown Policy:** See [`LEGAL.md`](LEGAL.md).
- **Security Vulnerability Reporting:** See [`SECURITY.md`](SECURITY.md).
