<h1 align="center">Armiger Ultra</h1>

<p align="center">
  <strong>Compact neural torrent &amp; media classification suite bundling Armiger (8.3 MB INT8 CPU), Armiger Ultra (32.3 MB FP32 GPU/CPU), the 48-tensor Swarm Viability Adapter (<code>swarm_adapter.pt</code>), and 6 modular plugins.</strong><br>
  Delivers 96.86% 6-dimension composite accuracy on the Public-Tracker cohort (97.60% with Multi-Source Corpus Swarm AUROC) on unmutated ground-truth keys, 0.21 ms/item batched on NVIDIA RTX GPU (0.29 ms on Apple M4 Pro MPS GPU, 2.44 ms single-item on M4 Pro CPU INT8), 99.23% balanced junk/sample accuracy (<code>N=977</code>), and 96.83% Multi-Source Corpus / 92.42% Public-Tracker swarm liveliness AUROC.
</p>

<p align="center">
  <a href="https://norvitech.com"><img alt="NorviTech Suite" src="https://img.shields.io/badge/NorviTech-Suite-FD8024.svg"></a>
  <a href="https://huggingface.co/NorviTech/armiger-ultra"><img alt="Hugging Face Models" src="https://img.shields.io/badge/Hugging_Face-NorviTech%2Farmiger--ultra-FFD21E.svg"></a>
  <a href="https://huggingface.co/NorviTech/armiger"><img alt="Armiger CPU" src="https://img.shields.io/badge/Armiger-8.3_MB_CPU-0284C7.svg"></a>
  <a href="LICENSE"><img alt="Licence Apache-2.0" src="https://img.shields.io/badge/licence-Apache--2.0-blue.svg"></a>
</p>

---

## Benchmarks

### Frontier Cloud LLMs & Local GPU LLM Comparison

![Armiger vs Frontier Cloud LLMs and Local GPU LLM](assets/armiger_vs_llms_overall_and_6dims.png)

![Cost Benchmark vs Frontier Cloud LLMs and Local GPU LLM](assets/armiger_vs_llms_cost_comparison.png)

- **Competitive with frontier LLMs on identical evaluation cohorts.** Achieves **`96.86%`** 6-dimension mean accuracy on the `N=757` Public-Tracker swarm cohort (**`97.60%`** with `N=10,278` Multi-Source Corpus Swarm AUROC, and **`96.85%`** with `8.3 MB` INT8 + plugins / **`96.50%`** standalone neural), compared to non-circular LOTO `Google Gemini Flash` (`94.56%` 6D), `OpenAI GPT (Fast Tier)` (`93.49%` 5D evaluated composite; not prompted for sample/junk definitions), `Anthropic Claude Haiku 4.5` (`92.88%` 6D), and `Local 8B Instruct LLM (FP16)` (`84.79%` 6D).
- **Sub-millisecond batched inference at zero API cost.** Classifies **`4,795 items/s`** (`0.21 ms/item`) on NVIDIA RTX GPU, **`3,416 items/s`** (`0.29 ms/item`) on Apple M4 Pro MPS GPU, and **`410 items/s` single-item** (`2.44 ms` p50) on Apple M4 Pro CPU INT8 — versus `185–450 ms/item` and `$94.90–$1,220.00` per million torrents for cloud LLM APIs.

### Open-Source Classifiers & 10-Category Breakdown

![Armiger vs Open-Source Torrent Classifiers Across 7 Core Dimensions](assets/armiger_vs_torrent_classifiers_6dims.png)

![Per-Category Recall Breakdown Across All 10 Media & Software Categories](assets/armiger_categorical_breakdown.png)

- **Universal 11-class coverage vs. title-only parsers.** Reaches **`95.49%`** 11-class accuracy (`97.30%` Movie-vs-TV, `96.53%` Catalog Matching) on `oct15_confirmation` (`N=2,835`) and **`95.42%`** weighted mean across all 5 evaluation splits (`N=10,262`), compared to `72.24%` for `MIMMO BERT 360M` (`~1.4 GB`), `53.58%` for `RTN`/`PTT`, and `38.69%` for `GuessIt` on the identical `N=2,835` denominator (note that `RTN`, `GuessIt`, `PTN`, and `Anitomy` are video/anime-focused title parsers rather than 11-class general classifiers).
- **Calibrated safety & junk gates.** Mondrian conformal thresholds achieve **`99.23%`** balanced junk/sample accuracy (**`99.90%`** overall accuracy, `976/977`, `65` positive items) on `beyond_confirmation`, deterministic filtering of disguised executable extensions (`.mkv.exe`, `.scr`, `.lnk`, `.vbs`), and **`98.53%`** adult/NSFW isolation (`N=2,781`).
- **Reproducible chart & provenance pipeline.** Every chart in [`assets/`](assets) is deterministically generated from [`benchmarks/canonical_eval_vectors.json`](benchmarks/canonical_eval_vectors.json) by [`benchmarks/generate_canonical_charts.py`](benchmarks/generate_canonical_charts.py) and verified by [`benchmarks/CHARTS_MANIFEST.json`](benchmarks/CHARTS_MANIFEST.json) and [`benchmarks/test_chart_integrity.py`](benchmarks/test_chart_integrity.py) (`10,000` paired bootstrap resamples with `seed=20261018`, exact two-sided McNemar $p$-values, identical-denominator lock, and LOTO anti-circularity lock). See [`docs/CHART_PIPELINE.md`](docs/CHART_PIPELINE.md).

---

## Quickstart

`armiger_ultra_pipeline.py` integrates the zero-tokenizer 8-head ONNX/PyTorch classifier (`armiger_inference.py`) with all 6 plugins behind a single Python API:

```python
from armiger_ultra_pipeline import ArmigerUltraPipeline

# Load full-precision Armiger Ultra (or pass variant="int8" for the 8.3 MB CPU fast-path)
pipeline = ArmigerUltraPipeline.from_directory(".")

pred = pipeline.classify({
    "infohash": "1111222233334444555566667777888899990000",
    "name": "Sintel.2010.2160p.UHD.Master.HEVC.FLAC.5.1-OPEN",
    "size": 74_500_000_000,
    "files": [{"path": "Sintel.2010.2160p.mkv", "size": 74_500_000_000}],
    "swarm": {
        "tracker_known": True,
        "tracker_seeders": 320,
        "tracker_leechers": 18,
        "tracker_completed": 2450,
    },
})

print(pred["category"])                           # "movie" (category head is independent of seeder counts)
print(pred["mondrian_gates"]["passed_all"])       # True (CSAM guard + Mondrian conformal gates + extension filter)
print(pred["swarm_viability"]["viability_tier"])  # "alive_viable" (computed via 48-tensor swarm_adapter.pt)
print(list(pred["heads"].keys()))                 # All 11 unified heads
```

Run the bundled verification suite:

```bash
pytest -q
```

---

## What's Included & The 6 Official Plugins

- **`armiger_ultra.fp32.onnx` (`32.3 MB`) & `armiger_ultra.pt` (`32.2 MB`)** — Full-precision 8-head byte-level RoPE Transformer + File-Tree Encoder (`0.21 ms/item` batch-64 on NVIDIA RTX GPU, `0.29 ms/item` on Apple M4 Pro MPS GPU).
- **`armiger.int8.onnx` (`8.3 MB`)** — Dynamically quantized CPU fast-path engine from the same checkpoint (`2.44 ms` single-item p50 on Apple M4 Pro CPU ONNX Runtime). See [`docs/base-model.md`](docs/base-model.md).
- **[SwarmSurvivalNet v2 Adapter](https://huggingface.co/NorviTech/armiger-swarm-survival)** (`swarm_survival.py`, `swarm_adapter.pt` · `167 KB`) — 48-tensor residual neural pass combining tracker/DHT seeder-leecher telemetry with content embeddings to reach **`96.83%`** Multi-Source Corpus AUROC (`N=10,278`, vs. `53.02%` raw seeders) and **`92.42%`** Public-Tracker AUROC (`N=757`) while keeping `category` predictions independent of seeder counts. See [`docs/plugins/swarm-survival.md`](docs/plugins/swarm-survival.md).
- **[Empirical Bayes Swarm Priors](https://huggingface.co/NorviTech/armiger-swarm-priors)** (`swarm_priors.py` · `29 KB`) — 4-pathway tracker scrape priors (`86.64%` formula AUROC), anomaly features, and passkey sanitization. See [`docs/plugins/swarm-priors.md`](docs/plugins/swarm-priors.md).
- **[Zero-Cue Catalog & Structural Disambiguator](https://huggingface.co/NorviTech/armiger-zero-cue-catalog)** (`zero_cue_plugin.py`, `validated_zero_cue_catalog.json` · `26.0 MB`) — `929,400`-entry `BLAKE2b-64` hashed catalog + structural rules, lifting `zero_cue_2000` (`N=2,000`) accuracy by **`+4.10 pp`** (`87.40%` $
ightarrow$ **`91.50%`**) and `beyond_confirmation` (`N=1,025`) by **`+2.24 pp`** (`92.88%` $
ightarrow$ **`95.12%`**). See [`docs/plugins/zero-cue-catalog.md`](docs/plugins/zero-cue-catalog.md).
- **[Cross-Head Audio Language & Anime Dub Guard](https://huggingface.co/NorviTech/armiger-audio-guard)** (`audio_language_guard.py` · `3.3 KB`) — Couples `title_language` with the `lang` head to disambiguate `.en.` prepositions, `<Foreign>.ENG.Sub` releases, regional dubs, and `MULTi` tags, lifting binary English-vs-Foreign audio accuracy on `oct15_confirmation` (`N=545`) from **`96.88%`** to **`97.80%`**. See [`docs/plugins/audio-guard.md`](docs/plugins/audio-guard.md).
- **[Conformal Calibration & Optional DHT Sync Plugins](https://huggingface.co/NorviTech/armiger-calibration-dht)** (`calibration.json`, `calibration_and_dht.py`, `armiger_dht_rsi.py`, `armiger_dht_db_sync.py`, `csam_guard.py`) — Per-head temperature scaling, Mondrian conformal thresholds, multi-stage CSAM/abuse pre-ingest blocking, and strictly opt-in (`opt_in_sharing=False` by default) experimental `BEP-44` catalog/calibration sync behind `AnonymityGuard`. See [`docs/plugins/calibration-dht.md`](docs/plugins/calibration-dht.md), [`docs/plugins/dht-db-sync.md`](docs/plugins/dht-db-sync.md), and [`docs/ANONYMOUS_OPT_IN_DHT.md`](docs/ANONYMOUS_OPT_IN_DHT.md).

---

## Separate Optional Addition: Ecosystem Connectors (`spencernorton/Armiger-Connectors`)

External HTTP, GraphQL, and Manticore connectors are maintained as a **separate repository** — **[`spencernorton/Armiger-Connectors`](https://github.com/spencernorton/Armiger-Connectors)** — distinct from the core classifier and the 6 official plugins above:

- **[Stock `bitmagnet` Connector (`bitmagnet/`)](https://github.com/spencernorton/Armiger-Connectors/tree/main/bitmagnet)** — Connects to unmodified [`bitmagnet-io/bitmagnet`](https://github.com/bitmagnet-io/bitmagnet) via GraphQL (`torrentContent.search` + `torrent.files`), pruning disguised executables (`.mkv.exe`, `.scr`, `.lnk`) via `torrent.delete` and enriching surviving items with grouped `torrent.putTags` mutations.
- **[Stock `rats-search` Connector (`rats-search/`)](https://github.com/spencernorton/Armiger-Connectors/tree/main/rats-search)** — Connects to unmodified [`librats/rats-search`](https://github.com/librats/rats-search) via REST (`:8095`), WebSocket (`:8096`), and local Manticore (`127.0.0.1:9306`), purging disguised executables via `/api/torrent.remove` and populating `contentType` and `contentCategory` (`Movie=1..Game=9`, `XXX=100`) using `rowIdFromHash` docid lookups.

---

## Audited Evaluation Splits (`N=10,262` Unmutated Ground-Truth Keys)

| Evaluation Split | Valid `N` | Standalone Neural (`FP32` / `INT8`) | Fair Hybrid (`+ ZeroCueCatalog`) | Lift |
| :--- | :---: | :---: | :---: | :---: |
| **`oct15_confirmation` (Gold Standard)** | `2,835` | `95.03%` / `95.06%` | **`95.49%`** / **`95.49%`** (`97.30%` Video) | `+0.46 pp` |
| **`fresh_confirmation` (Fresh Holdout)** | `1,402` | `94.51%` / `94.51%` | **`94.58%`** / **`94.58%`** | `+0.07 pp` |
| **`beyond_confirmation` (Hard Edge-Cases)** | `1,025` | `92.88%` / `92.88%` | **`95.12%`** (`98.78%` 4-Med) | **`+2.24 pp`** |
| **`zero_cue_2000` (Zero-Cue Specialist)** | `2,000` | `87.40%` / `87.30%` | **`91.50%`** / **`91.40%`** (`99.85%` 4-Med) | **`+4.10 pp`** |
| **`multilang_3000` (Multilingual)** | `3,000` | `97.50%` / `97.43%` | **`98.47%`** / **`98.43%`** | **`+0.97 pp`** |
| **5-Split Weighted Mean** | **`10,262`** | **`94.00%`** / **`93.98%`** | **`95.42%`** / **`95.39%`** | **`+1.42 pp`** |

---

## Intended Use, Copyright & Legal Compliance

`Armiger`, `Armiger Ultra`, and all bundled plugins are open-source machine learning metadata-classification utilities designed strictly for **lawful use cases**, including organizing licensed, public-domain, Creative Commons, and open-source digital media, automated disguised-executable screening (`.mkv.exe`, `.scr`, `.vbs`), and child-safety (CSAM/CSAE) pre-ingest blocking and adult/NSFW namespace isolation. All 929,400 lookup keys in [`validated_zero_cue_catalog.json`](validated_zero_cue_catalog.json) are stored exclusively as irreversible domain-separated 64-bit `BLAKE2b` digests (`digest_size=8`, `person=b"ArmigZCC"`).

- **License:** [Apache License, Version 2.0](LICENSE) · [`NOTICE`](NOTICE) · [`LEGAL.md`](LEGAL.md) · [`SECURITY.md`](SECURITY.md)
