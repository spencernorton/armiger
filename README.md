<h1 align="center">Armiger Ultra</h1>

<p align="center">
  <strong>Compact neural torrent &amp; media classification suite bundling Armiger (15.7 MB INT8 CPU), Armiger Ultra (32.5 MB FP32 GPU/CPU), the 48-tensor Swarm Viability Adapter (<code>swarm_adapter.pt</code>), and 6 modular plugins.</strong><br>
  Delivers 95.66% 6-dimension composite accuracy on the Public-Tracker cohort (96.39% with Multi-Source Corpus Swarm AUROC) on unmutated ground-truth keys, 2.51 ms single-item and 396.9 items/s batch-64 on Apple M4 Pro CPU INT8 (2.64 ms / 376.0 items/s FP32), 99.23% balanced junk/sample accuracy (<code>N=977</code>), and 96.83% Multi-Source Corpus / 92.42% Public-Tracker swarm liveliness AUROC.
</p>

<p align="center">
  <a href="https://norvitech.com"><img alt="NorviTech Suite" src="https://img.shields.io/badge/NorviTech-Suite-FD8024.svg"></a>
  <a href="https://huggingface.co/NorviTech/armiger-ultra"><img alt="Hugging Face Models" src="https://img.shields.io/badge/Hugging_Face-NorviTech%2Farmiger--ultra-FFD21E.svg"></a>
  <a href="https://huggingface.co/NorviTech/armiger"><img alt="Armiger CPU" src="https://img.shields.io/badge/Armiger-15.7_MB_CPU-0284C7.svg"></a>
  <a href="LICENSE"><img alt="Licence Apache-2.0" src="https://img.shields.io/badge/licence-Apache--2.0-blue.svg"></a>
</p>

---

## Benchmarks

### Cloud Teacher LLM Comparison

![Armiger vs Cloud Teacher LLMs](assets/armiger_vs_llms_overall_and_6dims.png)

![Cost Benchmark vs Cloud Teacher LLMs](assets/armiger_vs_llms_cost_comparison.png)

- **Competitive with cloud teacher LLMs on identical evaluation cohorts.** Achieves **`95.66%`** 6-dimension mean accuracy on the `N=757` Public-Tracker swarm cohort (**`96.39%`** with `N=10,278` Multi-Source Corpus Swarm AUROC). On common-support leave-one-teacher-out (LOTO) 11-class category panels, `OpenAI GPT (Fast Tier)` reaches `97.84%` vs. Armiger Ultra `94.56%` (`N=974`), and `Google Gemini Flash` reaches `97.05%` vs. Armiger Ultra `94.70%` (`N=982`); Armiger Ultra leads the 4-dimension LOTO composite against GPT (`95.74%` vs. `95.17%`) and trails Gemini (`95.83%` vs. `97.32%`). Fresh non-teacher LLM runs for this release are reported separately from these canonical teacher panels so denominators and circularity remain explicit.
- **Fresh non-teacher LLM baselines (identical inputs, no label curation role).** On the `beyond_confirmation` labelled subset (`982` input records from the `N=1,025` split; category scored at `N=967`), Armiger Ultra v3.0.0 hybrid reaches **`92.76%`** 11-class category accuracy vs. **`92.45%`** for `Claude Haiku 4.5` and **`47.67%`** for the locally served 8B-class `Qwen3-8B` (adult gate `98.63%` / `94.22%` at `N=952`; junk gate `98.03%` / `99.56%` at `N=913`). The same-subset Pre-Gold hybrid scores `94.62%`, consistent with the production-split tradeoff disclosed for the Gold line (see [`benchmarks/receipts/llm_beyond_fresh_receipt.json`](benchmarks/receipts/llm_beyond_fresh_receipt.json)).
- **Measured local inference at zero API cost.** On Apple M4 Pro CPU with ONNX Runtime (8 intra-op threads, all 9 outputs), the `15.7 MB` INT8 model runs at **`2.51 ms` single-item** and **`396.9 items/s` batch-64**; FP32 runs at `2.64 ms` single-item and `376.0 items/s` batch-64 (see `benchmarks/receipts/hardware_and_api_cost_receipt.json`).

### Open-Source Classifiers & 10-Category Breakdown

![Armiger vs Open-Source Torrent Classifiers Across 7 Core Dimensions](assets/armiger_vs_torrent_classifiers_6dims.png)

![Per-Category Recall Breakdown Across All 10 Media & Software Categories](assets/armiger_categorical_breakdown.png)

- **Universal 11-class coverage vs. title-only parsers.** Reaches **`94.85%`** 11-class accuracy (`95.83%` Movie-vs-TV, `95.06%` Catalog Matching, `96.70%` Strict Audio Language) on `oct15_confirmation` (`N=2,835`) and **`94.63%`** weighted mean across all 5 evaluation splits (`N=10,262`), compared to `72.24%` for `MIMMO BERT 360M` (`~1.4 GB`), `53.58%` for `RTN`/`PTT`, and `38.69%` for `GuessIt` on the identical `N=2,835` denominator (note that `RTN`, `GuessIt`, `PTN`, and `Anitomy` are video/anime-focused title parsers rather than 11-class general classifiers).
- **Calibrated safety & junk gates.** Mondrian conformal thresholds achieve **`99.23%`** balanced junk/sample accuracy on `beyond_confirmation` (`N=977`), deterministic filtering of disguised executable extensions (`.mkv.exe`, `.scr`, `.lnk`, `.vbs`), and **`98.49%`** adult/NSFW isolation (`N=2,781`).
- **Reproducible chart & provenance pipeline.** Every chart in [`assets/`](assets) is deterministically generated from [`benchmarks/canonical_eval_vectors.json`](benchmarks/canonical_eval_vectors.json) by [`benchmarks/generate_canonical_charts.py`](benchmarks/generate_canonical_charts.py) and verified by [`benchmarks/CHARTS_MANIFEST.json`](benchmarks/CHARTS_MANIFEST.json) and [`benchmarks/test_chart_integrity.py`](benchmarks/test_chart_integrity.py) (`10,000` paired bootstrap resamples with `seed=20261018`, exact two-sided McNemar $p$-values, identical-denominator lock, and LOTO anti-circularity lock). See [`docs/CHART_PIPELINE.md`](docs/CHART_PIPELINE.md).

---

## Quickstart

`armiger_ultra_pipeline.py` integrates the zero-tokenizer 9-head ONNX/PyTorch classifier (`armiger_inference.py`, including byte-level `span_bio` title/year/episode spans) with all 6 plugins behind a single Python API:

```python
from armiger_ultra_pipeline import ArmigerUltraPipeline

# Load full-precision Armiger Ultra (or pass variant="int8" for the 15.7 MB CPU fast-path)
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

- **`armiger_ultra.fp32.onnx` (`32.5 MB`) & `armiger_ultra.pt` (`32.5 MB`)** — Full-precision 9-head byte-level RoPE Transformer + File-Tree Encoder, including the byte-level `span_bio` head with 7×7 BIO Viterbi decoding (measured `2.64 ms` single-item / `376.0 items/s` batch-64 on Apple M4 Pro CPU ONNX Runtime).
- **`armiger.int8.onnx` (`15.7 MB`)** — Per-channel asymmetric dynamically quantized CPU fast-path engine from the same checkpoint (`2.51 ms` single-item / `396.9 items/s` batch-64 on Apple M4 Pro CPU ONNX Runtime). Embeddings remain FP32 because quantizing them fails the ≥99.0% FP32-vs-INT8 head-agreement gate. See [`docs/base-model.md`](docs/base-model.md).
- **[SwarmSurvivalNet v2 Adapter](https://huggingface.co/NorviTech/armiger-swarm-survival)** (`swarm_survival.py`, `swarm_adapter.pt` · `167 KB`) — 48-tensor residual neural pass combining tracker/DHT seeder-leecher telemetry with content embeddings to reach **`96.83%`** Multi-Source Corpus AUROC (`N=10,278`, vs. `53.02%` raw seeders) and **`92.42%`** Public-Tracker AUROC (`N=757`) while keeping `category` predictions independent of seeder counts. See [`docs/plugins/swarm-survival.md`](docs/plugins/swarm-survival.md).
- **[Empirical Bayes Swarm Priors](https://huggingface.co/NorviTech/armiger-swarm-priors)** (`swarm_priors.py` · `29 KB`) — 4-pathway tracker scrape priors (`86.64%` formula AUROC), anomaly features, and passkey sanitization. See [`docs/plugins/swarm-priors.md`](docs/plugins/swarm-priors.md).
- **[Zero-Cue Catalog & Structural Disambiguator](https://huggingface.co/NorviTech/armiger-zero-cue-catalog)** (`zero_cue_plugin.py`, `validated_zero_cue_catalog.json` · `26.0 MB`) — `931,142`-entry `BLAKE2b-64` hashed catalog + structural rules, lifting `zero_cue_2000` (`N=2,000`) accuracy by **`+5.45 pp`** (`85.55%` → **`91.00%`**) and `beyond_confirmation` (`N=1,025`) by **`+1.46 pp`** (`91.61%` → **`93.07%`**). See [`docs/plugins/zero-cue-catalog.md`](docs/plugins/zero-cue-catalog.md).
- **[Cross-Head Audio Language & Anime Dub Guard](https://huggingface.co/NorviTech/armiger-audio-guard)** (`audio_language_guard.py` · `3.3 KB`) — Couples `title_language` with the `lang` head to disambiguate `.en.` prepositions, `<Foreign>.ENG.Sub` releases, regional dubs, and `MULTi` tags, reaching **`96.70%`** strict English-vs-Foreign audio accuracy on `oct15_confirmation` (`N=545`) and **`97.82%`** on `beyond_confirmation` (`N=229`). See [`docs/plugins/audio-guard.md`](docs/plugins/audio-guard.md).
- **[Conformal Calibration & Optional DHT Sync Plugins](https://huggingface.co/NorviTech/armiger-calibration-dht)** (`calibration.json`, `calibration_and_dht.py`, `armiger_dht_rsi.py`, `armiger_dht_db_sync.py`, `csam_guard.py`) — Guarded per-head temperature scaling, Mondrian conformal thresholds, multi-stage CSAM/abuse pre-ingest blocking, and strictly opt-in (`opt_in_sharing=False` by default) experimental `BEP-44`/`BEP-45`/`BEP-46` catalog, calibration, and signed model-delta sync behind `AnonymityGuard`. The v3.0.0 signed seq=3006 bundle and receipt are in [`benchmarks/receipts/dht_seq3006/`](benchmarks/receipts/dht_seq3006). See [`docs/plugins/calibration-dht.md`](docs/plugins/calibration-dht.md), [`docs/plugins/dht-db-sync.md`](docs/plugins/dht-db-sync.md), and [`docs/ANONYMOUS_OPT_IN_DHT.md`](docs/ANONYMOUS_OPT_IN_DHT.md).

---

## Separate Optional Addition: Ecosystem Connectors (`spencernorton/Armiger-Connectors`)

External HTTP, GraphQL, and Manticore connectors are maintained as a **separate repository** — **[`spencernorton/Armiger-Connectors`](https://github.com/spencernorton/Armiger-Connectors)** — distinct from the core classifier and the 6 official plugins above:

- **[Stock `bitmagnet` Connector (`bitmagnet/`)](https://github.com/spencernorton/Armiger-Connectors/tree/main/bitmagnet)** — Connects to unmodified [`bitmagnet-io/bitmagnet`](https://github.com/bitmagnet-io/bitmagnet) via GraphQL (`torrentContent.search` + `torrent.files`), pruning disguised executables (`.mkv.exe`, `.scr`, `.lnk`) via `torrent.delete` and enriching surviving items with grouped `torrent.putTags` mutations.
- **[Stock `rats-search` Connector (`rats-search/`)](https://github.com/spencernorton/Armiger-Connectors/tree/main/rats-search)** — Connects to unmodified [`librats/rats-search`](https://github.com/librats/rats-search) via REST (`:8095`), WebSocket (`:8096`), and local Manticore (`127.0.0.1:9306`), purging disguised executables via `/api/torrent.remove` and populating `contentType` and `contentCategory` (`Movie=1..Game=9`, `XXX=100`) using `rowIdFromHash` docid lookups.

---

## Audited Evaluation Splits (`N=10,262` Unmutated Ground-Truth Keys)

| Evaluation Split | Valid `N` | Standalone Neural (`FP32` / `INT8`) | Fair Hybrid (`+ ZeroCueCatalog`) | Lift |
| :--- | :---: | :---: | :---: | :---: |
| **`oct15_confirmation` (Gold Standard)** | `2,835` | `94.46%` / `94.43%` | **`94.85%`** / **`94.81%`** (`95.83%` Video) | `+0.39 pp` |
| **`fresh_confirmation` (Fresh Holdout)** | `1,402` | `92.30%` / `92.15%` | **`92.58%`** / **`92.44%`** | `+0.29 pp` |
| **`beyond_confirmation` (Hard Edge-Cases)** | `1,025` | `91.61%` / `91.61%` | **`93.07%`** / **`92.98%`** | **`+1.46 pp`** |
| **`zero_cue_2000` (Zero-Cue Specialist)** | `2,000` | `85.55%` / `85.55%` | **`91.00%`** / **`90.95%`** | **`+5.45 pp`** |
| **`multilang_3000` (Multilingual)** | `3,000` | `97.40%` / `97.27%` | **`98.33%`** / **`98.27%`** | **`+0.93 pp`** |
| **5-Split Weighted Mean** | **`10,262`** | **`93.00%`** / **`92.94%`** | **`94.63%`** / **`94.56%`** | **`+1.63 pp`** |

---

## Intended Use, Copyright & Legal Compliance

`Armiger`, `Armiger Ultra`, and all bundled plugins are open-source machine learning metadata-classification utilities designed strictly for **lawful use cases**, including organizing licensed, public-domain, Creative Commons, and open-source digital media, automated disguised-executable screening (`.mkv.exe`, `.scr`, `.vbs`), and child-safety (CSAM/CSAE) pre-ingest blocking and adult/NSFW namespace isolation. All 931,142 lookup keys in [`validated_zero_cue_catalog.json`](validated_zero_cue_catalog.json) are stored exclusively as irreversible domain-separated 64-bit `BLAKE2b` digests (`digest_size=8`, `person=b"ArmigZCC"`).

- **License:** [Apache License, Version 2.0](LICENSE) · [`NOTICE`](NOTICE) · [`LEGAL.md`](LEGAL.md) · [`SECURITY.md`](SECURITY.md)
