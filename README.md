<h1 align="center">Armiger Ultra</h1>

<p align="center">
  <a href="https://norvitech.com"><img alt="NorviTech Suite" src="https://img.shields.io/badge/NorviTech-Suite-FD8024.svg"></a>
  <a href="https://huggingface.co/NorviTech/armiger-ultra"><img alt="Hugging Face Models" src="https://img.shields.io/badge/Hugging_Face-NorviTech%2Farmiger--ultra-FFD21E.svg"></a>
  <a href="https://huggingface.co/NorviTech/armiger"><img alt="Armiger CPU" src="https://img.shields.io/badge/Armiger-15.7_MB_CPU-0284C7.svg"></a>
  <a href="LICENSE"><img alt="Licence Apache-2.0" src="https://img.shields.io/badge/licence-Apache--2.0-blue.svg"></a>
</p>

**Version:** `v3.0.0` · **Weights:** `32.5 MB` FP32 ONNX / PyTorch + `15.7 MB` INT8 ONNX · **Parameters:** `8,112,098` · **License:** Apache-2.0

**Armiger Ultra v3** is a 9-head byte-level RoPE Transformer (`d=256`, `6` layers) with a file-tree encoder and a 6-plugin pipeline for raw BitTorrent/DHT metadata (`name`, file-tree `paths` + `sizes`, optional tracker/DHT counters). On Apple M4 Pro CPU with ONNX Runtime it runs at **2.51 ms/item** single-item INT8 (`396.9 items/s` batch-64; FP32 `2.64 ms` / `376.0 items/s`). All 6 plugins ship in this repo via `ArmigerUltraPipeline`.

## Architecture & Bundled Capabilities

- **9 simultaneous heads (`armiger_ultra.fp32.onnx` & `armiger.int8.onnx`):** `category` (11-class), `adult`, `junk`/malware, `anime`, `english_subs`, spoken `lang`, `title_language` (`english_title` vs `non_english_title`), `owner_scope` policy gate, and per-byte `span_bio` (`O`, `B-TITLE`, `I-TITLE`, `B-YEAR`, `I-YEAR`, `B-EP`, `I-EP`) via a constrained 7-tag Viterbi mask (`0` illegal transitions).
- **Bundled 6-plugin suite (`ArmigerUltraPipeline`):**
  1. **SwarmSurvivalNet (`swarm_survival.py` + `swarm_adapter.pt`):** 48-tensor residual adapter for swarm liveliness (`92.42%` AUROC on Public Tracker `N=757`; `96.83%` AUROC on Multi-Source `N=10,278`). See [`docs/plugins/swarm-survival.md`](docs/plugins/swarm-survival.md).
  2. **SwarmPriors (`swarm_priors.py`):** 4-pathway Empirical Bayes tracker/DHT peer-seeder priors (`86.64%` formula AUROC) with passkey sanitization. See [`docs/plugins/swarm-priors.md`](docs/plugins/swarm-priors.md).
  3. **ZeroCueCatalog (`zero_cue_plugin.py` + `validated_zero_cue_catalog.json`):** 931,142-entry BLAKE2b-64 offline catalog (`26.0 MB`) for zero-cue disambiguation (`+5.45 pp` on `zero_cue_2000`, `+1.46 pp` on `beyond_confirmation`). See [`docs/plugins/zero-cue-catalog.md`](docs/plugins/zero-cue-catalog.md).
  4. **AudioLanguageGuard (`audio_language_guard.py`):** Audio/subtitle-trap guard (`96.70%` strict English-vs-Foreign on `oct15_confirmation` `N=545`; `97.82%` on `beyond_confirmation` `N=229`). See [`docs/plugins/audio-guard.md`](docs/plugins/audio-guard.md).
  5. **CalibrationAndDHT (`calibration_and_dht.py` + `calibration.json`):** Guarded per-head temperature scaling and Mondrian conformal gates (`adult >= 0.5451`, `junk >= 0.1010`, `owner_scope >= 0.4770`). See [`docs/plugins/calibration-dht.md`](docs/plugins/calibration-dht.md).
  6. **ArmigerDHTDBSync (`armiger_dht_db_sync.py` + `csam_guard.py`):** Opt-in BEP-44/46 DHT sync and pre-ingestion CSAM filter (`opt_in_sharing=False` by default). See [`docs/plugins/dht-db-sync.md`](docs/plugins/dht-db-sync.md).

## Benchmark Verification

![Armiger vs Cloud Teacher LLMs](assets/armiger_vs_llms_overall_and_6dims.png)

![Armiger vs Open-Source Torrent Classifiers Across 7 Core Dimensions](assets/armiger_vs_torrent_classifiers_6dims.png)

- **Cloud teacher comparison (identical cohorts, non-circular LOTO):** **`95.66%`** 6-dimension mean on the `N=757` Public-Tracker swarm cohort (**`96.39%`** with `N=10,278` Multi-Source Corpus Swarm AUROC). On common-support leave-one-teacher-out category panels, `OpenAI GPT (Fast Tier)` reaches `97.84%` vs. Armiger Ultra `94.56%` (`N=974`) and `Google Gemini Flash` reaches `97.05%` vs. Armiger Ultra `94.70%` (`N=982`); Armiger Ultra leads the 4-dimension LOTO composite against GPT (`95.74%` vs. `95.17%`) and trails Gemini (`95.83%` vs. `97.32%`).
- **Fresh non-teacher baselines (identical inputs, no label-curation role):** on the `beyond_confirmation` labelled subset (category `N=967`), the v3.0.0 hybrid reaches **`92.76%`** vs. **`92.45%`** for `Claude Haiku 4.5` and **`47.67%`** for the locally served 8B-class `Qwen3-8B` (see [`benchmarks/receipts/llm_beyond_fresh_receipt.json`](benchmarks/receipts/llm_beyond_fresh_receipt.json)).
- **Open-source classifiers (identical `N=2,835` denominator):** **`94.85%`** 11-class accuracy on `oct15_confirmation` (`95.83%` Movie-vs-TV, `95.06%` Catalog Matching, `96.70%` Strict Audio Language), compared to `72.24%` for `MIMMO BERT 360M`, `53.58%` for `RTN`/`PTT`, and `38.69%` for `GuessIt`. Title-only parsers (`RTN`, `GuessIt`, `PTN`, `Anitomy`) are video/anime-focused rather than 11-class general classifiers.
- **Audited splits (`N=10,262` unmutated ground-truth keys; standalone neural → fair hybrid with ZeroCueCatalog):** `oct15_confirmation` (`N=2,835`) `94.46%` → **`94.85%`** FP32; `fresh_confirmation` (`N=1,402`) `92.30%` → **`92.58%`**; `beyond_confirmation` (`N=1,025`) `91.61%` → **`93.07%`**; `zero_cue_2000` (`N=2,000`) `85.55%` → **`91.00%`**; `multilang_3000` (`N=3,000`) `97.40%` → **`98.33%`**; 5-split weighted mean `93.00%` → **`94.63%`** FP32 (`92.94%` → **`94.56%`** INT8).
- **Locked Gold R5-Test per-head results:** **`98.83%`** Title Language (`N=511`), **`98.18%`** English Subtitles (`N=110`; thin support, preliminary), **`98.09%`** Owner Scope (`N=996`), and Title Dice F1 **`0.8730`** with `0` illegal BIO transitions (`N=996`). Safety gates: **`99.23%`** balanced junk/sample accuracy on `beyond_confirmation` (`N=977`) and **`98.49%`** adult/NSFW isolation (`N=2,781`). Per-head detail is in [`docs/base-model.md`](docs/base-model.md).
- **Provenance:** every chart in [`assets/`](assets) (including the cost and per-category figures) is deterministically generated from [`benchmarks/canonical_eval_vectors.json`](benchmarks/canonical_eval_vectors.json) by [`benchmarks/generate_canonical_charts.py`](benchmarks/generate_canonical_charts.py) and verified by [`benchmarks/CHARTS_MANIFEST.json`](benchmarks/CHARTS_MANIFEST.json) and [`benchmarks/test_chart_integrity.py`](benchmarks/test_chart_integrity.py) (`10,000` paired bootstrap resamples, `seed=20261018`, exact McNemar tests, identical-denominator and LOTO anti-circularity locks). See [`docs/CHART_PIPELINE.md`](docs/CHART_PIPELINE.md). Artifact hashes are pinned in [`config.json`](config.json).

## Quickstart (Unified 6-Plugin Pipeline)

Requires `numpy`, `onnxruntime`, `cryptography`, and `torch` (CPU is sufficient). The standalone path below needs only `numpy` + `onnxruntime`.

```python
from armiger_ultra_pipeline import ArmigerUltraPipeline

pipeline = ArmigerUltraPipeline.from_directory(".", variant="ultra")
res = pipeline.classify({
    "name": "Tears.of.Steel.2012.2160p.UHD.BluRay.REMUX.HEVC.TrueHD.7.1.Atmos-OPEN",
    "size": 68_400_000_000,
    "files": [{"path": "Tears.of.Steel.2012.mkv", "size": 68_400_000_000}],
    "swarm": {"tracker_known": True, "tracker_seeders": 142, "tracker_leechers": 19, "tracker_completed": 810},
})
print(res["category"], res["mondrian_gates"]["passed_all"], res["swarm_viability"]["viability_tier"])
```

## Quickstart (Standalone CPU ONNX Runtime)

```python
from armiger_inference import ArmigerStandaloneONNX

model = ArmigerStandaloneONNX()  # auto-loads armiger.int8.onnx, vocab.json, calibration.json
res = model.predict({
    "name": "Tears.of.Steel.2012.1080p.BluRay.x264-OPEN",
    "size": 8_500_000_000,
    "files": [{"path": "Tears.of.Steel.2012.mkv", "size": 8_500_000_000}],
})
print(res["heads"]["category"], res["matching_identity"], res["mondrian_gates"]["passed_all"])
```

Run the bundled verification suite from the project environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy onnxruntime cryptography pytest
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pytest -q
```

## Connectors & Integrations

- **Turnkey indexer, Torznab & `*Arr` connectors** ([`spencernorton/Armiger-Connectors`](https://github.com/spencernorton/Armiger-Connectors) · [`examples/`](examples/README.md)):
  - **Bitmagnet + Armiger Ultra turnkey stack (`bitmagnet-io/bitmagnet`):** [`examples/bitmagnet/docker-compose.yml`](examples/bitmagnet/docker-compose.yml) · [`bitmagnet/bitmagnet_connector.py`](https://github.com/spencernorton/Armiger-Connectors/blob/main/bitmagnet/bitmagnet_connector.py) — auto-bootstraps `NorviTech/armiger-ultra`, polls untagged torrents via GraphQL, prunes `.exe`/`.scr`/`.lnk` malware and honeypot traps, writes `armiger-*` kebab-case tags, and serves an HTTP bridge (`:8096`) plus a built-in Torznab pre-grab filter proxy (`:8097`).
  - **Stock Rats Search (`librats/rats-search`):** [`examples/rats-search/docker-compose.yml`](examples/rats-search/docker-compose.yml) · [`rats-search/rats_search_connector.py`](https://github.com/spencernorton/Armiger-Connectors/blob/main/rats-search/rats_search_connector.py) — drop-in REST/WebSocket/Manticore `mysql41` sidecar classifier and safety filter.
  - **DH-Waypoint `processorapi/v1` plugin** ([`spencernorton/DH-Waypoint`](https://github.com/spencernorton/DH-Waypoint)): [`examples/dh-waypoint/docker-compose.yml`](examples/dh-waypoint/docker-compose.yml) · [`dh-waypoint/armiger_waypoint_processor.py`](https://github.com/spencernorton/Armiger-Connectors/blob/main/dh-waypoint/armiger_waypoint_processor.py) — admission (`/v1/admit`), batch (`/v1/classify`), single-job (`/v1/process`), and 4-tier feedback (`/v1/feedback`).
  - **Torznab pre-grab proxy (`Prowlarr`/`Sonarr`/`Radarr`/`Lidarr`/`Readarr`):** [`examples/arr-stack/docker-compose.yml`](examples/arr-stack/docker-compose.yml) · [`arr-stack/armiger_torznab_proxy.py`](https://github.com/spencernorton/Armiger-Connectors/blob/main/arr-stack/armiger_torznab_proxy.py) — filters malware lures, dead swarms, and unrequested foreign-audio releases from Torznab XML before grab.
  - **Post-import `*Arr`/qBittorrent webhook:** [`arr-stack/armiger_arr_webhook_receiver.py`](https://github.com/spencernorton/Armiger-Connectors/blob/main/arr-stack/armiger_arr_webhook_receiver.py) — `ffprobe` ground truth for local conformal calibration and opt-in anonymous BEP-44/46 updates.

## Hugging Face Model Hub

- **Full FP32 + INT8 + 6-plugin suite:** [`NorviTech/armiger-ultra`](https://huggingface.co/NorviTech/armiger-ultra)
- **Lightweight CPU model (`15.7 MB` INT8 only):** [`NorviTech/armiger`](https://huggingface.co/NorviTech/armiger)
- **Organization hub and components:** [`NorviTech`](https://huggingface.co/NorviTech) · [`armiger-swarm-survival`](https://huggingface.co/NorviTech/armiger-swarm-survival) · [`armiger-swarm-priors`](https://huggingface.co/NorviTech/armiger-swarm-priors) · [`armiger-zero-cue-catalog`](https://huggingface.co/NorviTech/armiger-zero-cue-catalog) · [`armiger-audio-guard`](https://huggingface.co/NorviTech/armiger-audio-guard) · [`armiger-calibration-dht`](https://huggingface.co/NorviTech/armiger-calibration-dht) · [`armiger-dht-db-sync`](https://huggingface.co/NorviTech/armiger-dht-db-sync)

## Documentation

- **Base ONNX model guide (tensor contract, per-head metrics):** [`docs/base-model.md`](docs/base-model.md)
- **6 official plugin guides:** [`docs/plugins/`](docs/plugins)
- **Canonical chart and benchmark pipeline:** [`docs/CHART_PIPELINE.md`](docs/CHART_PIPELINE.md)
- **Anonymous opt-in DHT specification:** [`docs/ANONYMOUS_OPT_IN_DHT.md`](docs/ANONYMOUS_OPT_IN_DHT.md)
- **Machine-readable receipts (hardware, fresh LLM runs, DHT seq=3006 bundle):** [`benchmarks/receipts/`](benchmarks/receipts)

## Intended Use, Copyright & Legal Compliance

`Armiger`, `Armiger Ultra`, and all bundled plugins are open-source machine learning metadata-classification utilities designed strictly for **lawful use cases**, including organizing licensed, public-domain, Creative Commons, and open-source digital media, automated disguised-executable screening (`.mkv.exe`, `.scr`, `.vbs`), and child-safety (CSAM/CSAE) pre-ingest blocking and adult/NSFW namespace isolation. All 931,142 lookup keys in [`validated_zero_cue_catalog.json`](validated_zero_cue_catalog.json) are stored exclusively as irreversible domain-separated 64-bit `BLAKE2b` digests (`digest_size=8`, `person=b"ArmigZCC"`).

- **License:** [Apache License, Version 2.0](LICENSE) · [`NOTICE`](NOTICE) · [`LEGAL.md`](LEGAL.md) · [`SECURITY.md`](SECURITY.md)
