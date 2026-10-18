<h1 align="center">Armiger (8.3 MB INT8 CPU Engine)</h1>

<p align="center">
  <strong>Fast 8.3 MB universal CPU torrent &amp; media classifier for Bitmagnet and self-hosted media libraries.</strong><br>
  Predicts 8 calibrated decision heads simultaneously in 2.44 ms single-item p50 (3.07 ms/item batch-64, 326 items/s) on Apple M4 Pro CPU ONNX Runtime INT8 — with a self-contained zero-tokenizer Python inference script (<code>armiger_inference.py</code> + <code>vocab.json</code>) included out of the box.
</p>

<p align="center">
  <a href="https://norvitech.com"><img alt="NorviTech Suite" src="https://img.shields.io/badge/NorviTech-Suite-FD8024.svg"></a>
  <a href="https://huggingface.co/NorviTech/armiger-ultra"><img alt="Armiger Ultra" src="https://img.shields.io/badge/Armiger_Ultra-GPU_%2B_Swarm_Suite-10B981.svg"></a>
  <a href="../LICENSE"><img alt="Licence Apache-2.0" src="https://img.shields.io/badge/licence-Apache--2.0-blue.svg"></a>
</p>

---

## Benchmarks

![Armiger vs Frontier Cloud LLMs and Local GPU LLM](../assets/armiger_vs_llms_overall_and_6dims.png)

![Cost Benchmark vs Frontier Cloud LLMs and Local GPU LLM](../assets/armiger_vs_llms_cost_comparison.png)

![Armiger vs Open-Source Torrent Classifiers Across 7 Core Dimensions](../assets/armiger_vs_torrent_classifiers_6dims.png)

![Categorical Accuracy Breakdown Across All 10 Media & Software Categories](../assets/armiger_categorical_breakdown.png)

---

## Quickstart: Standalone ONNX Inference (`armiger_inference.py`)

**Armiger** (`armiger.int8.onnx`, `8.3 MB`) is a zero-tokenizer byte-level RoPE Transformer + File-Tree Encoder designed for real-time BitTorrent, DHT, Usenet, and media automation classification on any CPU:

```python
from armiger_inference import ArmigerONNXClassifier

clf = ArmigerONNXClassifier(
    onnx_path="armiger.int8.onnx",
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

## 8 Simultaneous Classification Heads

All metrics below are reproducibly computed using `armiger_inference.py` against the original unmutated ground-truth evaluation keys (`rows.y`):

- **`category` (11 classes: `movie`, `tv_show`, `xxx`, `music`, `comic`, `game`, `audiobook`, `software`, `ebook`, `course`, `unknown`)** — **`95.06%`** Standalone Neural INT8 (**`95.49%`** Fair Hybrid) on `oct15_confirmation` (`N=2,835`); **`93.95%`** INT8 / **`95.39%`** Fair Hybrid weighted mean across all 5 evaluation splits (`N=10,262`).
- **`adult` (Safety Gate: `non_adult` vs. `adult`)** — **`98.53%`** on `oct15_confirmation` (`2,740/2,781`), **`99.90%`** on `tc_truespec_2000`, and **`100.00%`** on `tc_truespec_multilang_3000`.
- **`junk` (Malware & Fake-Codec Filter: `clean` vs. `junk`)** — **`100.00%`** on `oct15_confirmation` (`2,170/2,170`) and `fresh_confirmation` (`1,291/1,291`); **`99.90%`** on `beyond_confirmation` (`976/977`).
- **`anime` (Anime & Dub Routing: `non_anime` vs. `anime`)** — **`99.61%`** on `oct15_confirmation` (`505/507`) and **`98.90%`** on `beyond_confirmation` (`629/636`).
- **`lang` (Audio/Release Language: `english`, `non_english`, `multi_or_unknown`)** — **`96.88%`** Standalone Neural / **`97.80%`** with `AudioLanguageGuard` on `oct15_confirmation` binary English-vs-Foreign (`N=545`).
- **`title_language` (Title Script/Language: `english_title` vs. `non_english_title`)** — **`93.70%`** on `oct15_confirmation` (`N=969`) and **`98.51%`** on `beyond_confirmation` (`N=604`).
- **`english_subs` (Subtitle Availability: `no_eng_subs`, `has_eng_subs`, `unknown`)** — **`93.04%`** on `oct15_confirmation` (`N=2,171`).
- **`owner_scope` (Catalog Policy Gate: `allowed_type` vs. `excluded`)** — **`98.65%`** on `oct15_confirmation` (`2,633/2,669`) and **`99.90%`** on `tc_truespec_multilang_3000`.

---

<p align="center">
  <a href="https://norvitech.com"><img alt="Part of the NorviTech Suite — open-source apps for the Linux desktop and the self-hosted stack" src="https://norvitech.com/assets/banner.svg" width="640"></a>
</p>
