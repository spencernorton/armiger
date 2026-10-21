<h1 align="center">Armiger (15.7 MB INT8 CPU Engine)</h1>

<p align="center">
  <strong>Fast 15.7 MB universal CPU torrent &amp; media classifier for Bitmagnet and self-hosted media libraries.</strong><br>
  Predicts 9 decision heads simultaneously (8 calibrated classification heads plus byte-level `span_bio`) in 2.51 ms single-item (396.9 items/s batch-64) on Apple M4 Pro CPU ONNX Runtime INT8 — with a self-contained zero-tokenizer Python inference script (<code>armiger_inference.py</code> + <code>vocab.json</code>) included out of the box.
</p>

<p align="center">
  <a href="https://norvitech.com"><img alt="NorviTech Suite" src="https://img.shields.io/badge/NorviTech-Suite-FD8024.svg"></a>
  <a href="https://huggingface.co/NorviTech/armiger-ultra"><img alt="Armiger Ultra" src="https://img.shields.io/badge/Armiger_Ultra-GPU_%2B_Swarm_Suite-10B981.svg"></a>
  <a href="../LICENSE"><img alt="Licence Apache-2.0" src="https://img.shields.io/badge/licence-Apache--2.0-blue.svg"></a>
</p>

---

## Benchmarks

![Armiger vs Cloud Teacher LLMs](../assets/armiger_vs_llms_overall_and_6dims.png)

![Cost Benchmark vs Cloud Teacher LLMs](../assets/armiger_vs_llms_cost_comparison.png)

![Armiger vs Open-Source Torrent Classifiers Across 7 Core Dimensions](../assets/armiger_vs_torrent_classifiers_6dims.png)

![Per-Category Recall Breakdown Across All 10 Media & Software Categories](../assets/armiger_categorical_breakdown.png)

Fresh non-teacher LLM baselines for this release (identical inputs, no label-curation role) are reported in [`benchmarks/receipts/llm_beyond_fresh_receipt.json`](../benchmarks/receipts/llm_beyond_fresh_receipt.json): on the `beyond_confirmation` labelled subset (category `N=967`), the v3.0.0 hybrid reaches `92.76%` 11-class category accuracy vs. `92.45%` for `Claude Haiku 4.5` and `47.67%` for the locally served 8B-class `Qwen3-8B`.

---

## Quickstart: Standalone ONNX Inference (`armiger_inference.py`)

**Armiger** (`armiger.int8.onnx`, `15.7 MB`) is a zero-tokenizer byte-level RoPE Transformer + File-Tree Encoder designed for real-time BitTorrent, DHT, Usenet, and media automation classification on any CPU:

```python
from armiger_inference import ArmigerONNXClassifier

clf = ArmigerONNXClassifier(
    onnx_path="armiger.int8.onnx",
    vocab_path="vocab.json",
    calibration_path="calibration.json",
)

pred = clf.predict({
    "name": "Sintel.2010.1080p.WEBM.AV1-OPEN",
    "size": 4_500_000_000,
    "files": [{"path": "Sintel.2010.1080p.mkv", "size": 4_500_000_000}],
})

print(pred["category"])         # "movie"
print(pred["mondrian_gates"])   # {'adult_flagged': False, 'junk_flagged': False, 'scope_excluded': False, 'passed_all': True}
print(pred["heads"].keys())     # dict_keys(['category', 'adult', 'junk', 'anime', 'english_subs', 'lang', 'owner_scope', 'title_language', 'span_bio'])
```

---

## 9 Simultaneous Classification Heads

All metrics below are reproducibly computed using `armiger_inference.py` against the original unmutated ground-truth evaluation keys (`rows.y`):

- **`category` (11 classes: `movie`, `tv_show`, `xxx`, `music`, `comic`, `game`, `audiobook`, `software`, `ebook`, `course`, `unknown`)** — **`94.43%`** Standalone Neural INT8 (**`94.81%`** Fair Hybrid) on `oct15_confirmation` (`N=2,835`); **`92.94%`** INT8 / **`94.56%`** Fair Hybrid weighted mean across all 5 evaluation splits (`N=10,262`).
- **`adult` (Safety Gate: `non_adult` vs. `adult`)** — **`98.49%`** on `oct15_confirmation` (`2,739/2,781`) and **`98.43%`** on `beyond_confirmation` (`N=1,016`); `zero_cue_2000` and `multilang_3000` contain no labeled adult positives, so only specificity is estimable there (`99.95%` / `99.83%` overall accuracy).
- **`junk` (Sample/Extra & Disguised-Executable Filter: `clean` vs. `junk`)** — **`99.23%`** balanced accuracy (**`99.90%`** overall accuracy, `976/977`, `65` positive items) on `beyond_confirmation`; positive items are sparse elsewhere (`99.95%` overall accuracy on `oct15_confirmation` `N=2,170` with 1 positive, `100.00%` on `fresh_confirmation` `N=1,291` with 3 positives).
- **`anime` (Anime & Dub Routing: `non_anime` vs. `anime`)** — **`99.61%`** on `oct15_confirmation` (`505/507`), **`98.43%`** on `beyond_confirmation` (`N=636`), and **`95.65%`** on `zero_cue_2000` (`N=1,998`).
- **`lang` (Audio/Release Language: `english`, `non_english`, `multi_or_unknown`)** — **`96.33%`** Standalone INT8 Neural / **`96.70%`** with `AudioLanguageGuard` on `oct15_confirmation` binary English-vs-Foreign (`N=545`).
- **`title_language` (Title Script/Language: `english_title` vs. `non_english_title`)** — **`98.83%`** accuracy (**`97.99%`** balanced) on the locked Gold R5-Test split (`N=511`).
- **`english_subs` (Subtitle Availability: `no_eng_subs`, `has_eng_subs`, `unknown`)** — **`98.18%`** accuracy (**`98.97%`** balanced) on the locked Gold R5-Test split (`N=110` labeled items; thin support — treat as preliminary).
- **`owner_scope` (Catalog Policy Gate: `allowed_type` vs. `excluded`)** — **`98.31%`** on `oct15_confirmation` (`2,624/2,669`, temperature-scaled neural head with Mondrian threshold `0.4770`) and **`98.09%`** on the locked Gold R5-Test split (`N=996`).
- **`span_bio` (Byte-Level Title/Year/Episode Spans: `O`, `B-TITLE`, `I-TITLE`, `B-YEAR`, `I-YEAR`, `B-EP`, `I-EP`)** — Title Dice F1 **`0.8730`** with **0 illegal BIO transitions** under Viterbi decoding on the locked Gold R5-Test split (`N=996`).

---

<p align="center">
  <a href="https://norvitech.com"><img alt="Part of the NorviTech Suite — open-source apps for the Linux desktop and the self-hosted stack" src="https://norvitech.com/assets/banner.svg" width="640"></a>
</p>
