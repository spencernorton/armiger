# Armiger v2.1 Base Model (`armiger.int8.onnx`)

**Armiger v2.1** (`armiger.int8.onnx`, 8.3 MB) is the standalone INT8 CPU-quantized neural classifier at the core of the Armiger suite. It processes raw torrent metadata—release name, total size, file paths, per-file byte sizes, and file extensions—and predicts **8 simultaneous classification heads** in **~0.8 ms on a single CPU core** with zero external service calls.

---

## Neural Architecture (6.1M Parameters)

1. **Release-Name Token BiGRU (`name_tokens`):**
   - Character- and word-level sub-tokenization with digit-folded CRC32 path/title hashing (`vocab.json`, 4,096 extensions + structural tokens).
2. **File-Tree Set-Transformer (`file_exts`, `file_sizes`, `file_depths`):**
   - Permutation-invariant multi-head self-attention over up to 64 files per release, capturing sidecar subtitles (`.srt`, `.ass`), disc structures (`BDMV`, `VIDEO_TS`), multi-episode layouts, and disguised executable droppers (`.mkv.exe`, `.lnk`, `libvlc.dll`).
3. **36-Feature Structural MLP (`stats`):**
   - Exact byte-ratio features, video/audio/archive/executable/document/image/subtitle proportions, Shannon entropy of file sizes, CJK/Cyrillic script ratios, and explicit structural flags.
4. **8 Calibrated Multi-Task Heads (`T = 0.72`–`0.92`):**
   - `category` (11 classes: `movie`, `tv_show`, `xxx`, `music`, `comic`, `game`, `audiobook`, `software`, `ebook`, `course`, `unknown`)
   - `adult` (`non_adult`, `adult`)
   - `junk` (`clean`, `junk`)
   - `anime` (`non_anime`, `anime`)
   - `english_subs` (`no_eng_subs`, `has_eng_subs`, `unknown`)
   - `lang` (`english`, `non_english`, `multi_or_unknown`)
   - `owner_scope` (`allowed_type`, `excluded`)
   - `title_language` (`english_title`, `non_english_title`)

---

## Standalone Usage

```python
from armiger_inference import ArmigerStandaloneONNX

model = ArmigerStandaloneONNX("armiger.int8.onnx")
result = model.predict({
    "name": "Sintel.2010.1080p.BluRay.x264-OPEN",
    "size": 2_400_000_000,
    "files": [
        {"path": "Sintel.2010.1080p.BluRay.x264-OPEN.mkv", "size": 2_399_946_000},
        {"path": "Subs/English.srt", "size": 54_000},
    ],
})
print(result["category"], result["heads"])
```
