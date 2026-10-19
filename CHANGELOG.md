# Changelog

All notable changes to **Armiger** and **Armiger Ultra** are documented in this file.

## [2.1.0] - 2026-10-18

### Added
- **Armiger v2.1 Base Model (`armiger.int8.onnx`, 8.3 MB):**
  - 8.1M-parameter (`8,050,139` trainable parameters) multi-modal byte-level RoPE Transformer + File-Tree Hierarchical Attention + 36-feature Structural MLP architecture with 8 simultaneous classification heads (`category`, `adult`, `junk`, `anime`, `english_subs`, `lang`, `owner_scope`, `title_language`).
  - Fast CPU inference (`2.44 ms` single-item p50, `3.07 ms/item` batch-64) via dynamic INT8 ONNX Runtime quantization.
- **Armiger Ultra v2.1 (`armiger_ultra.fp32.onnx`, `armiger_ultra.pt`, 32.3 MB):**
  - Full FP32 ONNX and PyTorch weights integrated with `ArmigerUltraPipeline` (`armiger_ultra_pipeline.py`) exposing 11 unified output heads (`8` neural heads + `swarm_viability`, `honeypot_trap`, and `csam_safety`).
- **6 Official Production Plugins:**
  1. **Swarm Survival & Honeypot Defense (`swarm_survival.py`, `swarm_adapter.pt`):** 48-tensor residual neural adapter (`SwarmResidualAdapter`), Beta-Binomial Empirical Bayes release-group reliability, and 5-feature malware/fake-seeder honeypot guard with a `0.00%` content-category flip guarantee.
  2. **Tracker Scrape & Bayesian Outcome Priors (`swarm_priors.py`):** Four causally separated pathways (`Way 1` active sampling weights, `Way 2` 14-d public tracker & release-group priors with modality dropout, `Way 3` 10-d disguised executable & `.zip`/`.lnk` trap detector, and `Way 4` stage-2 grab viability ranking).
  3. **Zero-Cue Catalog & Structural Disambiguator (`zero_cue_plugin.py`, `validated_zero_cue_catalog.json`):** Domain-separated 64-bit BLAKE2b (`person=b"ArmigZCC"`, `digest_size=8`) one-way hash lookup catalog covering 929,400 entries (`472,017` title+year, `257,016` anime, `200,367` unambiguous title-only) with zero plaintext titles redistributed.
  4. **Cross-Head Audio Language & Anime Dub Guard (`audio_language_guard.py`):** Joint `title_language` x `lang` x `anime` disambiguation resolving lowercase Romance `.en.` prepositions, `<Foreign>.ENG.Sub` false positives, regional dubs, and `[Dual-Audio]` releases.
  5. **Mondrian Conformal Safety Gates & DHT RSI Engine (`calibration_and_dht.py`, `armiger_dht_rsi.py`, `calibration.json`):** Per-head temperature-calibrated Mondrian conformal safety gates and opt-in, differential-privacy blinded, hybrid post-quantum signed (`Ed25519` + `WOTS+-SHAKE256`) recursive self-improvement feedback ingestion.
  6. **Modular CSAM-Protected Encrypted BEP-44/BEP-46 DHT DB Sync (`armiger_dht_db_sync.py`, `csam_guard.py`):** Multi-stage Unicode/homoglyph/leetspeak pre-ingest CSAM & malware blocker, Ed25519-signed blinded SHA-256 tombstones, hybrid X25519 + ML-KEM-768 (FIPS 203) key encapsulation with ChaCha20-Poly1305 AEAD shard encryption, and hybrid Ed25519 + WOTS+-SHAKE256 K-of-N distinct-ASN quorum verification.
- **Verification Suite:**
  - 211 automated tests (`test_armiger_ultra_pipeline.py`, `test_armiger_dht_rsi.py`, `test_dht_db_sync.py`, `test_ecosystem_integrations.py`, `test_anonymous_opt_in_dht_sharing.py`, `benchmarks/test_chart_integrity.py`) using Creative Commons and public-domain fixtures.
