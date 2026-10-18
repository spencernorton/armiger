# Legal, Copyright, and Acceptable Use Policy

This document outlines the legal, copyright, privacy, and content-safety obligations governing the **Armiger** and **Armiger Ultra** models, plugins, and SDK (`spencernorton/armiger`).

---

## 1. Purpose and Intended Use

Armiger is a metadata-only structural and linguistic classification engine designed for:
- **Index hygiene and taxonomy classification:** Categorizing BitTorrent and decentralized storage metadata (filenames, file sizes, directory structures, and extension distributions) across 11 content categories.
- **Malware, dropper, and honeypot defense:** Detecting disguised executable payloads (`.mkv.exe`, `.scr`, `.lnk`, DLL-sideloading traps, and fake-seeder honeypots) before any file transfer occurs.
- **Content safety and compliance enforcement:** Enforcing strict pre-ingest Child Sexual Abuse Material (CSAM) blocking, non-consensual abuse filtering, cryptographic tombstone revocations, and Mondrian conformal adult-content isolation.

Armiger **does not** host, index, link to, download, distribute, or provide access to any media files, `.torrent` payloads, magnet URIs, or copyrighted works.

---

## 2. Copyright & Intellectual Property Compliance

1. **Zero Copyrighted Content or Plaintext Titles:**
   - Neither the model weights (`armiger.int8.onnx`, `armiger_ultra.fp32.onnx`, `armiger_ultra.pt`, `swarm_adapter.pt`) nor the Zero-Cue Catalog (`validated_zero_cue_catalog.json`) contain or redistribute copyrighted media files or plaintext title strings.
   - All lookup keys in `validated_zero_cue_catalog.json` and all release-tag token tables in `armiger_inference.py` and `swarm_priors.py` are stored as irreversible, one-way **64-bit keyed BLAKE2b cryptographic hashes** (`digest_size=8`).
2. **Creative Commons & Public-Domain Test Fixtures:**
   - Every test fixture, code example, and documentation sample in this repository exclusively uses open-source software distributions (e.g., Ubuntu ISOs), public-domain literature (*Project Gutenberg*, *LibriVox*), Creative Commons Zero recordings (*Open Goldberg Variations*), or Creative Commons Attribution open movies and comics (*Blender Studio Open Movies*, *Pepper&Carrot*).
3. **No Circumvention or Infringement:**
   - Armiger does not bypass technological protection measures (DRM), access private trackers, or facilitate copyright infringement. Users are solely responsible for ensuring that their deployment of Armiger complies with applicable copyright laws in their jurisdiction (including 17 U.S.C. § 512 and EU Directive 2019/790).

---

## 3. Privacy & Anonymization Invariants

1. **Private-Tracker Exclusion (`swarm_priors.py`):**
   - Any announce URL, source tag, or metadata string containing passkeys, authkeys, user tokens, or private-tracker identifiers is automatically rejected and stripped before feature extraction or telemetry processing.
2. **Zero-PII Telemetry (`armiger_dht_rsi.py` & `armiger_dht_db_sync.py`):**
   - Local filesystem paths, usernames, IPv4/IPv6 addresses, MAC addresses, and indexer credentials are stripped by `TelemetryAnonymizer` prior to any optional DHT synchronization.

---

## 4. Child Safety & Zero-Tolerance CSAM Policy

1. **Mandatory Pre-Ingest Blocking (`csam_guard.py`):**
   - `CSAMGuard` performs multi-stage Unicode NFKD decomposition, zero-width character stripping, Cyrillic/Greek homoglyph normalization, and leetspeak deobfuscation before checking against zero-tolerance CSAM, underage exploitation, non-consensual abuse, and extreme-violence rules.
2. **Cryptographic Tombstone Revocation:**
   - `TombstoneStore` enforces Ed25519-signed blinded SHA-256 tombstone lists with monotonic sequence numbers, immediately purging and permanently blocking any revoked hash across participating nodes.

---

## 5. Notice and Takedown / Contact Procedure

If you believe any material in this repository infringes your copyright or trademark rights, or if you wish to report a safety or compliance concern, please contact:

- **Security & Compliance Advisories:** <https://github.com/spencernorton/armiger/security/advisories/new>
- **Repository Owner:** [@spencernorton](https://github.com/spencernorton)

Please include:
1. Identification of the specific file, line, or artifact in question;
2. A description of the copyrighted work or right claimed to be infringed;
3. Your contact information and a statement of good-faith belief.
