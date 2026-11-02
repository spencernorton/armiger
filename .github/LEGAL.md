# Legal, Copyright, and Acceptable Use Policy

This document outlines the legal, copyright, privacy, and content-safety obligations governing the **Armiger** and **Armiger Ultra** models, plugins, and SDK (`spencernorton/armiger`).

---

## 1. Purpose and Intended Use

Armiger is a metadata-only structural and linguistic classification engine designed for:
- **Index hygiene and taxonomy classification:** Categorizing BitTorrent and decentralized storage metadata (filenames, file sizes, directory structures, and extension distributions) across 11 content categories.
- **Metadata screening:** Flagging suspicious filenames, extension combinations and supplied swarm signals before a grab. These are metadata rules/predictions, not scans of executable or media contents.
- **Pre-ingest policy controls:** Applying metadata/blocklist checks, accepted cryptographic tombstone revocations and model thresholds. Passing these controls does not establish content safety, legality or error-free adult-content isolation.

Neural classification does not download or inspect media payloads. Connectors can work with index metadata and torrent/magnet links supplied by upstream services; the model-update subsystem can transfer model bundles and their `.torrent` metadata. Those functions do not establish that any particular upstream content or use is permitted.

---

## 2. Copyright & Intellectual Property Compliance

1. **Model and catalogue representation:**
   - Model weights and lookup tables are not media payloads. That distinction is not a blanket copyright or privacy clearance for all model, SDK or reference-data assets.
   - The packaged catalogue lookup uses 64-bit BLAKE2b digests. The default [`hash_catalog_key`](../src/armiger/zero_cue_plugin.py) uses personalization `person=b"armiger-zc-v2"` without a secret key unless a caller supplies one. Someone with candidate input strings can compute and compare their digests. This is not a zero-knowledge proof or a guarantee that titles cannot be inferred. Do not assume every SDK/reference asset or feature rule contains only hashed strings.
2. **Creative Commons & Public-Domain Test Fixtures:**
   - The documented open-media examples and their attributions are listed in [NOTICE](../NOTICE). That list is not a certification of the rights status of every possible input, fixture or upstream dataset; preserve the applicable attribution and source terms.
3. **No Circumvention or Infringement:**
   - Armiger does not bypass technological protection measures (DRM), access private trackers, or facilitate copyright infringement. Users are solely responsible for ensuring that their deployment of Armiger complies with applicable copyright laws in their jurisdiction (including 17 U.S.C. § 512 and EU Directive 2019/790).

---

## 3. Privacy controls and limits

1. **Private-Tracker Exclusion (`swarm_priors.py`):**
   - Recognized private-tracker, passkey and credential patterns are subject to filtering. Pattern matching does not establish detection of every sensitive string or private source.
2. **Optional telemetry (`armiger_dht_rsi.py` & `armiger_dht_db_sync.py`):**
   - Telemetry schema checks, filtering, coarse timestamps and noise reduce exported detail; sharing is off by default. These controls do not guarantee zero personal information, hide peer network addresses or establish end-to-end anonymity. Shared unkeyed catalogue hashes can permit candidate-title membership tests. See [the documented export paths and privacy limits](../docs/distributed-updates.md).

---

## 4. Child Safety & Zero-Tolerance CSAM Policy

1. **Mandatory metadata pre-ingest checks (`csam_guard.py`):**
   - `CSAMGuard` performs multi-stage Unicode NFKD decomposition, zero-width character stripping, Cyrillic/Greek homoglyph normalization, and leetspeak deobfuscation before applying metadata rules. The project's zero-tolerance policy is not a claim that metadata checks can identify all prohibited payloads. Do not use a passed flag as evidence that downloaded contents are safe or lawful.
2. **Cryptographic Tombstone Revocation:**
   - `TombstoneStore` applies accepted signed revocations with monotonic sequence checks locally. A node must receive and accept a revocation before enforcing it; this does not guarantee immediate propagation or permanent blocking across every participating/offline node.

---

## 5. Notice and Takedown / Contact Procedure

If you believe any material in this repository infringes your copyright or trademark rights, or if you wish to report a safety or compliance concern, please contact:

- **Security & Compliance Advisories:** <https://github.com/spencernorton/armiger/security/advisories/new>
- **Repository Owner:** [@spencernorton](https://github.com/spencernorton)

Please include:
1. Identification of the specific file, line, or artifact in question;
2. A description of the copyrighted work or right claimed to be infringed;
3. Your contact information and a statement of good-faith belief.
