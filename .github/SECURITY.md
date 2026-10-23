# Security Policy

## Supported Versions

Security fixes land on `main` and in the next library release.

| Version | Supported |
| --- | --- |
| 3.1.x library / 3.0.0 model | Yes |
| Older releases | No |

## Reporting a Vulnerability

Please **do not** open a public issue for suspected vulnerabilities in:
- `csam_guard.py` (pre-ingest content safety normalization, homoglyph/leetspeak deobfuscation, or tombstone verification),
- `armiger_dht_db_sync.py` (X25519 / ML-KEM-768 HKDF derivation, ChaCha20-Poly1305 AEAD shard encryption, Ed25519 / WOTS+-SHAKE256 quorum signature verification, or Merkle inclusion proofs),
- `armiger_dht_rsi.py` (`TelemetryAnonymizer` PII stripping or Ed25519 receipt verification),
- `swarm_priors.py` / `swarm_survival.py` (disguised executable or malware honeypot detection).

Report privately via one of:

1. **GitHub Private Vulnerability Reporting** on this repository:
   <https://github.com/spencernorton/armiger/security/advisories/new>

Include a minimal reproducer (synthetic metadata payload, shard manifest, or file-tree dictionary) and the affected version or commit hash. You can expect an initial response within 7 days.
