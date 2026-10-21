# Armiger Distributed Anonymous & Opt-In-By-Default DHT Sharing

This document specifies the cryptographic, differential-privacy, and wire-level firewall architecture governing distributed weight, calibration, and catalog metadata sharing in **Armiger** and **Armiger Ultra** (`armiger_dht_rsi.py`, `armiger_dht_db_sync.py`, `armiger_ultra_pipeline.py`), along with the local unit/integration test harness (`benchmarks/proof_anonymous_opt_in_sharing.py` and `test_anonymous_opt_in_dht_sharing.py`) that verifies default opt-out behavior and wire-level PII scrubbing.

---

## 1. Strict Opt-In-By-Default Zero-Egress Guarantee

Every Armiger node, pipeline (`ArmigerUltraPipeline`, `ArmigerPipeline`), adapter (`SeriesWebhookAdapter`, `MovieWebhookAdapter`, `AudioWebhookAdapter`, `PublicationWebhookAdapter`, `QBittorrentAdapter`, `TransmissionAdapter`), `ArmigerDHTRSIEngine`, `ArmigerDHTDBSync` (`ArmigerDHTDatabaseSync`), and integration bridge (`ArmigerWaypointAdapter`, `BitmagnetArmigerBridge`, `ArmigerArrWebhookService`) **strictly defaults to outbound sharing DISABLED**:

- **Default Constructor State:** `opt_in_sharing: bool = False`
- **Environment Variable Override:** Outbound sharing is enabled only when `opt_in_sharing=True` is explicitly passed or `ARMIGER_DHT_SHARE_OPT_IN` is set to `"1"`, `"true"`, or `"yes"` (defaults to `"0"`).
- **Zero-Egress Enforcement When `opt_in_sharing=False`:**
  - `export_signed_bundle()`, `export_signed_delta_bundle()`, `export_rsi_telemetry_bundle()`, `get_pending_telemetry_batch()`, `publish_to_dht()`, and `broadcast_weights()` are hard-blocked (`None` / `[]` / `(False, "opt_in_sharing_disabled", 0)`), emitting **`0` outbound bundles** and **`0` UDP `put` packets**.
  - Inbound read-only verification (`ingest_signed_bundle()`, `ingest_signed_delta_bundle()`, `verify_and_stage_update()`, `receive_and_apply_shard()`) remains available for verifying and applying trusted community updates locally without transmitting any packet from the local node.
- **Runtime Audit Telemetry (`sharing_status()`):**
  Every component exposes `sharing_status()` returning:
  ```json
  {
    "opt_in_sharing_enabled": false,
    "outbound_bundles_shared": 0,
    "outbound_packets_blocked_opt_out": 2600,
    "outbound_bytes_shared": 0,
    "ephemeral_pubkey_hex": null
  }
  ```

---

## 2. Five-Layer `AnonymityGuard` Defense-in-Depth (When Opted In)

Whenever an operator explicitly opts in (`opt_in_sharing=True` or `ARMIGER_DHT_SHARE_OPT_IN=1`), `AnonymityGuard` unconditionally enforces five layers of cryptographic and mathematical privacy before any payload is signed or transmitted over UDP:

1. **Ephemeral Unlinkable Ed25519 Key Rotation (`ephemeral_identity=True`):**
   - Each contribution session or epoch rotates to a fresh in-memory `Ed25519PrivateKey` via `rotate_ephemeral_identity()`.
   - Zero hardware, MAC, local IP, hostname, or filesystem metadata is ever derived into or attached to the signing key, preventing cross-epoch sender tracking.
2. **Coarse Epoch Timestamp Bucketing (`bucket_seconds = 3600`):**
   - All wire timestamps (`epoch`, `ts_utc`, `updated_utc`, `t`) are quantized to coarse UTC hour boundaries:
     $$t_{\text{wire}} = \lfloor t / 3600 \rfloor \times 3600$$
   - Any high-resolution timestamp with $t \bmod 3600 \neq 0$ is rejected by the wire firewall (`AnonymityViolationError`), eliminating clock-skew fingerprinting and webhook timing correlation.
3. **L2-Clipped + Differentially Private Weight & Calibration Aggregation (`min_k_samples >= 25`):**
   - **Minimum Sample $k$-Anonymity:** Any weight delta, reliability bin, or catalog observation with sample count $n < 25$ is rejected (`AnonymityViolationError`).
   - **L2 Norm Clipping:** Adapter weight and bias deltas $(\Delta w, \Delta b)$ are clipped to L2 norm $\|\Delta\|_2 \le C$ ($C = 1.0$).
   - **Calibrated Gaussian DP + Int8 Quantization:** Independent CSPRNG Gaussian noise $\mathcal{N}(0, (\sigma C)^2 I)$ ($\sigma = 0.05$) is added before symmetric int8 quantization (`delta_w_q8`, `delta_b_q8`, `scale`) and sample count bucketing (`n_bucket`).
4. **One-Way Domain-Separated Catalog Hashing (`armiger-catalog-v2`):**
   - Catalog record keys (`h`) are strictly 16-hex-character truncated `BLAKE2b-64` digests with domain separation:
     $$h = \text{BLAKE2b-64}_{\text{person}=\texttt{armiger-cat-v2}}(\texttt{"armiger-catalog-v2:"} \mathbin{\|} \text{normalize}(k))$$
   - Raw strings, titles, filenames, release names, IMDb/TMDB/TVDB IDs, 40-hex infohashes, URLs, passkeys, and paths are stripped or rejected (`AnonymityViolationError`).
5. **Deep Wire-Payload PII / Canary Firewall (`inspect_wire_payload`):**
   - Every serialized wire payload (`v`) is inspected before Ed25519 signing and before UDP transmission.
   - Immediately raises `AnonymityViolationError` on any:
     - IPv4/IPv6 literal or CIDR block
     - `.local`, `.lan`, `.internal`, `.tailnet`, `ts-net`, or FQDN hostname
     - Filesystem path (`/Users/`, `/home/`, `/srv/`, `/var/`, `C:\`, `\\`) or media/torrent file extension (`.mkv`, `.mp4`, `.torrent`, `.nzb`, `.exe`, `.scr`)
     - URL (`http://`, `https://`, `udp://`) or `magnet:?` URI
     - Tracker passkey, API key, or bearer token
     - Raw 40-hex BitTorrent infohash or external catalog ID (`tt\d+`, `tmdb:`, `tvdb:`)
     - Unexpected JSON schema key outside the strict mathematical whitelist:
       - Top-level keys: `{"v", "epoch", "seq", "model_tier", "delta_w_q8", "delta_b_q8", "scale", "reliability_bins", "n_bucket", "records", "ts_utc", "updated_utc"}`
       - Catalog `records` elements: strict 8-field tuple/dict `[h, c, r, s, a, v, n, t]` where `h` matches `^[0-9a-f]{16}$`, `c, r, s, a, v, n, t` are numeric, $n \ge 25$, and $t \bmod 3600 = 0$.

---

## 3. 12-Node Local UDP BEP-44 & Synthetic DP Convergence Verification Summary

Reproduce all three proofs locally with:

```bash
python3 benchmarks/proof_anonymous_opt_in_sharing.py
pytest test_anonymous_opt_in_dht_sharing.py
```

| Proof | Target Invariant | Measured Result | Status |
| :--- | :--- | :--- | :--- |
| **Proof 1: Default Zero-Egress** | 9 default components, 100 synthetic webhook/catalog events | `opt_in_sharing=False` on 9/9; `0` bundles & `0` UDP `put` packets emitted (`2,600` outbound attempts blocked) | **PASS** |
| **Proof 2: Wire Canary Firewall** | 500 adversarial PII/canary payloads across 10 categories on 12-node UDP swarm | `500 / 500` (`100.0%`) blocked before wire; `0 / 500` (`0.0%`) canary substrings in raw UDP datagrams | **PASS** |
| **Proof 3: Unlinkability & Synthetic DP Convergence** | 12-node local UDP swarm over 5 epochs (60 publications, synthetic adapter + temperature convergence) | `60 / 60` unique ephemeral Ed25519 keys (`8.33%` random linkability); `100%` hour-bucketed timestamps; MIA empirical accuracy `0.590`; ECE `0.08094 -> 0.01368` (`-83.10%`); Ultra F1 `0.93427 -> 0.98004` | **PASS** |
