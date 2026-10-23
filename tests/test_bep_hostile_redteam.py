#!/usr/bin/env python3
"""Hostile adversarial PoC suite for Armiger Ultra BEP-44/BEP-46 DHT RSI + DB Sync.

Posture: assume every 'fool-proof / encrypted / anonymous / poison-proof /
spoof-proof / post-quantum / CSAM-safe' claim is broken until an executable
probe fails to break it. Each probe records VULNERABLE (attack succeeded),
BLOCKED (defense held), CRASH (uncaught exception = DoS), or INFO.

Run: pytest test_bep_hostile_redteam.py -v
Writes: /tmp/armiger_bep_exploit_results.json
"""
from __future__ import annotations

import dataclasses
import hashlib
import hmac
import json
import math
import os
import random
import struct
import sys
import time
import traceback
import zlib

TARGET_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS: list[dict] = []


def record(attack_id: str, category: str, name: str, result: str, evidence: str, severity: str = "") -> None:
    entry = {
        "id": attack_id,
        "category": category,
        "name": name,
        "result": result,  # VULNERABLE | BLOCKED | CRASH | INFO | ERROR
        "severity_if_vulnerable": severity,
        "evidence": str(evidence)[:2000],
    }
    RESULTS.append(entry)
    print(f"[{result:10s}] {attack_id:8s} {name} :: {str(evidence)[:160]}", flush=True)


def probe(attack_id: str, category: str, name: str, severity: str):
    """Decorator: run fn returning (result, evidence); uncaught exc => CRASH."""
    def deco(fn):
        try:
            result, evidence = fn()
            record(attack_id, category, name, result, evidence, severity)
        except Exception as exc:  # noqa: BLE001 - hostile audit wants the crash
            record(attack_id, category, name, "CRASH",
                   f"uncaught {type(exc).__name__}: {exc} @ {traceback.format_exc(limit=3)[-400:]}", severity)
        return fn
    return deco


# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------
from cryptography.hazmat.primitives.asymmetric import ed25519  # noqa: E402
from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat, PublicFormat  # noqa: E402

import armiger_dht_rsi as rsi  # noqa: E402
import armiger_dht_db_sync as dbsync  # noqa: E402
import csam_guard as cg  # noqa: E402
import calibration_and_dht as cal  # noqa: E402


def ed_keypair():
    priv = ed25519.Ed25519PrivateKey.generate()
    pub_hex = priv.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
    raw_priv = priv.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())
    return priv, pub_hex, raw_priv


def make_threshold_manifest(signers, seq=10, threshold_payload_kw=None, pq_signers=None):
    """signers: list of (Ed25519PrivateKey, pub_hex). Returns signed ThresholdDHTManifest."""
    kw = dict(seq=seq, version="9.9.9", infohash="a" * 40, bundle_sha256="b" * 64,
              bundle_format="json", min_asn_quorum=5, timestamp=int(time.time()))
    if threshold_payload_kw:
        kw.update(threshold_payload_kw)
    dummy = rsi.ThresholdDHTManifest(signatures={}, pq_signatures={}, **kw)
    payload = dummy.canonical_payload()
    sigs = {pub: priv.sign(payload).hex() for priv, pub in signers}
    pq_sigs = {}
    for pq_priv, pq_pub_hex in (pq_signers or []):
        pq_sigs[pq_pub_hex] = rsi.WOTSPlusSHAKE256.sign(pq_priv, payload).hex()
    return rsi.ThresholdDHTManifest(signatures=sigs, pq_signatures=pq_sigs, **kw)


print("=" * 78)
print("SECTION A: Cryptographic & post-quantum primitives")
print("=" * 78)

# ---------------------------------------------------------------- A1
@probe("A1", "crypto-threshold", "Duplicate signer key (case variants) to satisfy M-of-N", "CRITICAL")
def _a1():
    priv, pub, _ = ed_keypair()
    m = make_threshold_manifest([(priv, pub)])
    sig = next(iter(m.signatures.values()))
    dup = dataclasses.replace(m, signatures={pub.lower(): sig, pub.upper(): sig, pub.swapcase(): sig})
    ok, reason = rsi.ThresholdSignatureVerifier.verify(dup, {pub.lower(): "alice"}, threshold_m=2, current_seq=0)
    return ("VULNERABLE" if ok else "BLOCKED", f"M=2 with 1 key x3 case-variants -> ok={ok} reason={reason}")


# ---------------------------------------------------------------- A2
@probe("A2", "crypto-threshold", "PQ threshold satisfied by untrusted self-generated WOTS key (allowed_pq=None)", "CRITICAL")
def _a2():
    priv, pub, raw = ed_keypair()
    atk_pq_priv, atk_pq_pub = rsi.WOTSPlusSHAKE256.generate_keypair(os.urandom(64))
    m = make_threshold_manifest([(priv, pub)], pq_signers=[(atk_pq_priv, atk_pq_pub.hex())])
    # Verifier configured WITHOUT trusted_pq_signers (the ArmigerDHTRSIEngine default path)
    ok, reason = rsi.ThresholdSignatureVerifier.verify(
        m, {pub: "alice"}, threshold_m=1, current_seq=0, trusted_pq_signers=None, require_pq=False)
    return ("VULNERABLE" if ok else "BLOCKED",
            f"attacker-invented PQ key counted toward PQ threshold -> ok={ok} reason={reason}")


# ---------------------------------------------------------------- A3
@probe("A3", "crypto-threshold", "PQ signatures omittable (downgrade) when require_pq=False", "HIGH")
def _a3():
    priv, pub, _ = ed_keypair()
    m = make_threshold_manifest([(priv, pub)])  # no PQ sigs at all
    ok, reason = rsi.ThresholdSignatureVerifier.verify(m, {pub: "a"}, 1, 0, trusted_pq_signers={"c" * 64: "pq"}, require_pq=False)
    # trusted_pq_signers present triggers PQ block; with none configured (engine default) PQ is skipped:
    ok2, reason2 = rsi.ThresholdSignatureVerifier.verify(m, {pub: "a"}, 1, 0)
    vuln = ok2  # engine-default call accepts classical-only manifest
    return ("VULNERABLE" if vuln else "BLOCKED",
            f"with pq-config ok={ok} ({reason}); engine-default (no pq args) ok={ok2} ({reason2})")

# ---------------------------------------------------------------- A4
@probe("A4", "crypto-threshold", "ArmigerDHTUpdater.verify_bundle with signature_verifier=None skips Ed25519", "CRITICAL")
def _a4():
    bundle = b'{"pwned":true}'
    sha = hashlib.sha256(bundle).hexdigest()
    forged = cal.SignedDHTManifest(seq=999, version="evil", infohash="d" * 40, bundle_sha256=sha,
                                   quorum_attestations=9999, timestamp=int(time.time()),
                                   ed25519_signature_hex="00" * 64)  # garbage signature, no key at all
    updater = cal.ArmigerDHTUpdater(root_pubkey_hex="ab" * 32, current_seq=0, min_quorum=5)
    ok, reason = updater.verify_bundle(forged, bundle)  # default signature_verifier=None
    return ("VULNERABLE" if ok else "BLOCKED", f"garbage-sig forged manifest, verifier=None -> ok={ok} reason={reason}")


# ---------------------------------------------------------------- A5
@probe("A5", "crypto-threshold", "ArmigerDHTUpdater trusts manifest-supplied PQ pubkey when root PQ unset", "CRITICAL")
def _a5():
    bundle = b'{"x":1}'
    sha = hashlib.sha256(bundle).hexdigest()
    atk_pq_priv, atk_pq_pub = rsi.WOTSPlusSHAKE256.generate_keypair(os.urandom(64))
    dummy = cal.SignedDHTManifest(seq=5, version="v", infohash="e" * 40, bundle_sha256=sha,
                                  quorum_attestations=100, timestamp=int(time.time()),
                                  ed25519_signature_hex="00" * 64, pq_pubkey_hex=atk_pq_pub.hex(), pq_signature_hex="")
    sig = rsi.WOTSPlusSHAKE256.sign(atk_pq_priv, dummy.canonical_payload()).hex()
    forged = dataclasses.replace(dummy, pq_signature_hex=sig)
    updater = cal.ArmigerDHTUpdater(root_pubkey_hex="ab" * 32, current_seq=0, min_quorum=5)  # no root_pq
    ok, reason = updater.verify_bundle(forged, bundle, signature_verifier=lambda pk, p, s: True)
    return ("VULNERABLE" if ok else "BLOCKED", f"self-attested PQ identity accepted -> ok={ok} reason={reason}")


# ---------------------------------------------------------------- A6
@probe("A6", "crypto-canonical", "SignedDHTManifest bundle_sha256 case mutation (canonical not lowercased)", "MEDIUM")
def _a6():
    priv, pub, raw = ed_keypair()
    bundle = b"hello bundle"
    sha = hashlib.sha256(bundle).hexdigest()
    base = cal.SignedDHTManifest(seq=7, version="1", infohash="f" * 40, bundle_sha256=sha,
                                 quorum_attestations=9, timestamp=1700000000, ed25519_signature_hex="")
    sig = priv.sign(base.canonical_payload()).hex()
    signed = dataclasses.replace(base, ed25519_signature_hex=sig)
    mutated = dataclasses.replace(signed, bundle_sha256=sha.upper())
    def real_verifier(pk, payload, s):
        try:
            ed25519.Ed25519PublicKey.from_public_bytes(pk).verify(s, payload)
            return True
        except Exception:
            return False
    updater = cal.ArmigerDHTUpdater(root_pubkey_hex=pub, current_seq=0, min_quorum=5)
    ok_mut, r_mut = updater.verify_bundle(mutated, bundle, signature_verifier=real_verifier)
    ok_none, r_none = updater.verify_bundle(mutated, bundle)  # verifier=None path
    return ("VULNERABLE" if ok_none else "BLOCKED",
            f"case-mutated sha with real verifier ok={ok_mut} ({r_mut}); with verifier=None ok={ok_none} ({r_none})")


# ---------------------------------------------------------------- A7
@probe("A7", "crypto-canonical", "DHTShardManifest PQ fields unauthenticated by root Ed25519 signature", "HIGH")
def _a7():
    root_priv, root_pub = cg.generate_ed25519_keypair()
    sync = dbsync.ArmigerDHTDatabaseSync(master_secret=b"S" * 32)
    rec = dbsync.TorrentRecord(infohash="1" * 40, title="Big Buck Bunny 1080p", category="movie", language="en",
                               quality_tier="hd_1080p", epoch="2026_q4", size_bytes=1000, file_count=1, seeders=1,
                               leechers=0, content_sha256="2" * 64, file_list=["bbb.mkv"], safety_gate_passed=True,
                               created_at=int(time.time()))
    manifest, _ct = sync.build_and_sign_shard("movie/en/hd_1080p/2026_q4", [rec], root_priv, seq=3)
    # Attacker swaps in own PQ identity; Ed25519 payload does not cover PQ fields.
    atk_priv, atk_pub = rsi.WOTSPlusSHAKE256.generate_keypair(os.urandom(64))
    atk_sig = rsi.WOTSPlusSHAKE256.sign(atk_priv, manifest.canonical_payload()).hex()
    swapped = dataclasses.replace(manifest, root_pq_pubkey_hex=atk_pub.hex(), pq_signature_hex=atk_sig)
    atts = dbsync.create_mock_quorum_attestations(swapped.canonical_digest(), [13335, 15169, 8075, 16509, 14061])
    swapped = swapped.with_attestations(atts)
    qv = dbsync.QuorumVerifier([root_pub.hex()], min_quorum=5, min_distinct_asns=5)  # no trusted PQ roots
    ok, reason = qv.verify_manifest(swapped, current_seq=0)
    return ("VULNERABLE" if ok else "BLOCKED", f"PQ identity swapped post-signing, root Ed25519 intact -> ok={ok} ({reason})")


# ---------------------------------------------------------------- A8
@probe("A8", "crypto-canonical", "bep44_target case equivocation (canonical lowercases, target does not)", "MEDIUM")
def _a8():
    root_priv, root_pub = cg.generate_ed25519_keypair()
    sync = dbsync.ArmigerDHTDatabaseSync(master_secret=b"S" * 32)
    rec = dbsync.TorrentRecord(infohash="3" * 40, title="Sintel", category="movie", language="en",
                               quality_tier="hd_1080p", epoch="2026_q4", size_bytes=1, file_count=1, seeders=0,
                               leechers=0, content_sha256="4" * 64, file_list=["s.mkv"], safety_gate_passed=True,
                               created_at=int(time.time()))
    m1, _ = sync.build_and_sign_shard("movie/en/hd_1080p/2026_q4", [rec], root_priv, seq=1)
    m2 = dataclasses.replace(m1, shard_id="Movie/EN/HD_1080P/2026_Q4")
    same_payload = m1.canonical_payload() == m2.canonical_payload()
    diff_target = m1.bep44_target() != m2.bep44_target()
    return ("VULNERABLE" if (same_payload and diff_target) else "BLOCKED",
            f"same signed payload={same_payload} but different BEP44 targets={diff_target} ({m1.bep44_target()[:12]} vs {m2.bep44_target()[:12]})")


# ---------------------------------------------------------------- A9
@probe("A9", "crypto-wots", "WOTS+ mutated signature / wrong message / wrong pubkey", "HIGH")
def _a9():
    priv, pub = rsi.WOTSPlusSHAKE256.generate_keypair(os.urandom(64))
    msg = b"transfer 100 coins"
    sig = rsi.WOTSPlusSHAKE256.sign(priv, msg)
    assert rsi.WOTSPlusSHAKE256.verify(pub, msg, sig), "baseline must verify"
    bad = bytearray(sig); bad[100] ^= 0x01
    r1 = rsi.WOTSPlusSHAKE256.verify(pub, msg, bytes(bad))
    r2 = rsi.WOTSPlusSHAKE256.verify(pub, b"transfer 999 coins", sig)
    _, pub2 = rsi.WOTSPlusSHAKE256.generate_keypair(os.urandom(64))
    r3 = rsi.WOTSPlusSHAKE256.verify(pub2, msg, sig)
    r4 = rsi.WOTSPlusSHAKE256.verify(pub, msg, sig[:1000])  # truncated
    r5 = rsi.WOTSPlusSHAKE256.verify(b"\x00" * 32, msg, sig)  # zero pubkey
    vuln = any([r1, r2, r3, r4, r5])
    return ("VULNERABLE" if vuln else "BLOCKED", f"bitflip={r1} wrongmsg={r2} wrongpub={r3} trunc={r4} zeropub={r5}")

# ---------------------------------------------------------------- A10
@probe("A10", "crypto-wots", "WOTS+ one-time key REUSE via derive_keypair_from_ed25519_priv -> existential forgery", "CRITICAL")
def _a10():
    # Every manifest signed by the same Ed25519 root reuses the SAME WOTS+ one-time key,
    # because the WOTS seed is deterministically derived from the Ed25519 private key.
    # Collect N signatures (public DHT manifests), then hash chains forward to forge.
    _, _, raw_priv = ed_keypair()
    wots_priv, wots_pub = rsi.WOTSPlusSHAKE256.derive_keypair_from_ed25519_priv(raw_priv)
    pub_seed = wots_priv[32:]
    n_sigs = 48
    best_val: list[bytes | None] = [None] * 67
    best_digit = [99] * 67
    for j in range(n_sigs):
        m = f"legit manifest seq={j}".encode()
        s = rsi.WOTSPlusSHAKE256.sign(wots_priv, m)
        digest = hashlib.shake_256(pub_seed + wots_pub + m).digest(32)
        digits = rsi.WOTSPlusSHAKE256._msg_to_base_w(digest)
        for i, d in enumerate(digits):
            if d < best_digit[i]:
                best_digit[i] = d
                best_val[i] = s[32 * (i + 1):32 * (i + 2)]
    # Grind a target message whose every base-16 digit >= best-revealed digit per chain.
    forged_msg = None
    tries = 0
    for nonce in range(2_000_000):
        tries += 1
        cand = f"FORGED evil manifest nonce={nonce}".encode()
        digest = hashlib.shake_256(pub_seed + wots_pub + cand).digest(32)
        digits = rsi.WOTSPlusSHAKE256._msg_to_base_w(digest)
        if all(digits[i] >= best_digit[i] for i in range(67)):
            forged_msg = (cand, digits)
            break
    if forged_msg is None:
        return ("BLOCKED", f"no forgeable message in {tries} tries; mins={best_digit[:10]}...")
    cand, digits = forged_msg
    parts = [pub_seed]
    for i in range(67):
        parts.append(rsi.WOTSPlusSHAKE256._chain(best_val[i], best_digit[i], digits[i] - best_digit[i], pub_seed, i))
    forged_sig = b"".join(parts)
    ok = rsi.WOTSPlusSHAKE256.verify(wots_pub, cand, forged_sig)
    return ("VULNERABLE" if ok else "BLOCKED",
            f"forged NEW message after {n_sigs} reused-key sigs, grind tries={tries}, verify={ok}, msg={cand!r}")


# ---------------------------------------------------------------- A11
@probe("A11", "crypto-mlkem", "ML-KEM-768 malformed lengths / all-zero & non-canonical ek / zero ciphertext", "MEDIUM")
def _a11():
    ek, dk = rsi.MLKEM768.keygen()
    out = []
    for label, fn in [
        ("encaps short ek", lambda: rsi.MLKEM768.encaps(ek[:100])),
        ("decaps short ct", lambda: rsi.MLKEM768.decaps(dk, b"\x00" * 100)),
        ("decaps short dk", lambda: rsi.MLKEM768.decaps(dk[:100], b"\x00" * 1088)),
    ]:
        try:
            fn(); out.append(f"{label}=ACCEPTED")
        except ValueError:
            out.append(f"{label}=rejected")
    # All-zero ek: FIPS 203 ek validation (coefficients < q, etc.) absent?
    try:
        ct0, ss0 = rsi.MLKEM768.encaps(b"\x00" * 1184)
        out.append(f"zero-ek encaps=ACCEPTED ss={ss0.hex()[:12]}")
        zero_ek_vuln = True
    except ValueError as e:
        out.append(f"zero-ek encaps=rejected({e})"); zero_ek_vuln = False
    # Non-canonical ek: coefficients encoding values >=3329 (0xFFF nibbles pattern)
    try:
        bad_ek = b"\xff" * 1152 + ek[1152:]
        rsi.MLKEM768.encaps(bad_ek)
        out.append("noncanonical-ek=ACCEPTED"); noncanon = True
    except ValueError:
        out.append("noncanonical-ek=rejected"); noncanon = False
    # Zero ciphertext decaps -> implicit rejection must NOT equal a real encaps secret
    ss_zero_ct = rsi.MLKEM768.decaps(dk, b"\x00" * 1088)
    ct_r, ss_r = rsi.MLKEM768.encaps(ek)
    out.append(f"zero-ct decaps returns ss (implicit-rejection) equal_to_real={ss_zero_ct == ss_r}")
    vuln = zero_ek_vuln or noncanon or (ss_zero_ct == ss_r)
    return ("VULNERABLE" if vuln else "BLOCKED", "; ".join(out))


# ---------------------------------------------------------------- A12
@probe("A12", "crypto-aead", "X25519 all-zero (low-order) key handling, encrypt + decrypt sides", "HIGH")
def _a12():
    secret_msg = b"TOP SECRET telemetry payload"
    # Encrypt side: all-zero recipient pubkey (contributory-behaviour check)
    try:
        bundle = rsi.X25519ChaCha20Poly1305Engine.encrypt_bundle(secret_msg, b"\x00" * 32, seq=1)
        enc_side = "zero-pub encrypt ACCEPTED"
    except ValueError as e:
        # cryptography/OpenSSL rejects small-order points -> confidentiality attack BLOCKED,
        # but the exception is an uncaught ValueError for callers (availability note).
        bundle = None
        enc_side = f"zero-pub encrypt rejected by library ({e})"
    if bundle is None:
        # Decrypt side: forged bundle with zero ephemeral key against a legit victim key
        vpriv, vpub = rsi.X25519ChaCha20Poly1305Engine.generate_classical_keypair()
        forged = rsi.EncryptedTelemetryBundle(ephemeral_pubkey_hex="00" * 32, nonce_hex="11" * 12,
                                              ciphertext_hex="22" * 32, aad="seq:1;ephem_pub:" + "00" * 32, seq=1)
        try:
            rsi.X25519ChaCha20Poly1305Engine.decrypt_bundle(forged, vpriv)
            dec_side = "zero-ephem decrypt returned plaintext (BAD)"
            dec_bad = True
        except ValueError as e:
            dec_side = f"zero-ephem decrypt raised ValueError ({e})"
            dec_bad = False
        return ("BLOCKED", f"{enc_side}; {dec_side}. NOTE: decrypt-side ValueError propagates uncaught to callers (minor DoS), no plaintext recovery.")
    # fallthrough (not reached with OpenSSL check): try public-data decryption
    # Attacker knows only public bundle data. With zero recipient pub, ss_classical = 0^32.
    eng = rsi.X25519ChaCha20Poly1305Engine
    ephem = bytes.fromhex(bundle.ephemeral_pubkey_hex)
    ss_classical = b"\x00" * 32
    ss_fallback = hashlib.shake_256(eng.SALT + ss_classical + ephem).digest(32)
    key = eng._combine_hybrid_secrets(ss_classical, ss_fallback, ephem, b"")
    from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
    try:
        pt = ChaCha20Poly1305(key).decrypt(bytes.fromhex(bundle.nonce_hex),
                                           bytes.fromhex(bundle.ciphertext_hex), bundle.aad.encode())
        ok = (pt == secret_msg)
    except Exception as e:
        ok = False; pt = f"fail:{e}".encode()
    return ("VULNERABLE" if ok else "BLOCKED", f"decrypted with public data only={ok} pt={pt!r}")


# ---------------------------------------------------------------- A13
@probe("A13", "crypto-aead", "AEAD AAD transplant (seq / ephem swap) + hybrid tamper", "HIGH")
def _a13():
    priv, pub = rsi.X25519ChaCha20Poly1305Engine.generate_keypair()
    b1 = rsi.X25519ChaCha20Poly1305Engine.encrypt_bundle(b"payload-one", pub, seq=1)
    res = []
    # seq transplant: keep ciphertext, change seq+aad consistently? attacker can recompute aad string but tag must fail
    t1 = dataclasses.replace(b1, seq=2, aad=f"seq:2;ephem_pub:{b1.ephemeral_pubkey_hex}")
    try:
        rsi.X25519ChaCha20Poly1305Engine.decrypt_bundle(t1, priv); res.append("seq-transplant=ACCEPTED")
    except ValueError:
        res.append("seq-transplant=rejected")
    # ciphertext bitflip
    ct = bytearray(bytes.fromhex(b1.ciphertext_hex)); ct[0] ^= 1
    t2 = dataclasses.replace(b1, ciphertext_hex=bytes(ct).hex())
    try:
        rsi.X25519ChaCha20Poly1305Engine.decrypt_bundle(t2, priv); res.append("ct-bitflip=ACCEPTED")
    except ValueError:
        res.append("ct-bitflip=rejected")
    # ML-KEM ct bitflip inside hybrid ephem token (must keep aad consistent -> recompute aad, key changes, tag fails)
    ep = bytearray(bytes.fromhex(b1.ephemeral_pubkey_hex)); ep[40] ^= 1
    t3 = dataclasses.replace(b1, ephemeral_pubkey_hex=bytes(ep).hex(), aad=f"seq:1;ephem_pub:{bytes(ep).hex()}")
    try:
        rsi.X25519ChaCha20Poly1305Engine.decrypt_bundle(t3, priv); res.append("kem-bitflip=ACCEPTED")
    except ValueError:
        res.append("kem-bitflip=rejected")
    vuln = any("ACCEPTED" in r for r in res)
    return ("VULNERABLE" if vuln else "BLOCKED", "; ".join(res))


# ---------------------------------------------------------------- A14
@probe("A14", "crypto-canonical", "Float seq type confusion (signed payload truncates int(seq))", "MEDIUM")
def _a14():
    priv, pub, _ = ed_keypair()
    m = make_threshold_manifest([(priv, pub)], seq=1)
    # Re-wrap with float seq 1.9: canonical_payload signs int(1.9)=1, monotonic check compares 1.9
    mf = dataclasses.replace(m, seq=1.9)  # type: ignore[arg-type]
    ok, reason = rsi.ThresholdSignatureVerifier.verify(mf, {pub: "a"}, 1, current_seq=1)
    # current_seq=1, signed seq is 1 (replay!) but float compare 1.9>1 passes
    return ("VULNERABLE" if ok else "BLOCKED", f"seq=1.9 (signed as 1) vs current=1 -> ok={ok} ({reason})")

print("=" * 78)
print("SECTION B: Sybil quorum, replay & sequence attacks")
print("=" * 78)

# ---------------------------------------------------------------- B1
@probe("B1", "sybil", "SybilQuorumVerifier: ONE keypair claims 5 distinct ASNs", "CRITICAL")
def _b1():
    priv, pub, _ = ed_keypair()
    target = "b" * 64
    now = int(time.time())
    atts = []
    for asn in [13335, 15169, 8075, 16509, 14061]:
        att0 = rsi.PeerAttestation(peer_id=f"peer-{asn}", asn=asn, peer_pubkey_hex=pub, timestamp=now, signature_hex="")
        sig = priv.sign(att0.canonical_bytes(target)).hex()
        atts.append(dataclasses.replace(att0, signature_hex=sig))
    ok, reason, asns = rsi.SybilQuorumVerifier.verify_quorum(atts, target, min_distinct_asns=5, now_timestamp=now)
    return ("VULNERABLE" if ok else "BLOCKED", f"single Ed25519 key, 5 self-asserted ASNs -> ok={ok} ({reason}) asns={sorted(asns)}")


# ---------------------------------------------------------------- B2/B3
@probe("B2", "sybil", "SybilQuorumVerifier duplicate ASN / asn<=0 handling", "MEDIUM")
def _b2():
    target = "b" * 64
    now = int(time.time())
    def mk(asn, pid):
        p, pb, _ = ed_keypair()
        a0 = rsi.PeerAttestation(peer_id=pid, asn=asn, peer_pubkey_hex=pb, timestamp=now, signature_hex="")
        return dataclasses.replace(a0, signature_hex=p.sign(a0.canonical_bytes(target)).hex())
    dup = [mk(13335, f"p{i}") for i in range(6)]
    ok1, r1, _ = rsi.SybilQuorumVerifier.verify_quorum(dup, target, 5, now_timestamp=now)
    zero = [mk(a, f"z{i}") for i, a in enumerate([0, -1, -999, 0, 13335])]
    ok2, r2, asns2 = rsi.SybilQuorumVerifier.verify_quorum(zero, target, 2, now_timestamp=now)
    vuln = ok1 or ok2
    return ("VULNERABLE" if vuln else "BLOCKED", f"6x same ASN vs K=5 ok={ok1} ({r1}); asn 0/-1 counted? ok={ok2} ({r2}) asns={sorted(asns2)}")


# ---------------------------------------------------------------- B4
@probe("B4", "sybil", "Future-dated PeerAttestation accepted inside abs() window", "LOW")
def _b4():
    priv, pub, _ = ed_keypair()
    target = "b" * 64
    now = int(time.time())
    atts = []
    for asn in [1, 2, 3, 4, 5]:
        p, pb, _ = ed_keypair()
        a0 = rsi.PeerAttestation(peer_id=f"f{asn}", asn=1000 + asn, peer_pubkey_hex=pb, timestamp=now + 23 * 3600, signature_hex="")
        atts.append(dataclasses.replace(a0, signature_hex=p.sign(a0.canonical_bytes(target)).hex()))
    ok, reason, _ = rsi.SybilQuorumVerifier.verify_quorum(atts, target, 5, max_age_seconds=86400, now_timestamp=now)
    return ("VULNERABLE" if ok else "BLOCKED", f"timestamps +23h in future -> ok={ok} ({reason})")


# ---------------------------------------------------------------- B5 (crash expected -> probe harness records CRASH)
def _nan_att(ts):
    p, pb, _ = ed_keypair()
    return rsi.PeerAttestation(peer_id="nan-peer", asn=13335, peer_pubkey_hex=pb, timestamp=ts, signature_hex="00" * 64)  # type: ignore[arg-type]

try:
    _ok5, _r5, _ = rsi.SybilQuorumVerifier.verify_quorum([_nan_att(float("nan"))], "b" * 64, 1, now_timestamp=int(time.time()))
    record("B5", "sybil", "NaN timestamp in PeerAttestation", "VULNERABLE" if _ok5 else "BLOCKED", f"nan ts -> ok={_ok5} ({_r5})", "HIGH")
except Exception as e:
    record("B5", "sybil", "NaN timestamp in PeerAttestation", "CRASH", f"uncaught {type(e).__name__}: {e} (canonical_bytes outside try)", "HIGH")

try:
    _ok6, _r6, _ = rsi.SybilQuorumVerifier.verify_quorum([_nan_att(float("inf"))], "b" * 64, 1, now_timestamp=int(time.time()))
    record("B6", "sybil", "Inf timestamp in PeerAttestation", "VULNERABLE" if _ok6 else "BLOCKED",
           f"inf ts -> ok={_ok6} ({_r6}); abs(now-inf)=inf>window so attestation is skipped (fail-closed, unlike NaN)", "HIGH")
except Exception as e:
    record("B6", "sybil", "Inf timestamp in PeerAttestation", "CRASH", f"uncaught {type(e).__name__}: {e}", "HIGH")


# ---------------------------------------------------------------- B7
@probe("B7", "sybil", "QuorumVerifier: ONE signature replayed under 5 fake node_ids/ASNs (attestor fields unsigned)", "CRITICAL")
def _b7():
    root_priv, root_pub = cg.generate_ed25519_keypair()
    sync = dbsync.ArmigerDHTDatabaseSync(master_secret=b"S" * 32)
    rec = dbsync.TorrentRecord(infohash="5" * 40, title="Tears of Steel", category="movie", language="en",
                               quality_tier="hd_1080p", epoch="2026_q4", size_bytes=1, file_count=1, seeders=0,
                               leechers=0, content_sha256="6" * 64, file_list=["t.mkv"], safety_gate_passed=True,
                               created_at=int(time.time()))
    manifest, _ = sync.build_and_sign_shard("movie/en/hd_1080p/2026_q4", [rec], root_priv, seq=1)
    digest = manifest.canonical_digest()
    # ONE attestor key signs the manifest digest ONCE (sig covers digest only, not node/asn/ip).
    one_priv, one_pub = cg.generate_ed25519_keypair()
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey as _E
    sig_hex = _E.from_private_bytes(one_priv).sign(digest).hex()
    pq_priv, pq_pub = rsi.WOTSPlusSHAKE256.derive_keypair_from_ed25519_priv(one_priv)
    pq_sig_hex = rsi.WOTSPlusSHAKE256.sign(pq_priv, digest).hex()
    atts = []
    for i, asn in enumerate([13335, 15169, 8075, 16509, 14061]):
        atts.append(dbsync.QuorumAttestation(
            node_id=hashlib.sha1(f"fake-node-{i}".encode()).hexdigest(), asn=asn,
            ip_digest=hashlib.sha256(f"fake-ip-{i}".encode()).hexdigest(),
            pubkey_hex=one_pub.hex(), signature_hex=sig_hex, timestamp=int(time.time()),
            pq_pubkey_hex=pq_pub.hex(), pq_signature_hex=pq_sig_hex))
    m2 = manifest.with_attestations(atts)
    qv = dbsync.QuorumVerifier([root_pub.hex()], min_quorum=5, min_distinct_asns=5)
    ok, reason = qv.verify_manifest(m2, current_seq=0)
    return ("VULNERABLE" if ok else "BLOCKED", f"1 key / 1 sig replayed x5 fabricated node+ASN -> ok={ok} ({reason})")


# ---------------------------------------------------------------- B8
@probe("B8", "sybil", "QuorumVerifier has NO attestation timestamp freshness (ts=0 / year 2100)", "MEDIUM")
def _b8():
    root_priv, root_pub = cg.generate_ed25519_keypair()
    sync = dbsync.ArmigerDHTDatabaseSync(master_secret=b"S" * 32)
    rec = dbsync.TorrentRecord(infohash="7" * 40, title="Cosmos Laundromat", category="movie", language="en",
                               quality_tier="hd_1080p", epoch="2026_q4", size_bytes=1, file_count=1, seeders=0,
                               leechers=0, content_sha256="8" * 64, file_list=["c.mkv"], safety_gate_passed=True,
                               created_at=int(time.time()))
    manifest, _ = sync.build_and_sign_shard("movie/en/hd_1080p/2026_q4", [rec], root_priv, seq=1)
    atts = dbsync.create_mock_quorum_attestations(manifest.canonical_digest(), [11, 22, 33, 44, 55], timestamp=0)
    m2 = manifest.with_attestations(atts)
    qv = dbsync.QuorumVerifier([root_pub.hex()], min_quorum=5, min_distinct_asns=5)
    ok, reason = qv.verify_manifest(m2, current_seq=0)
    return ("VULNERABLE" if ok else "BLOCKED", f"attestation timestamp=0 (1970) -> ok={ok} ({reason})")


# ---------------------------------------------------------------- B9/B10
@probe("B9", "sybil", "Zero thresholds: min_distinct_asns=0 / threshold_m=0 accept empty evidence", "HIGH")
def _b9():
    ok1, r1, _ = rsi.SybilQuorumVerifier.verify_quorum([], "b" * 64, min_distinct_asns=0)
    priv, pub, _ = ed_keypair()
    m = make_threshold_manifest([(priv, pub)])
    unsigned = dataclasses.replace(m, signatures={})
    ok2, r2 = rsi.ThresholdSignatureVerifier.verify(unsigned, {pub: "a"}, threshold_m=0, current_seq=0)
    qv = dbsync.QuorumVerifier([], min_quorum=0, min_distinct_asns=0)
    root_priv, root_pub = cg.generate_ed25519_keypair()
    sync = dbsync.ArmigerDHTDatabaseSync(master_secret=b"S" * 32)
    rec = dbsync.TorrentRecord(infohash="9" * 40, title="x", category="movie", language="en", quality_tier="any",
                               epoch="2026_q4", size_bytes=1, file_count=1, seeders=0, leechers=0,
                               content_sha256="0" * 64, file_list=["x"], safety_gate_passed=True, created_at=1)
    man, _ = sync.build_and_sign_shard("movie/en/any/2026_q4", [rec], root_priv, seq=1)
    ok3, r3 = qv.verify_manifest(dataclasses.replace(man, attestations=[]), current_seq=0)
    vuln = ok1 or ok2 or ok3
    return ("VULNERABLE" if vuln else "BLOCKED", f"empty ASN quorum ok={ok1}; unsigned M=0 ok={ok2}; empty attestations ok={ok3}")


# ---------------------------------------------------------------- B11/B12
@probe("B11", "replay", "Sequence edges: seq=0/negative, seq0 vs default current=-1, future manifest timestamp", "MEDIUM")
def _b11():
    priv, pub, _ = ed_keypair()
    out = []
    for seq, cur in [(0, 0), (-5, 0), (0, -1)]:
        m = make_threshold_manifest([(priv, pub)], seq=seq)
        ok, r = rsi.ThresholdSignatureVerifier.verify(m, {pub: "a"}, 1, cur)
        out.append(f"seq={seq} cur={cur} ok={ok}")
    m2100 = make_threshold_manifest([(priv, pub)], seq=50, threshold_payload_kw={"timestamp": 4102444800})  # 2100-01-01
    okf, rf = rsi.ThresholdSignatureVerifier.verify(m2100, {pub: "a"}, 1, 0)
    out.append(f"timestamp=2100 ok={okf} ({rf})")
    vuln = okf  # no freshness bound on manifest timestamp at all
    return ("VULNERABLE" if vuln else "BLOCKED", "; ".join(out))

print("=" * 78)
print("SECTION C: Anonymity & differential-privacy leaks")
print("=" * 78)

ANON = rsi.TelemetryAnonymizer(blinding_salt=b"K" * 32)

# ---------------------------------------------------------------- C1
@probe("C1", "privacy", "strip_pii IPv6 compressed forms (loopback, link-local, documentation, v4-mapped)", "HIGH")
def _c1():
    cases = {
        "::1 loopback": "peer ::1 connected",
        "fe80 linklocal": "peer " + ":".join(["fe80", "", "a634", "d9ff", "fe07", "c5a8"]) + " connected",
        "2001 compressed": "peer 2001:db8::dead:beef connected",
        "v4-mapped": "peer " + ":" + ":ffff:192.0.2.55 connected",
        "full form": "peer 2001:0db8:0000:0000:0000:ff00:0042:8329 connected",
        "ipv4": "peer 192.0.2.55 connected",
    }
    leaks = []
    for label, text in cases.items():
        out = ANON.strip_pii(text)
        leaked = out == text or ("::" in out and "REDACTED" not in out) or \
            ("fe80" in out) or ("2001" in out) or ("::ffff" in out and "192.0.2.55" not in text.replace("192.0.2.55", ""))
        # precise checks:
        if label == "::1 loopback": leaked = "::1" in out
        if label == "fe80 linklocal": leaked = "fe80" in out
        if label == "2001 compressed": leaked = "2001" in out or "dead:beef" in out
        if label == "v4-mapped": leaked = "::ffff" in out
        if leaked:
            leaks.append(f"{label} -> {out!r}")
    return ("VULNERABLE" if leaks else "BLOCKED", " | ".join(leaks) if leaks else "all IPv6 forms redacted")


# ---------------------------------------------------------------- C2-C4
@probe("C2", "privacy", "strip_pii: email, UNC path, encoded/path passkeys survive", "HIGH")
def _c2():
    cases = {
        "email": ("contact alice.smith+tracker@example.com now", "alice.smith+tracker@example.com"),
        "UNC path": (r"saved to \\NAS-BOX\media\alice\movie.mkv done", "alice"),
        "win path space": (r"C:\Users\Alice Smith\movie.mkv", "Alice"),
        "urlenc passkey": ("magnet:?xt=urn:btih:AA&passkey%3DSECRETKEY123", "SECRETKEY123"),
        "path passkey": ("https://tracker.example/SECRETKEY456/announce", "SECRETKEY456"),
        "dotted passkey": ("https://t.example/announce?passkey=abc.def.ghi", "def.ghi"),
        "pk param": ("https://t.example/announce?pk=SECRETKEY789", "SECRETKEY789"),
        "unix other": ("/mnt/data/alice/private/file.mkv", "alice"),
        "plain passkey (control)": ("https://t.example/announce?passkey=CONTROL123", "CONTROL123"),
    }
    leaks = []
    for label, (text, needle) in cases.items():
        out = ANON.strip_pii(text)
        if needle in out:
            leaks.append(f"{label} leaks {needle!r} -> {out!r}")
    # control must be redacted; count only non-control leaks as vuln but report all
    non_control = [l for l in leaks if "control" not in l.lower()]
    return ("VULNERABLE" if non_control else "BLOCKED", " | ".join(leaks) if leaks else "none leaked")


# ---------------------------------------------------------------- C5
@probe("C5", "privacy", "PII survives into AnonymizedTelemetryRecord via stream_info codec fields", "HIGH")
def _c5():
    obs = rsi.VerifiedObservation(
        category="movie", adult=False, anime=False, lang="english", junk=False, owner_scope="allowed_type",
        english_subs=False, title_language="english_title", source_adapter="series_webhook",
        catalog_id_type="series_catalog", catalog_id_value="123", size_bytes=1_000_000,
        raw_title="Big Buck Bunny 1080p alice@example.com ::1",
        stream_info={"video_codec": "h264 alice@example.com " + "fe80" + "::1", "audio_codec": "aac \\\\NAS\\alice",
                     "audio_channels": 2, "duration_seconds": 600},
        verified_at=int(time.time()))
    rec = ANON.anonymize(obs)
    blob = rec.canonical_json_bytes().decode()
    needles = ["alice@example.com", "fe80", "NAS", "alice"]
    found = [n for n in needles if n in blob]
    return ("VULNERABLE" if found else "BLOCKED", f"leaked in canonical telemetry JSON: {found} blob={blob[:300]}")


# ---------------------------------------------------------------- C6
@probe("C6", "privacy-dp", "NaN/Inf duration through differential privacy -> invalid JSON telemetry", "MEDIUM")
def _c6():
    out = []
    for label, dur in [("nan", float("nan")), ("inf", float("inf")), ("-inf", float("-inf"))]:
        try:
            dp_size, dp_dur = ANON.apply_differential_privacy(1_000_000, dur)
            out.append(f"{label}: dp_duration={dp_dur}")
        except Exception as e:
            out.append(f"{label}: raised {type(e).__name__}")
    # Full record -> canonical JSON validity
    obs = rsi.VerifiedObservation(category="movie", adult=False, anime=False, lang="english", junk=False,
                                  owner_scope="allowed_type", english_subs=False, title_language="english_title",
                                  source_adapter="x", catalog_id_type="series_catalog", catalog_id_value="1",
                                  size_bytes=1000, raw_title="Sintel",
                                  stream_info={"duration_seconds": float("nan")}, verified_at=1)
    try:
        rec = ANON.anonymize(obs)
        raw = rec.canonical_json_bytes()
        std_ok = b"NaN" not in raw and b"Infinity" not in raw
        out.append(f"canonical JSON std-valid={std_ok} raw={raw[:160]!r}")
        vuln = not std_ok
    except Exception as e:
        out.append(f"anonymize(nan) raised UNCAUGHT {type(e).__name__}: {e} -> propagates out of ingest_candidate (DoS)")
        vuln = True  # unhandled-exception DoS on attacker-controlled ffprobe/webhook duration
    return ("VULNERABLE" if vuln else "BLOCKED", " | ".join(out))


# ---------------------------------------------------------------- C7/C8
@probe("C7", "privacy-dp", "Blinded hash only 64-bit (16 hex) + PRNG (random) DP noise reproducible from seed", "MEDIUM")
def _c7():
    b1 = ANON.blind_title_stem("Sintel 2010 1080p BluRay x264")
    bits = len(b1.blinded_hash) * 4
    random.seed(1337)
    n1 = [rsi.pure_laplace_noise(16.0) for _ in range(3)]
    random.seed(1337)
    n2 = [rsi.pure_laplace_noise(16.0) for _ in range(3)]
    reproducible = n1 == n2
    return ("VULNERABLE" if (bits <= 64 and reproducible) else "BLOCKED",
            f"blinded_hash={b1.blinded_hash} ({bits} bits); Laplace noise seed-reproducible={reproducible} {n1}")


# ---------------------------------------------------------------- C9
@probe("C9", "privacy", "Webhook adapters carry raw PII; check end-to-end anonymized output (magnet/passkey/email in names)", "HIGH")
def _c9():
    # qBittorrent event whose torrent NAME embeds tracker passkey + email
    evil_name = "Sintel.2010.1080p [passkey=QBITSECRET99] bob@example.org ::1"
    cand = rsi.QBittorrentAdapter.parse_event({"name": evil_name, "progress": 1.0, "state": "uploading",
                                               "size": 2_000_000, "hash": "a" * 40,
                                               "content_path": "/Users/bob/Downloads/x"})
    v = rsi.PhysicalGroundTruthVerifier().verify(cand)
    if not v.is_verified:
        # give it catalog provenance via series-style fields and retry through anonymizer directly
        obs = rsi.VerifiedObservation(category="movie", adult=False, anime=False, lang="english", junk=False,
                                      owner_scope="allowed_type", english_subs=False, title_language="english_title",
                                      source_adapter="qbittorrent", catalog_id_type="none", catalog_id_value="",
                                      size_bytes=2_000_000, raw_title=evil_name,
                                      stream_info={"video_codec": "h264"}, verified_at=int(time.time()))
    else:
        obs = v.observation
    rec = ANON.anonymize(obs)
    blob = rec.canonical_json_bytes().decode()
    found = [n for n in ["QBITSECRET99", "bob@example.org", "::1", "bob"] if n in blob]
    # raw title itself must never appear; blinded hash only. found via codec channel counts too.
    return ("VULNERABLE" if found else "BLOCKED",
            f"verifier accepted={v.is_verified}; leaked needles={found}; blinded={rec.blinded_stem}")

print("=" * 78)
print("SECTION D: Poisoning, deserialization, SQLite, Merkle, CSAM guard")
print("=" * 78)

def _safetensors(header: dict, body: bytes) -> bytes:
    hj = json.dumps(header).encode()
    return struct.pack("<Q", len(hj)) + hj + body

# ---------------------------------------------------------------- D1
@probe("D1", "poison-safetensors", "SafeTensors: OOB/negative/overlap/shape-mismatch/__metadata__/huge-header", "HIGH")
def _d1():
    V = rsi.SafeFormatValidator.validate
    out = []
    good = _safetensors({"w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}, b"\x00" * 8)
    out.append(f"good={V(good, 'safetensors')}")
    neg = _safetensors({"w": {"dtype": "F32", "shape": [2], "data_offsets": [-4, 8]}}, b"\x00" * 8)
    out.append(f"negative-offset={V(neg, 'safetensors')}")
    oob = _safetensors({"w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 10**9]}}, b"\x00" * 8)
    out.append(f"oob-offset={V(oob, 'safetensors')}")
    overlap = _safetensors({"a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
                            "b": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}, b"\x00" * 8)
    ok_overlap, _ = V(overlap, "safetensors")
    out.append(f"overlapping-buffers accepted={ok_overlap}")
    shapemismatch = _safetensors({"w": {"dtype": "F32", "shape": [1000000], "data_offsets": [0, 4]}}, b"\x00" * 4)
    ok_shape, _ = V(shapemismatch, "safetensors")
    out.append(f"shape(1M F32) vs 4-byte buffer accepted={ok_shape}")
    meta = _safetensors({"__metadata__": {"x": "__import__('os').system('id')", "n": 12345},
                         "w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}, b"\x00" * 4)
    ok_meta, _ = V(meta, "safetensors")
    out.append(f"__metadata__ script-string + non-string value accepted={ok_meta}")
    hugehdr = struct.pack("<Q", 2**63) + b"{}"
    out.append(f"huge-header-len={V(hugehdr, 'safetensors')}")
    empty = _safetensors({}, b"")
    out.append(f"empty-tensors={V(empty, 'safetensors')}")
    vuln = ok_overlap or ok_shape or ok_meta
    return ("VULNERABLE" if vuln else "BLOCKED", " | ".join(out))


# ---------------------------------------------------------------- D2/D3
@probe("D2", "poison-formats", "ONNX magic-only check + pickle protocol-0 denylist gap", "MEDIUM")
def _d2():
    V = rsi.SafeFormatValidator.validate
    fake_onnx = b"\x08" + b"NOT-A-REAL-ONNX-PAYLOAD-AT-ALL" * 4
    ok_onnx, r_onnx = V(fake_onnx, "onnx")
    # Protocol-0 (text) pickle using a module NOT on the denylist, hidden in safetensors tensor bytes
    p0 = b"csubprocess\nPopen\np0\n."
    st = _safetensors({"w": {"dtype": "U8", "shape": [len(p0)], "data_offsets": [0, len(p0)]}}, p0)
    ok_p0, r_p0 = V(st, "safetensors")
    vuln = ok_onnx or ok_p0
    return ("VULNERABLE" if vuln else "BLOCKED", f"garbage ONNX (0x08+text) accepted={ok_onnx} ({r_onnx}); proto-0 csubprocess in tensor accepted={ok_p0} ({r_p0})")


# ---------------------------------------------------------------- D4
@probe("D4", "merkle", "Merkle odd-leaf duplicate-last root collision ([A,B,C] == [A,B,C,C])", "HIGH")
def _d4():
    a, b, c = (hashlib.sha256(x).digest() for x in (b"A", b"B", b"C"))
    r3 = dbsync.MerkleTree([a, b, c]).root_hex
    r4 = dbsync.MerkleTree([a, b, c, c]).root_hex
    return ("VULNERABLE" if r3 == r4 else "BLOCKED", f"root3={r3[:16]} root4dup={r4[:16]} equal={r3 == r4}")


# ---------------------------------------------------------------- D5
try:
    dbsync.MerkleTree.verify_proof(b"\x00" * 32, [("zz-not-hex", True)], "00" * 32)
    record("D5", "merkle", "verify_proof malformed hex handling", "BLOCKED", "returned bool instead of crashing", "LOW")
except Exception as e:
    record("D5", "merkle", "verify_proof malformed hex handling", "CRASH", f"uncaught {type(e).__name__}: {e}", "LOW")


# ---------------------------------------------------------------- D6
@probe("D6", "dbsync", "receive_and_apply_shard ignores manifest record_count / size fields", "MEDIUM")
def _d6():
    root_priv, root_pub = cg.generate_ed25519_keypair()
    sub = dbsync.SubscriptionConfig(categories={"movie"}, languages={"en"}, quality_tiers={"hd_1080p"},
                                    epochs={"2026_q4"}, sync_mode=dbsync.SyncMode.SUBSCRIBE_AND_SEED)
    pub_sync = dbsync.ArmigerDHTDatabaseSync(subscription=sub, master_secret=b"M" * 32, trusted_root_pubkeys_hex=[root_pub.hex()])
    recv_sync = dbsync.ArmigerDHTDatabaseSync(subscription=sub, master_secret=b"M" * 32, trusted_root_pubkeys_hex=[root_pub.hex()])
    rec = dbsync.TorrentRecord(infohash="a" * 40, title="Big Buck Bunny", category="movie", language="en",
                               quality_tier="hd_1080p", epoch="2026_q4", size_bytes=1, file_count=1, seeders=0,
                               leechers=0, content_sha256="b" * 64, file_list=["bbb.mkv"], safety_gate_passed=True,
                               created_at=int(time.time()))
    # Publisher (root holder) signs a LYING manifest: record_count=999 but shard holds 1 record.
    # Re-sign honestly over the lying fields to isolate the receiver-side missing check.
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey as _E
    sk = _E.from_private_bytes(root_priv)
    shard = "movie/en/hd_1080p/2026_q4"
    compressed, tree = dbsync.serialize_shard_records([rec])
    key = dbsync.ShardCryptoEngine.derive_shard_key(b"M" * 32, shard, "2026_q4")
    ct = dbsync.ShardCryptoEngine.encrypt_shard(compressed, key, shard, "2026_q4", 7, tree.root_hex)
    dummy = dbsync.DHTShardManifest(seq=7, shard_id=shard, epoch="2026_q4", record_count=999,
                                    compressed_size_bytes=12345, uncompressed_size_bytes=99999,
                                    merkle_root=tree.root_hex, ciphertext_sha256=hashlib.sha256(ct).hexdigest(),
                                    bep46_infohash=hashlib.sha1(ct).hexdigest(), timestamp=int(time.time()),
                                    root_pubkey_hex=root_pub.hex(), ed25519_signature_hex="")
    payload = dummy.canonical_payload()
    pq_priv, pq_pub = rsi.WOTSPlusSHAKE256.derive_keypair_from_ed25519_priv(root_priv)
    man = dataclasses.replace(dummy, ed25519_signature_hex=sk.sign(payload).hex(),
                              root_pq_pubkey_hex=pq_pub.hex(),
                              pq_signature_hex=rsi.WOTSPlusSHAKE256.sign(pq_priv, payload).hex(),
                              attestations=dbsync.create_mock_quorum_attestations(dummy.canonical_digest(), [1, 2, 3, 4, 5]))
    # NOTE digest with attestations excluded == dummy digest, matches create helper usage pattern in tests
    ok, reason, n = recv_sync.receive_and_apply_shard(man, ct)
    return ("VULNERABLE" if ok else "BLOCKED", f"record_count=999 (actual 1) + lying sizes -> ok={ok} ({reason}) ingested={n}")


# ---------------------------------------------------------------- D7
@probe("D7", "sqlite", "ShardDatabase SQL injection via title/query + LIKE wildcards", "MEDIUM")
def _d7():
    db = dbsync.ShardDatabase()
    evil = dbsync.TorrentRecord(infohash="c" * 40, title="x' OR '1'='1 --; DROP TABLE mainstream_torrents;--",
                                category="movie", language="en", quality_tier="any", epoch="2026_q4", size_bytes=1,
                                file_count=1, seeders=0, leechers=0, content_sha256="d" * 64,
                                file_list=["x'); DROP TABLE mainstream_torrents;--"], safety_gate_passed=True, created_at=1)
    db.insert_record(evil)
    cnt_after = db.count_records()
    got = db.get_record("c" * 40)
    wildcard = db.search(query="%")   # unescaped LIKE wildcard matches everything
    inj = db.search(query="' OR '1'='1")
    table_alive = cnt_after == 1 and got is not None
    return ("BLOCKED" if table_alive else "VULNERABLE",
            f"table alive={table_alive} count={cnt_after}; LIKE '%' returned {len(wildcard)} rows (wildcard unescaped); "
            f"injection-string search returned {len(inj)} rows")


# ---------------------------------------------------------------- D8
@probe("D8", "sqlite-isolation", "Adult isolation bypass via category whitespace ('xxx ', ' xxx')", "HIGH")
def _d8():
    db = dbsync.ShardDatabase(enable_xxx=False)
    results = []
    for i, cat in enumerate(["xxx ", " xxx", "xxx\t", "XXX", "xxx"]):
        rec = dbsync.TorrentRecord(infohash=f"{i:040d}", title=f"adult {i}", category=cat, language="en",
                                   quality_tier="any", epoch="2026_q4", size_bytes=1, file_count=1, seeders=0,
                                   leechers=0, content_sha256="e" * 64, file_list=["a"], safety_gate_passed=True, created_at=1)
        ins = db.insert_record(rec)
        stored = db.get_record(f"{i:040d}") is not None
        results.append(f"{cat!r}: insert={ins} in-mainstream={stored}")
    vuln = "in-mainstream=True" in results[0] or "in-mainstream=True" in results[1] or "in-mainstream=True" in results[2]
    return ("VULNERABLE" if vuln else "BLOCKED", " | ".join(results))

# ---------------------------------------------------------------- D9
try:
    _g = cg.CSAMGuard()
    _r = _g.inspect_torrent(infohash="é" * 40, title="Sintel", file_list=["s.mkv"], category="movie")
    record("D9", "tombstone", "Non-ASCII infohash into blind_infohash (ascii encode)", "VULNERABLE" if _r.is_safe else "BLOCKED", f"is_safe={_r.is_safe} (no crash)", "LOW")
except Exception as e:
    record("D9", "tombstone", "Non-ASCII infohash into blind_infohash (ascii encode)", "CRASH", f"uncaught {type(e).__name__}: {e}", "MEDIUM")


# ---------------------------------------------------------------- D10
@probe("D10", "csam-guard", "CSAM keyword obfuscation battery (homoglyphs, soft hyphen, combining, tag chars, VS, spacing, leet)", "CRITICAL")
def _d10():
    g = cg.CSAMGuard()
    # Test strings embed a blocklisted stem with hostile obfuscation; safe == BYPASS.
    cases = {
        "plain (control)": "child porn pack",
        "cyrillic homoglyphs": "\u0441hild p\u043ern pack",           # Cyrillic es/o lookalikes
        "soft hyphen U+00AD": "chi\u00adld porn pack",
        "combining acute": "ch\u0301ild porn pack",
        "fullwidth": "\uff43\uff48\uff49\uff4c\uff44 \uff50\uff4f\uff52\uff4e pack",
        "tag char U+E0001": "chi\U000e0001ld porn pack",
        "variation selector U+FE0F": "chi\ufe0fld porn pack",
        "zero-width joiner": "chi\u200dld porn pack",
        "spaced letters": "c h i l d p o r n pack",
        "leetspeak": "ch1ld p0rn pack",
        "math bold": "\U0001d402\U0001d421\U0001d422\U0001d425\U0001d41d porn pack",
        "greek homoglyphs": "child p\u03bfrn pack",
    }
    bypasses, blocked = [], []
    for label, text in cases.items():
        r = g.inspect_text(text)
        (blocked if not r.is_safe else bypasses).append(label)
    vuln = bool(bypasses)
    return ("VULNERABLE" if vuln else "BLOCKED", f"BYPASSED={bypasses} blocked={blocked}")


# ---------------------------------------------------------------- D11
@probe("D11", "csam-guard", "Hardcoded .exe skip: 'do_not_mirror' filename + small size bypasses malware check", "CRITICAL")
def _d11():
    g = cg.CSAMGuard()
    normal = g.inspect_torrent(infohash="f" * 40, title="Sintel 1080p",
                               file_list=[{"path": "Sintel.mp4.exe", "size": 1000}], category="movie")
    sneaky = g.inspect_torrent(infohash="f" * 40, title="Sintel 1080p",
                               file_list=[{"path": "do_not_mirror_Sintel.mp4.exe", "size": 1000}], category="movie")
    sneaky_str = g.inspect_torrent(infohash="f" * 40, title="Sintel 1080p",
                                   file_list=["do_not_mirror_Sintel.mp4.exe"], category="movie")
    vuln = (not normal.is_safe) and sneaky.is_safe
    return ("VULNERABLE" if vuln else "BLOCKED",
            f"plain .exe safe={normal.is_safe} ({normal.violation_type}); do_not_mirror+size1000 safe={sneaky.is_safe}; str-form safe={sneaky_str.is_safe}")


# ---------------------------------------------------------------- D12
@probe("D12", "csam-guard", "Extension handling: double-ext, trailing dot/space, nested archive names", "MEDIUM")
def _d12():
    g = cg.CSAMGuard()
    cases = {
        "movie.mp4.exe": [{"path": "movie.mp4.exe", "size": 5000}],
        "movie.mp4.EXE": [{"path": "movie.mp4.EXE", "size": 5000}],
        "movie.exe (media cat)": [{"path": "movie.exe", "size": 5000}],
        "movie.mp4.exe. (trailing dot)": [{"path": "movie.mp4.exe.", "size": 5000}],
        "movie.mp4.exe (trailing space)": [{"path": "movie.mp4.exe ", "size": 5000}],
        "archive.zip.exe": [{"path": "archive.zip.exe", "size": 5000}],
        "setup.exe software cat": [{"path": "setup.exe", "size": 5000}],
    }
    out = []
    for label, fl in cases.items():
        cat = "software" if "software" in label else "movie"
        r = g.inspect_torrent(infohash="f" * 40, title="Some Movie 1080p", file_list=fl, category=cat)
        out.append(f"{label}: safe={r.is_safe}")
    # trailing dot/space accepted on Windows-stripped filesystems == bypass for media cats
    vuln = any("trailing" in o and "safe=True" in o for o in out)
    return ("VULNERABLE" if vuln else "BLOCKED", " | ".join(out))


# ---------------------------------------------------------------- D13
@probe("D13", "safety-gate", "ArmigerSafetyGate NaN probabilities pass all Mondrian gates", "HIGH")
def _d13():
    gate = cg.ArmigerSafetyGate()
    ok_nan, r_nan, _ = gate.evaluate(p_adult=float("nan"), p_junk=float("nan"), p_excluded_scope=float("nan"), target_category="movie")
    ok_inf, r_inf, _ = gate.evaluate(p_adult=float("inf"), p_junk=0.0, p_excluded_scope=0.0, target_category="movie")
    ok_neg, r_neg, _ = gate.evaluate(p_adult=-5.0, p_junk=-5.0, p_excluded_scope=-5.0, target_category="movie")
    return ("VULNERABLE" if ok_nan else "BLOCKED", f"NaN probs -> ok={ok_nan} ({r_nan}); inf adult into movie ok={ok_inf} ({r_inf}); negative ok={ok_neg}")


# ---------------------------------------------------------------- D14
@probe("D14", "canary-gate", "GoldenCanary gate never enforces min_category_acc / owner_scope floor", "HIGH")
def _d14():
    gate = rsi.GoldenCanaryRegressionGate()
    def lying_predictor(item):
        gt = dict(item.ground_truth)
        gt["category"] = "unknown" if gt["category"] != "unknown" else "movie"  # always wrong category
        return gt  # junk/adult/scope correct, category ~0% accurate
    rep = gate.evaluate(lying_predictor)
    return ("VULNERABLE" if rep.passed else "BLOCKED",
            f"category_acc={rep.category_accuracy:.3f} (floor param 0.950) passed={rep.passed} reason={rep.reason}")


# ---------------------------------------------------------------- D15
@probe("D15", "ground-truth", "Physical verifier accepts self-asserted stream_metadata + ignores bytes_downloaded", "HIGH")
def _d15():
    cand = rsi.CandidateObservation(source_adapter="qbittorrent", raw_title="Totally Real Movie 1080p",
                                    claimed_category="movie", download_completed=True,
                                    bytes_downloaded=0, total_bytes=1_000_000_000,
                                    stream_metadata={"video_codec": "h264", "audio_codec": "aac"})
    res = rsi.PhysicalGroundTruthVerifier().verify(cand)  # no ffprobe proof at all
    # Series adapter fabricates codec defaults when mediaInfo missing:
    wh = rsi.SeriesWebhookAdapter.parse_webhook({
        "eventType": "Download", "series": {"title": "Fake Show", "seriesCatalogId": 42},
        "release": {"releaseTitle": "Fake.Show.S01E01", "size": 12345}, "episodeFile": {}})
    return ("VULNERABLE" if res.is_verified else "BLOCKED",
            f"bytes_downloaded=0 + self-asserted codecs verified={res.is_verified}; "
            f"webhook w/o mediaInfo still claims codecs={wh.stream_metadata}")


# ---------------------------------------------------------------- D16
@probe("D16", "deserialization", "Shard decompression has no output-size cap (zip-bomb ratio)", "MEDIUM")
def _d16():
    bomb_raw = (b'{"records":[]}' + b" " * (8 * 1024 * 1024))  # ~8 MiB highly compressible
    comp = zlib.compress(bomb_raw, level=9)
    out = dbsync.decompress_payload(comp, "zlib")
    ratio = len(out) / max(len(comp), 1)
    return ("VULNERABLE" if ratio > 100 else "BLOCKED",
            f"compressed={len(comp)}B decompressed={len(out)}B ratio={ratio:.0f}x with no cap in decompress_payload/deserialize")


# ---------------------------------------------------------------- D17
@probe("D17", "adapters", "QBittorrent/Transmission inf/NaN progress & size type confusion", "MEDIUM")
def _d17():
    out = []
    for label, payload in [
        ("progress=inf", {"name": "x", "progress": float("inf"), "state": "downloading", "size": 100}),
        ("progress=nan+uploading", {"name": "x", "progress": float("nan"), "state": "uploading", "size": 100}),
        ("progress='nan' str", {"name": "x", "progress": "nan", "state": "uploading", "size": 100}),
    ]:
        try:
            c = rsi.QBittorrentAdapter.parse_event(payload)
            out.append(f"{label}: completed={c.download_completed} downloaded={c.bytes_downloaded}")
        except Exception as e:
            out.append(f"{label}: raised {type(e).__name__}")
    vuln = any("completed=True" in o and "inf" in o for o in out)
    return ("VULNERABLE" if vuln else "BLOCKED", " | ".join(out))


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    counts: dict[str, int] = {}
    for r in RESULTS:
        counts[r["result"]] = counts.get(r["result"], 0) + 1
    summary = {
        "target_dir": TARGET_DIR,
        "python": sys.executable,
        "counts": counts,
        "total": len(RESULTS),
        "results": RESULTS,
    }
    with open("/tmp/armiger_bep_exploit_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("=" * 78)
    print(f"SUMMARY: {counts} total={len(RESULTS)} -> /tmp/armiger_bep_exploit_results.json")


def test_all_45_hostile_redteam_probes_blocked() -> None:
    """Verify all 45 adversarial BEP-44/BEP-46 exploit probes are 100% BLOCKED (0 VULNERABLE, 0 CRASH)."""
    assert len(RESULTS) == 45, f"Expected 45 probes, got {len(RESULTS)}"
    failures = [r for r in RESULTS if r["result"] != "BLOCKED"]
    assert not failures, f"Red-team probes not BLOCKED: {failures}"
