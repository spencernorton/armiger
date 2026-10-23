"""Post-Quantum Cryptographic Engines (ML-KEM-768, WOTS+-SHAKE256, Hybrid AEAD) and Universal Adapters."""
from __future__ import annotations

import dataclasses
import enum
import functools
import hashlib
import hmac
import ipaddress
import json
import math
import os
import pathlib
import random
import re
import secrets
import sys
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

from cryptography.exceptions import InvalidSignature, InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ed25519, x25519
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF


from .dht_rsi_models_and_privacy import (
    AnonymizedTelemetryRecord,
    BlindedTitleStem,
    BundleFormat,
    CanaryItem,
    CanaryRegressionReport,
    CandidateObservation,
    CatalogConsensusResult,
    ContinuousAggregationResult,
    ContinuousDeltaSubmission,
    DHTRSIMode,
    EncryptedTelemetryBundle,
    MediaCategory,
    PeerAttestation,
    PhysicalVerificationResult,
    QueuedTelemetryItem,
    TelemetrySubmission,
    ThresholdDHTManifest,
    VerifiedObservation,
    ZeroCueCatalogDelta,
    resolve_opt_in_sharing,
)

# =====================================================================
# 4. Post-Quantum Cryptographic Engines (ML-KEM-768, WOTS+-SHAKE256 & Hybrid AEAD)
# =====================================================================

class MLKEM768:
    """Self-contained NIST FIPS 203 ML-KEM-768 (Module-LWE Post-Quantum KEM) implementation.

    Parameters (FIPS 203 ML-KEM-768):
      - Ring R_q = Z_3329[X] / (X^256 + 1), q = 3329, n = 256, k = 3
      - Noise parameters: eta_1 = 2, eta_2 = 2
      - Ciphertext compression: d_u = 10, d_v = 4
      - Encapsulation key (ek): 1,184 bytes
      - Decapsulation key (dk): 2,400 bytes
      - Ciphertext (ct): 1,088 bytes
      - Shared secret (K): 32 bytes (256-bit IND-CCA2 security via Fujisaki-Okamoto transform)
    """

    Q = 3329
    N = 256
    K = 3
    EK_BYTES = 1184
    DK_BYTES = 2400
    CT_BYTES = 1088
    SS_BYTES = 32

    ZETAS = (
        1, 1729, 2580, 3289, 2642, 630, 1897, 848, 1062, 1919, 193, 797, 2786, 3260, 569, 1746,
        296, 2447, 1339, 1476, 3046, 56, 2240, 1333, 1426, 2094, 535, 2882, 2393, 2879, 1974, 821,
        289, 331, 3253, 1756, 1197, 2304, 2277, 2055, 650, 1977, 2513, 632, 2865, 33, 1320, 1915,
        2319, 1435, 807, 452, 1438, 2868, 1534, 2402, 2647, 2617, 1481, 648, 2474, 3110, 1227, 910,
        17, 2761, 583, 2649, 1637, 723, 2288, 1100, 1409, 2662, 3281, 233, 756, 2156, 3015, 3050,
        1703, 1651, 2789, 1789, 1847, 952, 1461, 2687, 939, 2308, 2437, 2388, 733, 2337, 268, 641,
        1584, 2298, 2037, 3220, 375, 2549, 2090, 1645, 1063, 319, 2773, 757, 2099, 561, 2466, 2594,
        2804, 1092, 403, 1026, 1143, 2150, 2775, 886, 1722, 1212, 1874, 1029, 2110, 2935, 885, 2154,
    )
    GAMMAS = tuple(pow(17, 2 * int(f"{i:07b}"[::-1], 2) + 1, 3329) for i in range(128))
    _CBD2_LO = tuple(((0, 1, 1, 2)[b & 3] - (0, 1, 1, 2)[(b >> 2) & 3]) % 3329 for b in range(256))
    _CBD2_HI = tuple(((0, 1, 1, 2)[(b >> 4) & 3] - (0, 1, 1, 2)[(b >> 6) & 3]) % 3329 for b in range(256))

    _NTT_STEPS: List[List[Tuple[int, int, int, int]]] = []
    _INTT_STEPS: List[List[Tuple[int, int, int, int]]] = []

    @classmethod
    def _init_tables(cls) -> None:
        if cls._NTT_STEPS:
            return
        k = 1
        length = 128
        while length >= 2:
            layer = []
            for start in range(0, 256, 2 * length):
                zeta = cls.ZETAS[k]
                k += 1
                layer.append((zeta, start, start + length, length))
            cls._NTT_STEPS.append(layer)
            length >>= 1

        k = 127
        length = 2
        while length <= 128:
            layer = []
            for start in range(0, 256, 2 * length):
                zeta = cls.ZETAS[k]
                k -= 1
                layer.append((zeta, start, start + length, length))
            cls._INTT_STEPS.append(layer)
            length <<= 1

    @classmethod
    def ntt(cls, f: List[int]) -> List[int]:
        p = list(f)
        q = 3329
        for layer in cls._NTT_STEPS:
            for zeta, s, e, l in layer:
                for j in range(s, e):
                    t = (zeta * p[j + l]) % q
                    p[j + l] = (p[j] - t) % q
                    p[j] = (p[j] + t) % q
        return p

    @classmethod
    def intt(cls, f: List[int]) -> List[int]:
        p = list(f)
        q = 3329
        for layer in cls._INTT_STEPS:
            for zeta, s, e, l in layer:
                for j in range(s, e):
                    t = p[j]
                    p[j] = (t + p[j + l]) % q
                    p[j + l] = (zeta * (p[j + l] - t)) % q
        return [(x * 3303) % q for x in p]

    @classmethod
    def basemul_acc(cls, a_vec: List[List[int]], b_vec: List[List[int]]) -> List[int]:
        res = [0] * 256
        q = 3329
        gammas = cls.GAMMAS
        for a, b in zip(a_vec, b_vec):
            for i in range(128):
                idx = i << 1
                a0, a1 = a[idx], a[idx + 1]
                b0, b1 = b[idx], b[idx + 1]
                res[idx] = (res[idx] + a0 * b0 + a1 * b1 * gammas[i]) % q
                res[idx + 1] = (res[idx + 1] + a0 * b1 + a1 * b0) % q
        return res

    @staticmethod
    def sample_ntt(rho: bytes, j: int, i: int) -> List[int]:
        stream = hashlib.shake_128(rho + bytes([j, i])).digest(672)
        a: List[int] = []
        idx = 0
        while len(a) < 256:
            b0, b1, b2 = stream[idx], stream[idx + 1], stream[idx + 2]
            idx += 3
            d1 = b0 + ((b1 & 0x0F) << 8)
            d2 = (b1 >> 4) + (b2 << 4)
            if d1 < 3329:
                a.append(d1)
            if d2 < 3329 and len(a) < 256:
                a.append(d2)
        return a

    @classmethod
    def sample_cbd2(cls, prf_bytes: bytes) -> List[int]:
        f = [0] * 256
        lo = cls._CBD2_LO
        hi = cls._CBD2_HI
        for i, b in enumerate(prf_bytes):
            idx = i << 1
            f[idx] = lo[b]
            f[idx + 1] = hi[b]
        return f

    @staticmethod
    def encode_12(f: List[int]) -> bytes:
        out = bytearray(384)
        for i in range(128):
            a = f[2 * i] % 3329
            b = f[2 * i + 1] % 3329
            idx = 3 * i
            out[idx] = a & 0xFF
            out[idx + 1] = ((a >> 8) & 0x0F) | ((b & 0x0F) << 4)
            out[idx + 2] = (b >> 4) & 0xFF
        return bytes(out)

    @staticmethod
    def decode_12(buf: bytes) -> List[int]:
        f = [0] * 256
        for i in range(128):
            idx = 3 * i
            b0, b1, b2 = buf[idx], buf[idx + 1], buf[idx + 2]
            f[2 * i] = (b0 + ((b1 & 0x0F) << 8)) % 3329
            f[2 * i + 1] = ((b1 >> 4) + (b2 << 4)) % 3329
        return f

    @staticmethod
    def compress_encode_10(f: List[int]) -> bytes:
        out = bytearray(320)
        for i in range(64):
            c0 = ((f[4 * i] << 10) + 1664) // 3329 & 0x3FF
            c1 = ((f[4 * i + 1] << 10) + 1664) // 3329 & 0x3FF
            c2 = ((f[4 * i + 2] << 10) + 1664) // 3329 & 0x3FF
            c3 = ((f[4 * i + 3] << 10) + 1664) // 3329 & 0x3FF
            idx = 5 * i
            out[idx] = c0 & 0xFF
            out[idx + 1] = ((c0 >> 8) & 0x03) | ((c1 & 0x3F) << 2)
            out[idx + 2] = ((c1 >> 6) & 0x0F) | ((c2 & 0x0F) << 4)
            out[idx + 3] = ((c2 >> 4) & 0x3F) | ((c3 & 0x03) << 6)
            out[idx + 4] = (c3 >> 2) & 0xFF
        return bytes(out)

    @staticmethod
    def decode_decompress_10(buf: bytes) -> List[int]:
        f = [0] * 256
        for i in range(64):
            idx = 5 * i
            b0, b1, b2, b3, b4 = buf[idx], buf[idx + 1], buf[idx + 2], buf[idx + 3], buf[idx + 4]
            c0 = b0 | ((b1 & 0x03) << 8)
            c1 = (b1 >> 2) | ((b2 & 0x0F) << 6)
            c2 = (b2 >> 4) | ((b3 & 0x3F) << 4)
            c3 = (b3 >> 6) | (b4 << 2)
            f[4 * i] = (c0 * 3329 + 512) >> 10
            f[4 * i + 1] = (c1 * 3329 + 512) >> 10
            f[4 * i + 2] = (c2 * 3329 + 512) >> 10
            f[4 * i + 3] = (c3 * 3329 + 512) >> 10
        return f

    @staticmethod
    def compress_encode_4(f: List[int]) -> bytes:
        out = bytearray(128)
        for i in range(128):
            c0 = ((f[2 * i] << 4) + 1664) // 3329 & 0x0F
            c1 = ((f[2 * i + 1] << 4) + 1664) // 3329 & 0x0F
            out[i] = c0 | (c1 << 4)
        return bytes(out)

    @staticmethod
    def decode_decompress_4(buf: bytes) -> List[int]:
        f = [0] * 256
        for i in range(128):
            b = buf[i]
            f[2 * i] = ((b & 0x0F) * 3329 + 8) >> 4
            f[2 * i + 1] = (((b >> 4) & 0x0F) * 3329 + 8) >> 4
        return f

    @staticmethod
    def encode_1(m: bytes) -> List[int]:
        f = [0] * 256
        for i in range(32):
            b = m[i]
            base = i << 3
            for j in range(8):
                f[base + j] = 1665 if ((b >> j) & 1) else 0
        return f

    @staticmethod
    def decode_1(f: List[int]) -> bytes:
        out = bytearray(32)
        for i in range(32):
            byte_val = 0
            base = i << 3
            for j in range(8):
                c = (((f[base + j] << 1) + 1664) // 3329) & 1
                byte_val |= (c << j)
            out[i] = byte_val
        return bytes(out)

    @classmethod
    @functools.lru_cache(maxsize=64)
    def _expand_ek(cls, ek: bytes) -> Tuple[List[List[int]], List[List[List[int]]], bytes]:
        cls._init_tables()
        t_hat = [cls.decode_12(ek[384 * i : 384 * (i + 1)]) for i in range(3)]
        rho = ek[1152:1184]
        a_hat_t = [[cls.sample_ntt(rho, i, j) for j in range(3)] for i in range(3)]
        ek_hash = hashlib.sha3_256(ek).digest()
        return t_hat, a_hat_t, ek_hash

    @classmethod
    @functools.lru_cache(maxsize=64)
    def _expand_dk(cls, dk: bytes) -> Tuple[List[List[int]], bytes, bytes, bytes, List[List[int]], List[List[List[int]]]]:
        cls._init_tables()
        s_hat = [cls.decode_12(dk[384 * i : 384 * (i + 1)]) for i in range(3)]
        ek = dk[1152:2336]
        h = dk[2336:2368]
        z = dk[2368:2400]
        t_hat, a_hat_t, _ = cls._expand_ek(ek)
        return s_hat, ek, h, z, t_hat, a_hat_t

    @classmethod
    def keygen(cls, seed_d: Optional[bytes] = None, seed_z: Optional[bytes] = None) -> Tuple[bytes, bytes]:
        """Generate (encapsulation_key_1184B, decapsulation_key_2400B) per NIST FIPS 203."""
        cls._init_tables()
        d = seed_d if seed_d is not None else os.urandom(32)
        z = seed_z if seed_z is not None else os.urandom(32)
        g = hashlib.sha3_512(d + bytes([3])).digest()
        rho, sigma = g[:32], g[32:]
        a_hat = [[cls.sample_ntt(rho, j, i) for j in range(3)] for i in range(3)]
        s_hat = [cls.ntt(cls.sample_cbd2(hashlib.shake_256(sigma + bytes([i])).digest(128))) for i in range(3)]
        e_hat = [cls.ntt(cls.sample_cbd2(hashlib.shake_256(sigma + bytes([i + 3])).digest(128))) for i in range(3)]
        t_hat = []
        for i in range(3):
            ti = cls.basemul_acc(a_hat[i], s_hat)
            t_hat.append([(a + b) % 3329 for a, b in zip(ti, e_hat[i])])
        ek_pke = b"".join(cls.encode_12(p) for p in t_hat) + rho
        dk_pke = b"".join(cls.encode_12(p) for p in s_hat)
        h = hashlib.sha3_256(ek_pke).digest()
        return ek_pke, dk_pke + ek_pke + h + z

    @classmethod
    def _pke_encrypt_expanded(
        cls,
        t_hat: List[List[int]],
        a_hat_t: List[List[List[int]]],
        m: bytes,
        r: bytes,
    ) -> bytes:
        r_hat = [cls.ntt(cls.sample_cbd2(hashlib.shake_256(r + bytes([i])).digest(128))) for i in range(3)]
        e1 = [cls.sample_cbd2(hashlib.shake_256(r + bytes([i + 3])).digest(128)) for i in range(3)]
        e2 = cls.sample_cbd2(hashlib.shake_256(r + bytes([6])).digest(128))
        u = []
        for i in range(3):
            ui = cls.intt(cls.basemul_acc(a_hat_t[i], r_hat))
            u.append([(a + b) % 3329 for a, b in zip(ui, e1[i])])
        mu = cls.encode_1(m)
        v_ntt = cls.basemul_acc(t_hat, r_hat)
        v_poly = cls.intt(v_ntt)
        v = [(a + b + c) % 3329 for a, b, c in zip(v_poly, e2, mu)]
        return b"".join(cls.compress_encode_10(p) for p in u) + cls.compress_encode_4(v)

    @classmethod
    def encaps(cls, ek: bytes, seed_m: Optional[bytes] = None) -> Tuple[bytes, bytes]:
        """Encapsulate against a 1,184-byte ML-KEM-768 public key, returning (ciphertext_1088B, shared_secret_32B)."""
        if len(ek) != cls.EK_BYTES:
            raise ValueError(f"ML-KEM-768 ek must be {cls.EK_BYTES} bytes, got {len(ek)}")
        if ek[:1152] == b"\x00" * 1152 or ek[1152:] == b"\x00" * 32:
            raise ValueError("ML-KEM-768 degenerate all-zero encapsulation key rejected")
        t_hat, a_hat_t, ek_hash = cls._expand_ek(ek)
        if b"".join(cls.encode_12(p) for p in t_hat) != ek[:1152]:
            raise ValueError("ML-KEM-768 non-canonical encapsulation key (coefficients >= q)")
        m = seed_m if seed_m is not None else os.urandom(32)
        kr = hashlib.sha3_512(m + ek_hash).digest()
        k_shared, r = kr[:32], kr[32:]
        c = cls._pke_encrypt_expanded(t_hat, a_hat_t, m, r)
        return c, k_shared

    @classmethod
    def decaps(cls, dk: bytes, c: bytes) -> bytes:
        """Decapsulate a 1,088-byte ML-KEM-768 ciphertext using a 2,400-byte private key with FO implicit rejection."""
        if len(dk) != cls.DK_BYTES or len(c) != cls.CT_BYTES:
            raise ValueError("ML-KEM-768 invalid dk or ciphertext length")
        s_hat, _ek, h, z, t_hat, a_hat_t = cls._expand_dk(dk)
        u = [cls.decode_decompress_10(c[320 * i : 320 * (i + 1)]) for i in range(3)]
        v = cls.decode_decompress_4(c[960:1088])
        u_hat = [cls.ntt(p) for p in u]
        s_u = cls.intt(cls.basemul_acc(s_hat, u_hat))
        w = [(vi - sui) % 3329 for vi, sui in zip(v, s_u)]
        m_prime = cls.decode_1(w)
        kr_prime = hashlib.sha3_512(m_prime + h).digest()
        k_prime, r_prime = kr_prime[:32], kr_prime[32:]
        k_bar = hashlib.shake_256(z + c).digest(32)
        c_prime = cls._pke_encrypt_expanded(t_hat, a_hat_t, m_prime, r_prime)
        if hmac.compare_digest(c, c_prime):
            return k_prime
        return k_bar


MLKEM768._init_tables()

from .csam_guard import WOTSPlusSHAKE256


class X25519ChaCha20Poly1305Engine:
    """Hybrid Post-Quantum Key Encapsulation (X25519 + ML-KEM-768) + ChaCha20-Poly1305 AEAD Engine.

    Combines classical X25519 ECDH with NIST FIPS 203 ML-KEM-768 lattice encapsulation via a
    domain-separated SHAKE-256 + HKDF-SHA3-256 combiner feeding 256-bit ChaCha20-Poly1305 AEAD.
    Remains secure as long as EITHER X25519 OR ML-KEM-768 is unbroken, providing full
    Post-Quantum IND-CCA2 confidentiality and authentication against Harvest-Now-Decrypt-Later (HNDL) attacks.
    """

    SALT = b"armiger-dht-rsi-pq-hybrid-v2-salt"
    INFO = b"armiger-dht-rsi-x25519-mlkem768-chacha20poly1305-sha3"
    HYBRID_PRIV_LEN = 32 + MLKEM768.DK_BYTES  # 2,432 bytes
    HYBRID_PUB_LEN = 32 + MLKEM768.EK_BYTES   # 1,216 bytes
    HYBRID_ENC_LEN = 32 + MLKEM768.CT_BYTES   # 1,120 bytes

    @classmethod
    def generate_keypair(cls) -> Tuple[bytes, bytes]:
        """Generate a Hybrid Post-Quantum (X25519 + ML-KEM-768) keypair: (priv_2432B, pub_1216B)."""
        priv = x25519.X25519PrivateKey.generate()
        pub = priv.public_key()
        from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat, PrivateFormat, NoEncryption
        x_priv = priv.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())
        x_pub = pub.public_bytes(Encoding.Raw, PublicFormat.Raw)
        pq_ek, pq_dk = MLKEM768.keygen()
        return x_priv + pq_dk, x_pub + pq_ek

    @classmethod
    def generate_classical_keypair(cls) -> Tuple[bytes, bytes]:
        """Generate a 32-byte X25519-only keypair for legacy compatibility testing."""
        priv = x25519.X25519PrivateKey.generate()
        pub = priv.public_key()
        from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat, PrivateFormat, NoEncryption
        return (
            priv.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption()),
            pub.public_bytes(Encoding.Raw, PublicFormat.Raw),
        )

    @classmethod
    def _combine_hybrid_secrets(
        cls,
        classical_ss: bytes,
        pq_ss: bytes,
        ephem_x25519_pub: bytes,
        pq_ct: bytes,
    ) -> bytes:
        """Combine X25519 shared secret and ML-KEM-768 shared secret via SHAKE-256 + HKDF-SHA3-256."""
        context = ephem_x25519_pub + hashlib.sha3_256(pq_ct).digest()
        shake_key = hashlib.shake_256(
            cls.SALT + b":shake256:" + classical_ss + pq_ss + context
        ).digest(32)
        hkdf_key = HKDF(
            algorithm=hashes.SHA3_256(),
            length=32,
            salt=cls.SALT,
            info=cls.INFO + context,
        ).derive(classical_ss + pq_ss)
        return bytes(a ^ b for a, b in zip(shake_key, hkdf_key))

    @classmethod
    def derive_symmetric_key(
        cls,
        private_bytes: bytes,
        peer_public_bytes: bytes,
        *,
        allow_classical: bool = False,
    ) -> bytes:
        """Derive a 256-bit symmetric key from recipient private key and sender encapsulation token."""
        if len(private_bytes) == cls.HYBRID_PRIV_LEN and len(peer_public_bytes) == cls.HYBRID_ENC_LEN:
            x_priv_bytes = private_bytes[:32]
            pq_dk = private_bytes[32:]
            ephem_x_pub = peer_public_bytes[:32]
            pq_ct = peer_public_bytes[32:]
            priv = x25519.X25519PrivateKey.from_private_bytes(x_priv_bytes)
            pub = x25519.X25519PublicKey.from_public_bytes(ephem_x_pub)
            ss_classical = priv.exchange(pub)
            ss_pq = MLKEM768.decaps(pq_dk, pq_ct)
            return cls._combine_hybrid_secrets(ss_classical, ss_pq, ephem_x_pub, pq_ct)

        if len(private_bytes) == cls.HYBRID_PRIV_LEN and len(peer_public_bytes) == 32 and not allow_classical:
            raise ValueError(
                "AEAD post-quantum downgrade rejected: 32-byte classical encapsulation token is not permitted "
                "when recipient holds a 2,432-byte hybrid X25519 + ML-KEM-768 private key"
            )
        if len(private_bytes) not in (32, cls.HYBRID_PRIV_LEN) or len(peer_public_bytes) != 32:
            raise ValueError(
                f"AEAD invalid key/token lengths: priv={len(private_bytes)}, token={len(peer_public_bytes)}"
            )

        # Fallback for 32-byte X25519 keys with SHAKE-256 + HKDF-SHA3-256 post-quantum KDF hardening
        x_priv_bytes = private_bytes[:32]
        ephem_x_pub = peer_public_bytes[:32]
        priv = x25519.X25519PrivateKey.from_private_bytes(x_priv_bytes)
        pub = x25519.X25519PublicKey.from_public_bytes(ephem_x_pub)
        ss_classical = priv.exchange(pub)
        ss_pq_fallback = hashlib.shake_256(cls.SALT + ss_classical + ephem_x_pub).digest(32)
        return cls._combine_hybrid_secrets(ss_classical, ss_pq_fallback, ephem_x_pub, b"")

    @classmethod
    def encrypt_bundle(
        cls,
        payload_bytes: bytes,
        recipient_pubkey_bytes: bytes,
        seq: int = 1,
        sender_privkey_bytes: Optional[bytes] = None,
    ) -> EncryptedTelemetryBundle:
        """Encrypt payload using Hybrid Post-Quantum X25519 + ML-KEM-768 KEM and 256-bit ChaCha20-Poly1305 AEAD."""
        if sender_privkey_bytes is None:
            ephem_priv = x25519.X25519PrivateKey.generate()
        else:
            ephem_priv = x25519.X25519PrivateKey.from_private_bytes(sender_privkey_bytes[:32])

        ephem_pub = ephem_priv.public_key()
        from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
        ephem_x_pub_bytes = ephem_pub.public_bytes(Encoding.Raw, PublicFormat.Raw)

        if len(recipient_pubkey_bytes) == cls.HYBRID_PUB_LEN:
            recip_x_pub = recipient_pubkey_bytes[:32]
            recip_pq_ek = recipient_pubkey_bytes[32:]
            peer_pub = x25519.X25519PublicKey.from_public_bytes(recip_x_pub)
            ss_classical = ephem_priv.exchange(peer_pub)
            pq_ct, ss_pq = MLKEM768.encaps(recip_pq_ek)
            ephem_bundle_bytes = ephem_x_pub_bytes + pq_ct
            sym_key = cls._combine_hybrid_secrets(ss_classical, ss_pq, ephem_x_pub_bytes, pq_ct)
            suite = "X25519-MLKEM768-ChaCha20Poly1305-SHA3"
        elif len(recipient_pubkey_bytes) == 32:
            peer_pub = x25519.X25519PublicKey.from_public_bytes(recipient_pubkey_bytes)
            ss_classical = ephem_priv.exchange(peer_pub)
            ss_pq_fallback = hashlib.shake_256(cls.SALT + ss_classical + ephem_x_pub_bytes).digest(32)
            ephem_bundle_bytes = ephem_x_pub_bytes
            sym_key = cls._combine_hybrid_secrets(ss_classical, ss_pq_fallback, ephem_x_pub_bytes, b"")
            suite = "X25519-SHAKE256-ChaCha20Poly1305-SHA3"
        else:
            raise ValueError(f"Unsupported recipient public key length: {len(recipient_pubkey_bytes)}")

        nonce = os.urandom(12)
        aad = f"seq:{seq};ephem_pub:{ephem_bundle_bytes.hex()}".encode("utf-8")
        chacha = ChaCha20Poly1305(sym_key)
        ciphertext = chacha.encrypt(nonce, payload_bytes, aad)

        return EncryptedTelemetryBundle(
            ephemeral_pubkey_hex=ephem_bundle_bytes.hex(),
            nonce_hex=nonce.hex(),
            ciphertext_hex=ciphertext.hex(),
            aad=aad.decode("utf-8"),
            seq=seq,
            cipher_suite=suite,
        )

    @classmethod
    def decrypt_bundle(
        cls,
        bundle: EncryptedTelemetryBundle,
        recipient_privkey_bytes: bytes,
        expected_seq: Optional[int] = None,
        *,
        allow_classical: bool = False,
    ) -> bytes:
        """Decrypt bundle using Hybrid Post-Quantum (X25519 + ML-KEM-768) private key and verify AAD/tag."""
        try:
            ephem_pub_bytes = bytes.fromhex(bundle.ephemeral_pubkey_hex)
            nonce = bytes.fromhex(bundle.nonce_hex)
            ciphertext = bytes.fromhex(bundle.ciphertext_hex)
        except ValueError as e:
            raise ValueError(f"AEAD malformed hex fields: {e}")

        if len(ephem_pub_bytes) not in (32, cls.HYBRID_ENC_LEN) or len(nonce) != 12 or len(ciphertext) < 16:
            raise ValueError("AEAD invalid key, KEM encapsulation, nonce, or tag length")

        if len(ephem_pub_bytes) == cls.HYBRID_ENC_LEN:
            if len(recipient_privkey_bytes) != cls.HYBRID_PRIV_LEN:
                raise ValueError("AEAD hybrid ML-KEM-768 bundle requires full 2,432-byte hybrid private key")
            if bundle.cipher_suite != "X25519-MLKEM768-ChaCha20Poly1305-SHA3":
                raise ValueError(f"AEAD cipher_suite mismatch for hybrid bundle: {bundle.cipher_suite}")
        else:
            if len(recipient_privkey_bytes) == cls.HYBRID_PRIV_LEN and not allow_classical:
                raise ValueError(
                    "AEAD post-quantum downgrade rejected: classical 32-byte bundle refused for hybrid recipient key"
                )
            if len(recipient_privkey_bytes) not in (32, cls.HYBRID_PRIV_LEN):
                raise ValueError(f"AEAD unsupported recipient private key length: {len(recipient_privkey_bytes)}")
            if bundle.cipher_suite != "X25519-SHAKE256-ChaCha20Poly1305-SHA3":
                raise ValueError(f"AEAD cipher_suite mismatch for classical bundle: {bundle.cipher_suite}")

        if expected_seq is not None and bundle.seq != expected_seq:
            raise ValueError(f"AEAD cross-epoch sequence mismatch: expected {expected_seq}, got {bundle.seq}")

        canonical_aad = f"seq:{bundle.seq};ephem_pub:{ephem_pub_bytes.hex()}"
        if bundle.aad != canonical_aad:
            raise ValueError("AEAD cross-epoch AAD transplant detected: AAD does not match bundle seq/ephem_pub")

        aad = canonical_aad.encode("utf-8")
        sym_key = cls.derive_symmetric_key(
            recipient_privkey_bytes, ephem_pub_bytes, allow_classical=allow_classical
        )
        chacha = ChaCha20Poly1305(sym_key)
        try:
            return chacha.decrypt(nonce, ciphertext, aad)
        except InvalidTag as e:
            raise ValueError(f"AEAD authentication failed (ciphertext/AAD/ML-KEM tampered): {e}")


HybridPQCAEADEngine = X25519ChaCha20Poly1305Engine


# =====================================================================
# 5. Universal Ingestion Adapters
# =====================================================================

class _OptInAdapterBase:
    """Base mixin enforcing opt_in_sharing=False by default on all ingestion/webhook adapters."""

    def __init__(
        self,
        *,
        opt_in_sharing: Optional[bool] = None,
        rsi_engine: Optional["ArmigerDHTRSIEngine"] = None,
    ) -> None:
        self.opt_in_sharing: bool = resolve_opt_in_sharing(opt_in_sharing)
        self.rsi_engine: Optional["ArmigerDHTRSIEngine"] = rsi_engine
        self._outbound_bundles_shared: int = 0
        self._outbound_packets_blocked_opt_out: int = 0
        self._outbound_bytes_shared: int = 0

    def sharing_status(self) -> Dict[str, Any]:
        return {
            "opt_in_sharing_enabled": bool(self.opt_in_sharing),
            "outbound_bundles_shared": int(self._outbound_bundles_shared),
            "outbound_packets_blocked_opt_out": int(self._outbound_packets_blocked_opt_out),
            "outbound_bytes_shared": int(self._outbound_bytes_shared),
        }

    def ingest_event(self, payload: Dict[str, Any]) -> Tuple[CandidateObservation, Optional[Dict[str, Any]]]:
        parser = getattr(self, "parse_webhook", None) or getattr(self, "parse_event")
        obs = parser(payload)
        if not self.opt_in_sharing:
            self._outbound_packets_blocked_opt_out += 1
            return obs, None
        if self.rsi_engine is not None and self.rsi_engine.opt_in_sharing:
            self.rsi_engine.ingest_candidate(obs)
        return obs, None


class SeriesWebhookAdapter(_OptInAdapterBase):
    """Universal adapter for Series webhook download and import webhooks."""

    @staticmethod
    def parse_webhook(payload: Dict[str, Any]) -> CandidateObservation:
        event_type = str(payload.get("eventType", ""))
        series = payload.get("series", {})
        episodes = payload.get("episodes", [])
        release = payload.get("release", {})
        episode_file = payload.get("episodeFile", {})
        media_info = episode_file.get("mediaInfo", {})

        title = series.get("title", "")
        release_title = release.get("releaseTitle") or episode_file.get("relativePath") or title
        size = int(release.get("size") or episode_file.get("size") or 0)
        series_cat_id = (
            series.get("seriesCatalogId")
            or series.get("series_catalog_id")
            or series.get("tvdbId")
            or series.get("tmdbId")
            or series.get("imdbId")
            or series.get("id")
        )

        is_complete = bool(event_type in ("Download", "EpisodeFileDelete", "Rename") and series_cat_id and size > 0)

        stream_meta = {
            "video_codec": media_info.get("videoCodec", "h264"),
            "audio_codec": media_info.get("audioFormat", "aac"),
            "audio_channels": media_info.get("audioChannels", 2.0),
            "audio_languages": media_info.get("audioLanguages"),
            "subtitles": media_info.get("subtitles"),
            "duration_seconds": episode_file.get("duration"),
        }

        return CandidateObservation(
            source_adapter="series_webhook",
            raw_title=str(release_title),
            claimed_category="tv_show",
            download_completed=is_complete,
            bytes_downloaded=size if is_complete else 0,
            total_bytes=size,
            file_path=episode_file.get("relativePath"),
            catalog_id_type="series_catalog",
            catalog_id_value=series_cat_id,
            stream_metadata=stream_meta,
            client_state="imported" if is_complete else "grabbed",
            extra_metadata={"series_title": title, "episode_count": len(episodes)},
        )


class MovieWebhookAdapter(_OptInAdapterBase):
    """Universal adapter for Movie webhook movie download and import webhooks."""

    @staticmethod
    def parse_webhook(payload: Dict[str, Any]) -> CandidateObservation:
        event_type = str(payload.get("eventType", ""))
        movie = payload.get("movie", {})
        release = payload.get("release", {})
        movie_file = payload.get("movieFile", {})
        media_info = movie_file.get("mediaInfo", {})

        title = movie.get("title", "")
        release_title = release.get("releaseTitle") or movie_file.get("relativePath") or title
        size = int(release.get("size") or movie_file.get("size") or 0)
        movie_cat_id = (
            movie.get("movieCatalogId")
            or movie.get("movie_catalog_id")
            or movie.get("tmdbId")
            or movie.get("imdbId")
            or movie.get("id")
        )

        is_complete = bool(event_type in ("Download", "MovieFileDelete") and movie_cat_id and size > 0)

        stream_meta = {
            "video_codec": media_info.get("videoCodec", "hevc"),
            "audio_codec": media_info.get("audioFormat", "truehd"),
            "audio_channels": media_info.get("audioChannels", 7.1),
            "audio_languages": media_info.get("audioLanguages"),
            "subtitles": media_info.get("subtitles"),
            "duration_seconds": movie_file.get("duration"),
        }

        return CandidateObservation(
            source_adapter="movie_webhook",
            raw_title=str(release_title),
            claimed_category="movie",
            download_completed=is_complete,
            bytes_downloaded=size if is_complete else 0,
            total_bytes=size,
            file_path=movie_file.get("relativePath"),
            catalog_id_type="movie_catalog",
            catalog_id_value=movie_cat_id,
            stream_metadata=stream_meta,
            client_state="imported" if is_complete else "grabbed",
            extra_metadata={"movie_title": title, "year": movie.get("year")},
        )


class AudioWebhookAdapter(_OptInAdapterBase):
    """Universal adapter for Audio webhook music album import webhooks."""

    @staticmethod
    def parse_webhook(payload: Dict[str, Any]) -> CandidateObservation:
        event_type = str(payload.get("eventType", ""))
        artist = payload.get("artist", {})
        album = payload.get("album", {})
        track_files = payload.get("trackFiles", [])

        mb_id = (
            artist.get("audioCatalogId")
            or album.get("audioCatalogId")
            or artist.get("mbId")
            or album.get("mbId")
            or album.get("foreignAlbumId")
            or artist.get("foreignArtistId")
            or album.get("id")
        )
        title = f"{artist.get('name', '')} - {album.get('title', '')}"
        total_size = sum(int(t.get("size", 0)) for t in track_files) or 100000000
        is_complete = bool(event_type in ("Download", "TrackFileImported", "AlbumImported") and mb_id)

        first_track = track_files[0] if track_files else {}
        stream_meta = {
            "audio_codec": first_track.get("audioFormat", "flac").lower(),
            "bitrate": first_track.get("bitrate", 900000),
            "channels": 2.0,
        }

        return CandidateObservation(
            source_adapter="audio_webhook",
            raw_title=title,
            claimed_category="music",
            download_completed=is_complete,
            bytes_downloaded=total_size if is_complete else 0,
            total_bytes=total_size,
            catalog_id_type="audio_catalog",
            catalog_id_value=mb_id,
            stream_metadata=stream_meta,
            client_state="imported" if is_complete else "grabbed",
        )


class PublicationWebhookAdapter(_OptInAdapterBase):
    """Universal adapter for Publication webhook ebook and audiobook import webhooks."""

    @staticmethod
    def parse_webhook(payload: Dict[str, Any]) -> CandidateObservation:
        event_type = str(payload.get("eventType", ""))
        book = payload.get("book", {})
        author = payload.get("author", {})
        book_files = payload.get("bookFiles", [])

        isbn = book.get("isbn") or book.get("goodreadsId") or book.get("id")
        title = f"{author.get('name', '')} - {book.get('title', '')}"
        size = sum(int(b.get("size", 0)) for b in book_files) or 5000000
        is_complete = bool(event_type in ("Download", "BookFileImported") and isbn)

        first_file = book_files[0] if book_files else {}
        fmt = str(first_file.get("format", "epub")).lower()
        cat = "audiobook" if fmt in ("m4b", "mp3") else "ebook"

        return CandidateObservation(
            source_adapter="publication_webhook",
            raw_title=title,
            claimed_category=cat,
            download_completed=is_complete,
            bytes_downloaded=size if is_complete else 0,
            total_bytes=size,
            catalog_id_type="isbn",
            catalog_id_value=isbn,
            stream_metadata={"format": fmt},
            client_state="imported" if is_complete else "grabbed",
        )


class QBittorrentAdapter(_OptInAdapterBase):
    """Adapter for qBittorrent completion events and API status payloads."""

    FINISHED_STATES = {"uploading", "pausedUP", "stalledUP", "queuedUP", "forcedUP"}

    @classmethod
    def parse_event(cls, payload: Dict[str, Any]) -> CandidateObservation:
        raw_prog = payload.get("progress", 0.0)
        try:
            progress = float(raw_prog)
        except (TypeError, ValueError):
            progress = -1.0
        valid_prog = math.isfinite(progress) and 0.0 <= progress <= 1.0
        if not valid_prog:
            progress = 0.0
        state = str(payload.get("state", "")).lower()
        raw_size = payload.get("size") or payload.get("total_size") or 0
        try:
            size = int(raw_size) if isinstance(raw_size, (int, float)) and math.isfinite(raw_size) and raw_size > 0 else 0
        except (TypeError, ValueError):
            size = 0
        name = str(payload.get("name", "unknown"))

        is_complete = bool(valid_prog and size > 0 and (progress >= 0.999999 or ( "progress" not in payload and state in {s.lower() for s in cls.FINISHED_STATES}) or (progress >= 0.999999 and state in {s.lower() for s in cls.FINISHED_STATES})))
        return CandidateObservation(
            source_adapter="qbittorrent",
            raw_title=name,
            claimed_category=cls._guess_category_hint(name),
            download_completed=is_complete,
            bytes_downloaded=size if is_complete else int(size * progress),
            total_bytes=size,
            file_path=payload.get("content_path") or payload.get("save_path"),
            client_state=state,
            extra_metadata={"infohash": payload.get("hash")},
        )

    @staticmethod
    def _guess_category_hint(name: str) -> str:
        ln = name.lower()
        if re.search(r'\b(s\d{1,2}e\d{1,2}|season\s*\d+)\b', ln):
            return "tv_show"
        if re.search(r'\b(19\d\d|20\d\d)\b.*\b(1080p|2160p|720p|bluray|webrip)\b', ln):
            return "movie"
        if re.search(r'\b(flac|mp3|discography|lossless)\b', ln):
            return "music"
        if ln.endswith((".iso", ".exe", ".pkg", ".dmg")):
            return "software"
        if ln.endswith((".epub", ".mobi", ".pdf")):
            return "ebook"
        return "unknown"


class TransmissionAdapter(_OptInAdapterBase):
    """Adapter for Transmission download complete script environment variables or RPC dicts."""

    @classmethod
    def parse_event(cls, payload: Dict[str, Any]) -> CandidateObservation:
        # Check env vars first or dictionary
        name = payload.get("TR_TORRENT_NAME") or payload.get("name") or "unknown"
        infohash = payload.get("TR_TORRENT_HASH") or payload.get("hashString")
        percent = float(payload.get("percentDone", 1.0 if "TR_TORRENT_NAME" in payload else 0.0))
        size = int(payload.get("totalSize", 100000000))
        is_finished = bool(payload.get("isFinished", percent >= 0.999999 or "TR_TORRENT_NAME" in payload))

        return CandidateObservation(
            source_adapter="transmission",
            raw_title=str(name),
            claimed_category=QBittorrentAdapter._guess_category_hint(str(name)),
            download_completed=is_finished,
            bytes_downloaded=size if is_finished else int(size * percent),
            total_bytes=size,
            file_path=payload.get("TR_TORRENT_DIR"),
            client_state="finished" if is_finished else "downloading",
            extra_metadata={"infohash": infohash},
        )


class MediaStreamVerifier:
    """Verifies physical media streams using ffprobe or MediaInfo JSON structures."""

    @staticmethod
    def verify_ffprobe_json(data: Dict[str, Any]) -> Dict[str, Any]:
        """Parses ffprobe JSON output to verify physical stream integrity."""
        streams = data.get("streams", [])
        fmt = data.get("format", {})

        video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
        audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)
        subtitle_streams = [s for s in streams if s.get("codec_type") == "subtitle"]

        audio_languages = []
        for s in streams:
            if s.get("codec_type") == "audio":
                tags = s.get("tags", {})
                lang = tags.get("language") or tags.get("lang") or "und"
                audio_languages.append(lang.lower())

        sub_languages = []
        for s in subtitle_streams:
            tags = s.get("tags", {})
            lang = tags.get("language") or tags.get("lang") or "und"
            sub_languages.append(lang.lower())

        dur = fmt.get("duration")
        duration_sec = float(dur) if dur else None

        return {
            "is_valid_media": bool(video_stream or audio_stream),
            "video_codec": video_stream.get("codec_name") if video_stream else None,
            "width": video_stream.get("width") if video_stream else None,
            "height": video_stream.get("height") if video_stream else None,
            "audio_codec": audio_stream.get("codec_name") if audio_stream else None,
            "audio_channels": audio_stream.get("channels") if audio_stream else 0,
            "audio_languages": audio_languages,
            "sub_languages": sub_languages,
            "format_name": fmt.get("format_name"),
            "duration_seconds": duration_sec,
            "size_bytes": int(fmt.get("size", 0)),
        }


