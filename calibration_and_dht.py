"""Per-Head Temperature / Conformal Calibration & BEP-44 Decentralized DHT Updater Plugin.

Provides:
  1. `ArmigerCalibrator`: Post-hoc temperature scaling ($T_h$), Menon class-prior logit adjustment,
     Mondrian conformal safety gates, and schema-validated data-driven plugin threshold/rule
     overrides (`Gap S3-01`) loaded from `calibration.json`.
  2. `ArmigerDHTUpdater`: Decentralized `BEP-44` + `BEP-46` + `Ed25519` + Height-10 `XMSS` Merkle
     one-time `WOTS+` signature verification (`Gap S3-03`) + `BLAKE2b-128` Quorum Verification Cache
     (`73.7x` speedup) for hot-reloading signed `zero-cue`, `swarm-priors`, and 9-head delta bundles.
"""
from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import math
import pathlib
import re
import time
from typing import Any

try:
    import re._parser as _sre_parse  # type: ignore[attr-defined]
    import re._constants as _sre_constants  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover
    import sre_parse as _sre_parse  # type: ignore[no-redef]
    import sre_constants as _sre_constants  # type: ignore[no-redef]

try:
    from armiger.pq_xmss_merkle import QuorumSignatureCache, XMSSSignature, XMSSStatefulVerifier
except ImportError:
    from pq_xmss_merkle import QuorumSignatureCache, XMSSSignature, XMSSStatefulVerifier  # type: ignore[no-redef]


MAX_CUSTOM_REGEX_PATTERN_LEN = 512
MAX_CUSTOM_REGEX_INPUT_CHARS = 512


def _softmax(logits: list[float], temperature: float = 1.0, bias: list[float] | None = None) -> list[float]:
    t = max(float(temperature), 1e-4)
    adj = [
        (float(x) + (float(bias[i]) if bias and i < len(bias) else 0.0)) / t
        for i, x in enumerate(logits)
    ]
    m = max(adj)
    exps = [math.exp(x - m) for x in adj]
    s = sum(exps) or 1.0
    return [e / s for e in exps]


_ALLOWED_PLUGIN_SCHEMAS: dict[str, dict[str, Any]] = {
    "zero_cue": {
        "thresholds": {
            "confidence_ceiling",
            "zero_cue_confidence_ceiling",
            "anime_threshold",
            "anime_catalog_min_prob",
            "adult_siterip_floor",
            "comic_archive_floor",
            "movie_pack_sum_floor",
            "movie_pack_margin",
            "tv_cue_adult_ceiling",
            "tv_cue_xxx_ceiling",
            "tv_cue_margin",
            "tv_cue_floor",
            "game_archive_adult_ceiling",
            "game_archive_floor",
            "course_book_floor",
        },
        "positive_floats": {
            "category_temperature",
        },
        "float_vectors": {
            "category_prior_boost": 11,
        },
        "token_lists": {
            "extra_tv_tokens",
            "extra_movie_tokens",
            "extra_course_tokens",
            "extra_game_tokens",
        },
        "regexes": {
            "explicit_tv_regex",
            "explicit_movie_pack_regex",
            "game_archive_regex",
            "adult_siterip_regex",
            "explicit_comic_regex",
            "course_book_regex",
        },
        "aliases": (
            ("confidence_ceiling", "zero_cue_confidence_ceiling"),
            ("anime_threshold", "anime_catalog_min_prob"),
        ),
    },
    "audio_guard": {
        "thresholds": {
            "en_threshold",
            "english_compatible_threshold",
            "anime_threshold",
            "anime_dub_floor_prob",
            "foreign_title_threshold",
        },
        "positive_floats": set(),
        "float_vectors": {},
        "token_lists": {
            "extra_dub_tokens",
            "extra_foreign_audio_tokens",
        },
        "regexes": {
            "dual_or_dub_regex",
            "multi_subs_only_regex",
            "explicit_en_regex",
            "explicit_non_en_regex",
        },
        "aliases": (
            ("en_threshold", "english_compatible_threshold"),
        ),
    },
    "span_bio": {
        "thresholds": {
            "boundary_confidence_floor",
        },
        "positive_floats": set(),
        "float_vectors": {},
        "token_lists": set(),
        "regexes": set(),
        "aliases": (),
    },
}

_NESTED_QUANTIFIER_RE = re.compile(r"\([^()]*[+*][^()]*\)\s*[+*]")
_ASCII_UNIVERSE = frozenset(range(128))
_EXTENDED_WORD_CHARS = frozenset(
    [ord(c) for c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"]
    + [c for c in range(128, 512) if chr(c).isalnum()]
)
_EXTENDED_DIGIT_CHARS = frozenset(c for c in range(512) if chr(c).isdigit())
_EXTENDED_SPACE_CHARS = frozenset(c for c in range(512) if chr(c).isspace())
_UNIVERSAL_SENTINEL = -1
_UNICODE_WORD_SENTINEL = -2
_UNICODE_DIGIT_SENTINEL = -3
_UNICODE_SPACE_SENTINEL = -4
_UNICODE_NONWORD_SENTINEL = -5
_UNICODE_NONDIGIT_SENTINEL = -6
_UNICODE_NONSPACE_SENTINEL = -7
_ALL_HIGH_UNICODE_SENTINELS = frozenset({
    _UNICODE_WORD_SENTINEL,
    _UNICODE_DIGIT_SENTINEL,
    _UNICODE_SPACE_SENTINEL,
    _UNICODE_NONWORD_SENTINEL,
    _UNICODE_NONDIGIT_SENTINEL,
    _UNICODE_NONSPACE_SENTINEL,
})


def _high_unicode_sentinels_for_codepoint(cp: int) -> set[int]:
    """Return conservative Unicode category sentinels for codepoints >= 512."""
    if cp < 512:
        return set()
    ch = chr(cp)
    tags: set[int] = set()
    if ch.isalnum() or ch == "_":
        tags.add(_UNICODE_WORD_SENTINEL)
    else:
        tags.add(_UNICODE_NONWORD_SENTINEL)
    if ch.isdigit():
        tags.add(_UNICODE_DIGIT_SENTINEL)
    else:
        tags.add(_UNICODE_NONDIGIT_SENTINEL)
    if ch.isspace():
        tags.add(_UNICODE_SPACE_SENTINEL)
    else:
        tags.add(_UNICODE_NONSPACE_SENTINEL)
    return tags


def _expand_codepoint(c: int, ignore_case: bool) -> set[int]:
    """Expand a Unicode codepoint `c` into its exact or case-folded codepoint set plus high-Unicode category sentinels."""
    if c < 0:
        return set()
    raw_cps = {c}
    if ignore_case:
        ch = chr(c)
        for variant in (ch.lower(), ch.upper(), ch.casefold()):
            for vch in variant:
                raw_cps.add(ord(vch))
    out = set(raw_cps)
    for cp in raw_cps:
        out.update(_high_unicode_sentinels_for_codepoint(cp))
    return out


def _category_charset(sub_av: Any) -> frozenset[int]:
    """Return conservative codepoint + high-Unicode sentinel set for a regex category (`\\d`, `\\w`, `\\s`, `\\D`, `\\W`, `\\S`)."""
    if sub_av == _sre_constants.CATEGORY_DIGIT:
        return frozenset(_EXTENDED_DIGIT_CHARS | {_UNICODE_DIGIT_SENTINEL, _UNICODE_WORD_SENTINEL, _UNICODE_NONSPACE_SENTINEL})
    if sub_av == _sre_constants.CATEGORY_WORD:
        return frozenset(_EXTENDED_WORD_CHARS | {_UNICODE_WORD_SENTINEL, _UNICODE_DIGIT_SENTINEL, _UNICODE_NONDIGIT_SENTINEL, _UNICODE_NONSPACE_SENTINEL})
    if sub_av == _sre_constants.CATEGORY_SPACE:
        return frozenset(_EXTENDED_SPACE_CHARS | {_UNICODE_SPACE_SENTINEL, _UNICODE_NONWORD_SENTINEL, _UNICODE_NONDIGIT_SENTINEL})
    if sub_av == getattr(_sre_constants, "CATEGORY_NOT_DIGIT", None):
        return frozenset((set(range(512)) - _EXTENDED_DIGIT_CHARS) | (_ALL_HIGH_UNICODE_SENTINELS - {_UNICODE_DIGIT_SENTINEL}) | {_UNICODE_WORD_SENTINEL, _UNICODE_NONSPACE_SENTINEL})
    if sub_av == getattr(_sre_constants, "CATEGORY_NOT_WORD", None):
        return frozenset((set(range(512)) - _EXTENDED_WORD_CHARS) | (_ALL_HIGH_UNICODE_SENTINELS - {_UNICODE_WORD_SENTINEL, _UNICODE_DIGIT_SENTINEL}))
    if sub_av == getattr(_sre_constants, "CATEGORY_NOT_SPACE", None):
        return frozenset((set(range(512)) - _EXTENDED_SPACE_CHARS) | (_ALL_HIGH_UNICODE_SENTINELS - {_UNICODE_SPACE_SENTINEL}))
    return frozenset(_ASCII_UNIVERSE | set(range(128, 512)) | _ALL_HIGH_UNICODE_SENTINELS | {_UNIVERSAL_SENTINEL})


def _charsets_overlap(a: set[int] | frozenset[int] | None, b: set[int] | frozenset[int] | None) -> bool:
    """Return True if two codepoint sets (supporting `_UNIVERSAL_SENTINEL` and high-Unicode category sentinels) share at least one character."""
    if not a or not b:
        return False
    if _UNIVERSAL_SENTINEL in a or _UNIVERSAL_SENTINEL in b:
        return True
    return bool(a & b)


def _atom_charset(op: Any, av: Any, flags: int = 0) -> frozenset[int] | None:
    """Return the set of Unicode codepoints matched by a single regex AST atom,
    or `None` if the atom is non-consuming (anchor/boundary) or compound.
    """
    ignore_case = bool(flags & re.IGNORECASE)
    if op == _sre_constants.LITERAL:
        return frozenset(_expand_codepoint(int(av), ignore_case))
    if op == _sre_constants.NOT_LITERAL:
        excluded = {cp for cp in _expand_codepoint(int(av), ignore_case) if cp >= 0}
        return frozenset((set(range(512)) - excluded) | _ALL_HIGH_UNICODE_SENTINELS | {_UNIVERSAL_SENTINEL})
    if op == _sre_constants.ANY:
        return frozenset(set(range(512)) | _ALL_HIGH_UNICODE_SENTINELS | {_UNIVERSAL_SENTINEL})
    if op == _sre_constants.IN:
        chars: set[int] = set()
        negate = False
        for sub_op, sub_av in av:
            if sub_op == _sre_constants.NEGATE:
                negate = True
            elif sub_op == _sre_constants.LITERAL:
                chars.update(_expand_codepoint(int(sub_av), ignore_case))
            elif sub_op == _sre_constants.RANGE:
                lo, hi = max(0, int(sub_av[0])), int(sub_av[1])
                if hi - lo <= 512:
                    for c in range(lo, hi + 1):
                        chars.update(_expand_codepoint(c, ignore_case))
                else:
                    chars.update(set(range(512)) | _ALL_HIGH_UNICODE_SENTINELS | {_UNIVERSAL_SENTINEL})
            elif sub_op == _sre_constants.CATEGORY:
                chars.update(_category_charset(sub_av))
        if negate:
            explicit_cps = {cp for cp in chars if cp >= 0}
            return frozenset((set(range(512)) - explicit_cps) | _ALL_HIGH_UNICODE_SENTINELS)
        return frozenset(chars)
    return None


def _in_has_overlapping_items(av: Any, flags: int) -> bool:
    """Return True if an `IN` node (including single-char `|` branches merged by `sre_parse`)
    contains overlapping character items (e.g. `([a-z]|a)+`, `(\\w|a)+`, or `([aé]|[bé])+`).
    """
    ignore_case = bool(flags & re.IGNORECASE)
    seen: set[int] = set()
    for sub_op, sub_av in av:
        if sub_op == _sre_constants.NEGATE:
            continue
        item_chars: set[int] = set()
        if sub_op == _sre_constants.LITERAL:
            item_chars.update(_expand_codepoint(int(sub_av), ignore_case))
        elif sub_op == _sre_constants.RANGE:
            lo, hi = max(0, int(sub_av[0])), int(sub_av[1])
            if hi - lo <= 512:
                for c in range(lo, hi + 1):
                    item_chars.update(_expand_codepoint(c, ignore_case))
            else:
                item_chars.update(set(range(512)) | _ALL_HIGH_UNICODE_SENTINELS | {_UNIVERSAL_SENTINEL})
        elif sub_op == _sre_constants.CATEGORY:
            item_chars.update(_category_charset(sub_av))
        if _charsets_overlap(seen, item_chars):
            return True
        seen.update(item_chars)
    return False


def _branch_arm_signature(nodes: Any, flags: int) -> tuple[frozenset[int], str, bool]:
    """Return `(first_charset, literal_prefix_lower, has_non_literal)` for an alternation branch arm."""
    first_chars: frozenset[int] | None = None
    lit_prefix: list[str] = []
    has_non_literal = False
    atomic_group_op = getattr(_sre_constants, "ATOMIC_GROUP", None)

    for op, av in nodes:
        if op == _sre_constants.AT:
            continue
        if op == _sre_constants.SUBPATTERN:
            sub_nodes = av[3] if len(av) >= 4 else av[-1]
            sub_first, sub_pref, sub_nonlit = _branch_arm_signature(sub_nodes, flags)
            if first_chars is None:
                first_chars = sub_first
            if not has_non_literal:
                lit_prefix.append(sub_pref)
            has_non_literal = has_non_literal or sub_nonlit
            continue
        if atomic_group_op is not None and op == atomic_group_op:
            sub_first, sub_pref, sub_nonlit = _branch_arm_signature(av, flags)
            if first_chars is None:
                first_chars = sub_first
            if not has_non_literal:
                lit_prefix.append(sub_pref)
            has_non_literal = has_non_literal or sub_nonlit
            continue
        if op == _sre_constants.LITERAL:
            cset = _atom_charset(op, av, flags) or _ASCII_UNIVERSE
            if first_chars is None:
                first_chars = cset
            if not has_non_literal:
                lit_prefix.append(chr(int(av)).lower())
        else:
            cset = _atom_charset(op, av, flags) or _ASCII_UNIVERSE
            if first_chars is None:
                first_chars = cset
            has_non_literal = True

    if first_chars is None:
        return frozenset(), "", True
    return first_chars, "".join(lit_prefix), has_non_literal


def _check_regex_safety(pattern: str, field_name: str = "regex") -> re.Pattern[str]:
    """Validate that `pattern` is bounded, syntactically valid, and free of catastrophic-backtracking
    constructs (nested quantifiers including inside atomic groups or possessive repeats, overlapping-alternative
    repetition such as `(a|a)+` or `(ab|a)*`, ambiguous adjacent quantifiers, backreferences, lookarounds,
    and any unrecognized AST operation).
    """
    if not isinstance(pattern, str) or not (1 <= len(pattern) <= MAX_CUSTOM_REGEX_PATTERN_LEN):
        raise ValueError(f"Invalid regex string length for {field_name} (max {MAX_CUSTOM_REGEX_PATTERN_LEN} chars)")
    if _NESTED_QUANTIFIER_RE.search(pattern):
        raise ValueError(f"Unsafe nested quantifier in regex {field_name}")

    try:
        parsed = _sre_parse.parse(pattern)
        compiled = re.compile(pattern)
    except re.error as exc:
        raise ValueError(f"Invalid regex syntax in {field_name}: {exc}") from exc

    flags = int(getattr(parsed.state, "flags", 0)) | int(compiled.flags)
    possessive_repeat_op = getattr(_sre_constants, "POSSESSIVE_REPEAT", None)
    atomic_group_op = getattr(_sre_constants, "ATOMIC_GROUP", None)
    repeat_ops = tuple(
        op
        for op in (_sre_constants.MAX_REPEAT, _sre_constants.MIN_REPEAT, possessive_repeat_op)
        if op is not None
    )
    forbidden_ops = {
        getattr(_sre_constants, "GROUPREF", None),
        getattr(_sre_constants, "GROUPREF_EXISTS", None),
        getattr(_sre_constants, "ASSERT", None),
        getattr(_sre_constants, "ASSERT_NOT", None),
    } - {None}
    allowed_ops = (
        {
            _sre_constants.LITERAL,
            _sre_constants.NOT_LITERAL,
            _sre_constants.ANY,
            _sre_constants.IN,
            _sre_constants.BRANCH,
            _sre_constants.SUBPATTERN,
            _sre_constants.AT,
            _sre_constants.CATEGORY,
            *repeat_ops,
        }
        | ({atomic_group_op} if atomic_group_op is not None else set())
    )

    def _walk(subpattern: Any, *, repeat_depth: int = 0) -> tuple[frozenset[int], bool, bool]:
        """Walk AST and return `(consumed_charset, is_nullable, contains_quantifier)`."""
        consumed_union: set[int] = set()
        prev_repeat_charset: frozenset[int] | None = None
        seq_nullable = True
        seq_has_quantifier = False

        for op, av in subpattern:
            if op in forbidden_ops:
                raise ValueError(f"Unsupported backreference or lookaround assertion in regex {field_name}")
            if op not in allowed_ops:
                raise ValueError(f"Unsupported regex AST operation {op!r} in {field_name}")

            if op in repeat_ops:
                min_rep, max_rep, body = int(av[0]), int(av[1]), av[2]
                is_quantified = (max_rep > 1) or (min_rep != max_rep)
                if repeat_depth > 0 and is_quantified:
                    raise ValueError(f"Unsafe nested quantifier in regex {field_name}")
                body_chars, body_nullable, _ = _walk(
                    body,
                    repeat_depth=repeat_depth + (1 if is_quantified else 0),
                )
                if is_quantified and (not body_chars or body_nullable):
                    raise ValueError(f"Unsafe zero-width repetition in regex {field_name}")
                if _charsets_overlap(prev_repeat_charset, body_chars):
                    raise ValueError(f"Unsafe adjacent overlapping quantifiers in regex {field_name}")
                if is_quantified:
                    seq_has_quantifier = True
                    if min_rep == 0 and prev_repeat_charset is not None:
                        prev_repeat_charset = frozenset(prev_repeat_charset | body_chars)
                    else:
                        prev_repeat_charset = body_chars
                if min_rep > 0 and not body_nullable:
                    seq_nullable = False
                consumed_union.update(body_chars)
                continue

            if atomic_group_op is not None and op == atomic_group_op:
                sub_chars, sub_nullable, sub_has_q = _walk(av, repeat_depth=repeat_depth)
                if _charsets_overlap(prev_repeat_charset, sub_chars):
                    raise ValueError(f"Unsafe adjacent overlapping quantifiers in regex {field_name}")
                if sub_has_q:
                    seq_has_quantifier = True
                    prev_repeat_charset = (
                        frozenset(prev_repeat_charset | sub_chars)
                        if (sub_nullable and prev_repeat_charset is not None)
                        else sub_chars
                    )
                elif not sub_nullable:
                    prev_repeat_charset = None
                if not sub_nullable:
                    seq_nullable = False
                consumed_union.update(sub_chars)
                continue

            if op == _sre_constants.SUBPATTERN:
                sub_add_flags = int(av[1]) if len(av) >= 4 else 0
                sub_del_flags = int(av[2]) if len(av) >= 4 else 0
                sub_body = av[3] if len(av) >= 4 else av[-1]
                local_flags = (flags | sub_add_flags) & ~sub_del_flags
                sub_chars, sub_nullable, sub_has_q = _walk_with_flags(
                    sub_body, repeat_depth=repeat_depth, local_flags=local_flags
                )
                if _charsets_overlap(prev_repeat_charset, sub_chars):
                    raise ValueError(f"Unsafe adjacent overlapping quantifiers in regex {field_name}")
                if sub_has_q:
                    seq_has_quantifier = True
                    prev_repeat_charset = (
                        frozenset(prev_repeat_charset | sub_chars)
                        if (sub_nullable and prev_repeat_charset is not None)
                        else sub_chars
                    )
                elif not sub_nullable:
                    prev_repeat_charset = None
                if not sub_nullable:
                    seq_nullable = False
                consumed_union.update(sub_chars)
                continue

            if op == _sre_constants.BRANCH:
                _, branches = av
                arm_sigs: list[tuple[frozenset[int], str, bool]] = []
                branch_nullable = False
                branch_has_q = False
                branch_chars: set[int] = set()
                for arm in branches:
                    if not len(arm):
                        if repeat_depth > 0:
                            raise ValueError(
                                f"Unsafe overlapping-alternative repetition in regex {field_name}"
                            )
                        raise ValueError(f"Empty alternation branch in regex {field_name}")
                    arm_c, arm_null, arm_q = _walk(arm, repeat_depth=repeat_depth)
                    branch_chars.update(arm_c)
                    branch_nullable = branch_nullable or arm_null
                    branch_has_q = branch_has_q or arm_q
                    consumed_union.update(arm_c)
                    arm_sigs.append(_branch_arm_signature(arm, flags))

                if _charsets_overlap(prev_repeat_charset, branch_chars):
                    raise ValueError(f"Unsafe adjacent overlapping quantifiers in regex {field_name}")

                if repeat_depth > 0:
                    # Inside a quantifier `(alt1|alt2)+` or `(alt1|alt2)*`: prohibit any overlapping
                    # starting characters or prefix-overlapping alternatives such as `(a|a)+` or `(ab|a)*`.
                    for i in range(len(arm_sigs)):
                        fc_i, pref_i, nonlit_i = arm_sigs[i]
                        if not fc_i or (nonlit_i and not pref_i):
                            raise ValueError(
                                f"Unsafe overlapping-alternative repetition in regex {field_name}"
                            )
                        for j in range(i + 1, len(arm_sigs)):
                            fc_j, pref_j, _ = arm_sigs[j]
                            if _charsets_overlap(fc_i, fc_j) or (
                                pref_i and pref_j and (pref_i.startswith(pref_j) or pref_j.startswith(pref_i))
                            ):
                                raise ValueError(
                                    f"Unsafe overlapping-alternative repetition in regex {field_name}"
                                )
                if branch_has_q or branch_nullable:
                    seq_has_quantifier = seq_has_quantifier or branch_has_q
                    prev_repeat_charset = (
                        frozenset((prev_repeat_charset or frozenset()) | branch_chars)
                        if branch_nullable
                        else frozenset(branch_chars)
                    )
                else:
                    prev_repeat_charset = None
                if not branch_nullable:
                    seq_nullable = False
                continue

            if op == _sre_constants.IN and repeat_depth > 0 and _in_has_overlapping_items(av, flags):
                raise ValueError(
                    f"Unsafe overlapping-alternative repetition in regex {field_name}"
                )

            cset = _atom_charset(op, av, flags)
            if cset is not None:
                if _charsets_overlap(prev_repeat_charset, cset):
                    raise ValueError(f"Unsafe adjacent overlapping quantifiers in regex {field_name}")
                consumed_union.update(cset)
                prev_repeat_charset = None
                seq_nullable = False

        return frozenset(consumed_union), seq_nullable, seq_has_quantifier

    def _walk_with_flags(
        subpattern: Any, *, repeat_depth: int, local_flags: int
    ) -> tuple[frozenset[int], bool, bool]:
        nonlocal flags
        old_flags = flags
        flags = local_flags
        try:
            return _walk(subpattern, repeat_depth=repeat_depth)
        finally:
            flags = old_flags

    _walk(parsed, repeat_depth=0)
    return compiled


@functools.lru_cache(maxsize=256)
def compile_safe_custom_regex(pattern: str, field_name: str = "custom_regex") -> re.Pattern[str]:
    """Compile and cache a custom override regex after verifying AST safety."""
    return _check_regex_safety(pattern, field_name=field_name)


def safe_regex_search(
    pattern_or_str: str | re.Pattern[str] | None,
    text: str,
    *,
    field_name: str = "custom_regex",
    max_input_chars: int = MAX_CUSTOM_REGEX_INPUT_CHARS,
) -> bool:
    """Safely evaluate a validated custom regex against a length-bounded slice of `text`."""
    if not pattern_or_str or not text:
        return False
    bounded_text = str(text)[: max(1, int(max_input_chars))]
    if isinstance(pattern_or_str, re.Pattern):
        compiled = compile_safe_custom_regex(pattern_or_str.pattern, field_name=field_name)
    else:
        compiled = compile_safe_custom_regex(str(pattern_or_str), field_name=field_name)
    return bool(compiled.search(bounded_text))


def validate_plugin_overrides_schema(raw_overrides: Any) -> dict[str, dict[str, Any]]:
    """Validate data-driven JSON `plugin_overrides` inside `calibration.json` (`Gap S3-01`).

    Strictly enforces:
      - Only known plugin namespaces (`zero_cue`, `audio_guard`, `span_bio`)
      - Only schema-whitelisted threshold floats in `[0.0, 1.0]`, bounded positive temperatures,
        11-class prior vectors, bounded token lists, and AST-verified ReDoS-safe regular expressions
      - Canonical alias synchronization between schema keys and resolver keys (`confidence_ceiling`
        <-> `zero_cue_confidence_ceiling`, `anime_threshold` <-> `anime_catalog_min_prob`,
        `en_threshold` <-> `english_compatible_threshold`)
    """
    if raw_overrides is None:
        return {}
    if not isinstance(raw_overrides, dict):
        raise ValueError("plugin_overrides must be a JSON object")
    validated: dict[str, dict[str, Any]] = {}
    for plugin_name, block in raw_overrides.items():
        if plugin_name not in _ALLOWED_PLUGIN_SCHEMAS:
            raise ValueError(f"Unsupported plugin override namespace: {plugin_name!r}")
        if not isinstance(block, dict):
            raise ValueError(f"plugin_overrides[{plugin_name!r}] must be a JSON object")
        schema = _ALLOWED_PLUGIN_SCHEMAS[plugin_name]
        allowed_t: set[str] = schema["thresholds"]
        allowed_pos: set[str] = schema["positive_floats"]
        allowed_vecs: dict[str, int] = schema["float_vectors"]
        allowed_tokens: set[str] = schema["token_lists"]
        allowed_r: set[str] = schema["regexes"]
        aliases: tuple[tuple[str, str], ...] = schema["aliases"]

        clean_block: dict[str, Any] = {}
        for k, v in block.items():
            if k in allowed_t:
                if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
                    raise ValueError(f"Invalid numeric threshold for {plugin_name}.{k}: {v!r}")
                fv = float(v)
                if not (0.0 <= fv <= 1.0):
                    raise ValueError(f"Threshold {plugin_name}.{k}={fv} outside [0.0, 1.0]")
                clean_block[k] = fv
            elif k in allowed_pos:
                if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
                    raise ValueError(f"Invalid positive float for {plugin_name}.{k}: {v!r}")
                fv = float(v)
                if not (0.05 <= fv <= 10.0):
                    raise ValueError(f"Temperature {plugin_name}.{k}={fv} outside [0.05, 10.0]")
                clean_block[k] = fv
            elif k in allowed_vecs:
                expected_len = allowed_vecs[k]
                if not isinstance(v, (list, tuple)) or len(v) != expected_len:
                    raise ValueError(f"{plugin_name}.{k} must be a list of {expected_len} floats")
                vec: list[float] = []
                for item in v:
                    if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)):
                        raise ValueError(f"Invalid float element in {plugin_name}.{k}: {item!r}")
                    f_item = float(item)
                    if not (-5.0 <= f_item <= 5.0):
                        raise ValueError(f"Element {f_item} in {plugin_name}.{k} outside [-5.0, 5.0]")
                    vec.append(f_item)
                clean_block[k] = vec
            elif k in allowed_tokens:
                if not isinstance(v, (list, tuple)) or len(v) > 64:
                    raise ValueError(f"{plugin_name}.{k} must be a list of at most 64 token strings")
                tok_list: list[str] = []
                for tok in v:
                    if not isinstance(tok, str) or not (1 <= len(tok.strip()) <= 64):
                        raise ValueError(f"Invalid token string in {plugin_name}.{k}: {tok!r}")
                    tok_list.append(tok.strip())
                clean_block[k] = tok_list
            elif k in allowed_r:
                _check_regex_safety(v, field_name=f"{plugin_name}.{k}")
                clean_block[k] = v
            else:
                raise ValueError(f"Unknown schema key in plugin_overrides[{plugin_name!r}]: {k!r}")

        for key_a, key_b in aliases:
            in_a = key_a in clean_block
            in_b = key_b in clean_block
            if in_a and in_b and abs(float(clean_block[key_a]) - float(clean_block[key_b])) > 1e-9:
                raise ValueError(
                    f"Conflicting alias values in {plugin_name}: {key_a}={clean_block[key_a]} vs {key_b}={clean_block[key_b]}"
                )
            if in_a and not in_b:
                clean_block[key_b] = clean_block[key_a]
            elif in_b and not in_a:
                clean_block[key_a] = clean_block[key_b]

        validated[plugin_name] = clean_block
    return validated


class ArmigerCalibrator:
    """Applies `calibration.json` temperature scaling, Menon prior adjustments, Mondrian gates,
    and schema-validated data-driven plugin overrides (`Gap S3-01`).
    """

    DEFAULT_MONDRIAN = {
        "adult": 0.5451,
        "junk": 0.1010,
        "scope": 0.4770,
        "owner_scope": 0.4770,
    }

    def __init__(
        self,
        calibration_path: str | pathlib.Path | dict[str, Any] | None = None,
        *,
        profile: str | None = None,
    ) -> None:
        if isinstance(calibration_path, dict):
            self.calibration_path = None
            raw_cfg = dict(calibration_path)
        else:
            if calibration_path is None:
                calibration_path = pathlib.Path(__file__).with_name("calibration.json")
            self.calibration_path = pathlib.Path(calibration_path)
            raw_cfg = (
                json.loads(self.calibration_path.read_text(encoding="utf-8"))
                if self.calibration_path.exists()
                else {}
            )
        if not isinstance(raw_cfg, dict):
            raise ValueError("calibration config must be a JSON object")
        self.config: dict[str, Any] = raw_cfg

        eff_profile = profile or raw_cfg.get("active_profile")
        profiles = raw_cfg.get("profiles") if isinstance(raw_cfg.get("profiles"), dict) else {}
        active_block = profiles.get(eff_profile, raw_cfg) if eff_profile and eff_profile in profiles else raw_cfg

        raw_heads = active_block.get("heads")
        if not isinstance(raw_heads, dict):
            raw_heads = raw_cfg.get("temperature_scaling_stop_fit") or {}
        parsed_heads: dict[str, dict[str, Any]] = {}
        for h_name, h_val in raw_heads.items():
            if not isinstance(h_val, dict):
                raise ValueError(f"invalid head calibration block for {h_name}")
            t_raw = h_val.get("temperature", h_val.get("temperature_T", 1.0))
            if isinstance(t_raw, bool) or not isinstance(t_raw, (int, float)) or not math.isfinite(float(t_raw)) or float(t_raw) <= 0.0:
                raise ValueError(f"invalid temperature for head {h_name}")
            t_clamped = min(max(float(t_raw), 0.25), 2.5)
            entry: dict[str, Any] = {"temperature": t_clamped}
            if "logit_bias" in h_val and h_val["logit_bias"] is not None:
                if not isinstance(h_val["logit_bias"], list) or any(
                    isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(float(x))
                    for x in h_val["logit_bias"]
                ):
                    raise ValueError(f"invalid logit_bias for head {h_name}")
                entry["logit_bias"] = [min(max(float(x), -1.5), 1.5) for x in h_val["logit_bias"]]
            parsed_heads[h_name] = entry
        self.heads = parsed_heads

        raw_m = active_block.get("mondrian_thresholds")
        if not isinstance(raw_m, dict):
            m_Legacy = raw_cfg.get("mondrian_conformal_thresholds")
            if isinstance(m_Legacy, dict):
                raw_m = {}
                for k_m, gk in (("adult", "adult"), ("junk", "junk"), ("owner_scope", "owner_scope")):
                    v_m = m_Legacy.get(k_m)
                    if isinstance(v_m, dict):
                        # accept `threshold`, `threshold_G250`, `threshold_r5clean`, ... (trainers suffix the key per run)
                        t_keys = ["threshold"] + sorted(k for k in v_m if str(k).startswith("threshold_"))
                        t_v = next((v_m[k] for k in t_keys if k in v_m and isinstance(v_m[k], (int, float))), self.DEFAULT_MONDRIAN[gk])
                        raw_m[gk] = float(t_v)
                    elif isinstance(v_m, (int, float)):
                        raw_m[gk] = float(v_m)
            else:
                raw_m = {}

        adult_t = float(raw_m.get("adult", self.DEFAULT_MONDRIAN["adult"]))
        junk_t = float(raw_m.get("junk", self.DEFAULT_MONDRIAN["junk"]))
        scope_t = float(raw_m.get("scope", raw_m.get("owner_scope", self.DEFAULT_MONDRIAN["scope"])))
        for tk, tv in (("adult", adult_t), ("junk", junk_t), ("scope", scope_t)):
            if not math.isfinite(tv) or not (0.0 < tv < 1.0):
                raise ValueError(f"invalid mondrian threshold {tk}={tv}")
        self.mondrian = {
            "adult": adult_t,
            "junk": junk_t,
            "scope": scope_t,
            "owner_scope": scope_t,
        }

        # Mondrian per-stratum gate thresholds (written by recalibrate_mondrian_stratum_weighted.py):
        # {"owner_scope": {"stratum_key": "category", "strata": {"movie": 0.21, ...}}}
        self.mondrian_strata: dict[str, dict[str, float]] = {}
        self.mondrian_stratum_key: dict[str, str] = {}
        raw_strat = active_block.get("mondrian_stratified_thresholds")
        if raw_strat is None:
            raw_strat = raw_cfg.get("mondrian_stratified_thresholds")
        if isinstance(raw_strat, dict):
            for gate, block in raw_strat.items():
                if gate not in ("adult", "junk", "owner_scope", "scope") or not isinstance(block, dict):
                    raise ValueError(f"invalid mondrian_stratified_thresholds block for {gate!r}")
                strata = block.get("strata")
                if not isinstance(strata, dict):
                    raise ValueError(f"mondrian_stratified_thresholds[{gate!r}].strata must be an object")
                clean: dict[str, float] = {}
                for k, v in strata.items():
                    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)) or not (0.0 < float(v) < 1.0):
                        raise ValueError(f"invalid stratified mondrian threshold {gate}.{k}={v!r}")
                    clean[str(k)] = float(v)
                g_key = "owner_scope" if gate == "scope" else gate
                self.mondrian_strata[g_key] = clean
                self.mondrian_stratum_key[g_key] = str(block.get("stratum_key", "category"))

        raw_overrides = (
            active_block.get("plugin_overrides")
            if "plugin_overrides" in active_block
            else raw_cfg.get("plugin_overrides")
        )
        self.plugin_overrides: dict[str, dict[str, Any]] = validate_plugin_overrides_schema(raw_overrides)

    def get_plugin_rules(self, plugin_name: str) -> dict[str, Any]:
        """Return validated data-driven threshold/regex overrides for `plugin_name`."""
        return dict(self.plugin_overrides.get(plugin_name, {}))

    def thresholds(self) -> dict[str, float]:
        return {
            "adult": float(self.mondrian["adult"]),
            "junk": float(self.mondrian["junk"]),
            "scope": float(self.mondrian["scope"]),
        }

    def calibrate_head(self, head_name: str, raw_logits: list[float]) -> list[float]:
        if not raw_logits or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(float(x)) for x in raw_logits):
            n = max(len(raw_logits), 2)
            return [0.0] * (n - 1) + [1.0]
        hcfg = self.heads.get(head_name, {})
        temp = float(hcfg.get("temperature", 1.0))
        bias = hcfg.get("logit_bias")
        return _softmax(raw_logits, temperature=temp, bias=bias)

    def calibrate_probs(self, probs: dict[str, list[float]]) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {}
        for head_name, vec in probs.items():
            hcfg = self.heads.get(head_name, {})
            temp = float(hcfg.get("temperature", 1.0))
            bias = hcfg.get("logit_bias")
            if abs(temp - 1.0) <= 1e-6 and not bias:
                out[head_name] = [float(v) for v in vec]
            else:
                pseudo_logits = [math.log(max(float(v), 1e-12)) for v in vec]
                scaled = _softmax(pseudo_logits, temperature=temp, bias=bias)
                rounded = [round(float(v), 6) for v in scaled]
                if rounded:
                    diff = round(1.0 - sum(rounded), 6)
                    best_i = max(range(len(rounded)), key=lambda i: rounded[i])
                    rounded[best_i] = round(max(0.0, min(1.0, rounded[best_i] + diff)), 6)
                out[head_name] = rounded
        return out

    def gate_threshold(self, gate: str, stratum: str | None = None) -> float:
        """Threshold for `gate` (`adult` / `junk` / `owner_scope`): the Mondrian per-stratum value when
        `stratum` (e.g. the predicted category) is known and fitted, else the global value.
        """
        g_key = "owner_scope" if gate == "scope" else gate
        if stratum is not None:
            per = self.mondrian_strata.get(g_key, {})
            if str(stratum) in per:
                return float(per[str(stratum)])
        return float(self.mondrian[g_key])

    def evaluate_safety_gates(
        self,
        p_adult: float,
        p_junk: float,
        p_excluded_scope: float,
        *,
        stratum: str | None = None,
    ) -> dict[str, bool]:
        def _flag(val: float, thresh: float) -> bool:
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val) or val < 0.0 or val > 1.0:
                return True
            return bool(val >= thresh)

        return {
            "adult_flagged": _flag(p_adult, self.gate_threshold("adult", stratum)),
            "junk_flagged": _flag(p_junk, self.gate_threshold("junk", stratum)),
            "scope_excluded": _flag(p_excluded_scope, self.gate_threshold("owner_scope", stratum)),
        }


@dataclasses.dataclass(frozen=True)
class SignedDHTManifest:
    """BEP-44 / BEP-46 Hybrid Post-Quantum (Ed25519 + WOTS+-SHAKE256 or Height-10 XMSS Merkle) signed update manifest."""

    seq: int
    version: str
    infohash: str
    bundle_sha256: str
    quorum_attestations: int
    timestamp: int
    ed25519_signature_hex: str
    pq_pubkey_hex: str = ""
    pq_signature_hex: str = ""
    xmss_signature_hex: str | None = None

    def canonical_payload(self) -> bytes:
        body: dict[str, Any] = {
            "bundle_sha256": self.bundle_sha256.lower(),
            "infohash": self.infohash.lower(),
            "quorum_attestations": int(self.quorum_attestations),
            "seq": int(self.seq),
            "timestamp": int(self.timestamp),
            "version": self.version,
        }
        if self.pq_pubkey_hex:
            body["pq_pubkey_hex"] = self.pq_pubkey_hex.lower()
        return json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")


class ArmigerDHTUpdater:
    """Post-Quantum zero-trust verification (WOTS+-SHAKE256 + Height-10 XMSS Merkle + BLAKE2b-128 cache)
    and canary rollback guard for decentralized BEP-44 DHT updates.
    """

    def __init__(
        self,
        root_pubkey_hex: str,
        current_seq: int = 0,
        min_quorum: int = 5,
        mode: str = "opt_in_verify",
        root_pq_pubkey_hex: str = "",
        xmss_merkle_root_hex: str | None = None,
        xmss_height: int = 10,
        signature_cache: QuorumSignatureCache | None = None,
        require_pq: bool = True,
    ) -> None:
        if mode not in ("off", "opt_in_verify", "federated_quorum"):
            raise ValueError(f"Unsupported DHT update mode: {mode}")
        if not isinstance(require_pq, bool):
            raise TypeError(f"require_pq must be bool, got {type(require_pq).__name__}")
        self.require_pq = require_pq
        self._min_quorum_is_bool = isinstance(min_quorum, bool)
        self._seq_is_bool = isinstance(current_seq, bool)
        self.root_pubkey_hex = root_pubkey_hex.lower()
        self.root_pq_pubkey_hex = root_pq_pubkey_hex.lower()
        self.current_seq = int(current_seq)
        self.min_quorum = int(min_quorum)
        self.mode = mode
        self.xmss_merkle_root_hex = xmss_merkle_root_hex.lower() if xmss_merkle_root_hex else None
        self._xmss_verifier = (
            XMSSStatefulVerifier(
                self.xmss_merkle_root_hex,
                current_seq=self.current_seq,
                expected_height=xmss_height,
            )
            if self.xmss_merkle_root_hex is not None
            else None
        )
        self.signature_cache = signature_cache if signature_cache is not None else QuorumSignatureCache()

    @staticmethod
    @functools.lru_cache(maxsize=512)
    def _verify_ed25519_cached(pub_bytes: bytes, payload: bytes, sig_bytes: bytes) -> bool:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        try:
            Ed25519PublicKey.from_public_bytes(pub_bytes).verify(sig_bytes, payload)
            return True
        except (InvalidSignature, ValueError):
            return False

    @staticmethod
    @functools.lru_cache(maxsize=512)
    def _verify_wots_cached(pq_pub_bytes: bytes, payload: bytes, pq_sig_bytes: bytes) -> bool:
        try:
            from armiger.csam_guard import WOTSPlusSHAKE256
        except ImportError:
            from csam_guard import WOTSPlusSHAKE256
        return bool(WOTSPlusSHAKE256.verify(pq_pub_bytes, payload, pq_sig_bytes))

    def verify_bundle(
        self,
        manifest: SignedDHTManifest,
        bundle_bytes: bytes,
        signature_verifier: Any | None = None,
        *,
        commit_seq: bool | None = None,
    ) -> tuple[bool, str]:
        """Verify monotonic sequence, quorum threshold, SHA-256 digest, Ed25519 signature, and
        Post-Quantum WOTS+-SHAKE256 or Height-10 XMSS Merkle one-time signature.
        """
        if self.mode == "off":
            return False, "dht_updates_disabled"
        if signature_verifier is not None:
            return False, "custom_signature_verifier_disallowed"
        if self._min_quorum_is_bool or self._seq_is_bool or isinstance(manifest.quorum_attestations, bool):
            return False, "invalid_quorum_bool_type"
        if isinstance(manifest.seq, bool) or not isinstance(manifest.seq, int) or manifest.seq <= 0:
            return False, "invalid_manifest_seq_type"
        if isinstance(manifest.timestamp, bool) or not isinstance(manifest.timestamp, int) or manifest.timestamp <= 0:
            return False, "invalid_manifest_timestamp_type"
        if manifest.seq <= self.current_seq:
            return False, f"stale_or_replayed_seq:{manifest.seq}<={self.current_seq}"
        if manifest.seq > (2**63 - 1) or manifest.seq > self.current_seq + 1024:
            return False, f"seq_jump_exceeds_max_window:{manifest.seq}"
        if self._xmss_verifier is None:
            now_ts = int(time.time())
            if (manifest.timestamp - now_ts) > 300 or (now_ts - manifest.timestamp) > 14 * 86400:
                return False, f"manifest_timestamp_out_of_bounds:{manifest.timestamp}"
        if self.min_quorum <= 0 or manifest.quorum_attestations < self.min_quorum:
            return False, f"insufficient_quorum:{manifest.quorum_attestations}<{self.min_quorum}"
        if (
            manifest.bundle_sha256 != manifest.bundle_sha256.lower()
            or manifest.infohash != manifest.infohash.lower()
            or manifest.ed25519_signature_hex != manifest.ed25519_signature_hex.lower()
            or manifest.pq_pubkey_hex != manifest.pq_pubkey_hex.lower()
            or manifest.pq_signature_hex != manifest.pq_signature_hex.lower()
        ):
            return False, "non_canonical_uppercase_hex_in_manifest"

        actual_sha = hashlib.sha256(bundle_bytes).hexdigest()
        if actual_sha != manifest.bundle_sha256.lower():
            return False, "bundle_sha256_mismatch"

        payload = manifest.canonical_payload()
        try:
            pub_bytes = bytes.fromhex(self.root_pubkey_hex)
            sig_bytes = bytes.fromhex(manifest.ed25519_signature_hex)
        except ValueError:
            return False, "ed25519_signature_invalid_hex"

        if not self.signature_cache.verify_with_cache(
            scheme="ed25519",
            pubkey_or_root=pub_bytes,
            seq=manifest.seq,
            payload=payload,
            signature=sig_bytes,
            verifier_fn=self._verify_ed25519_cached,
        ):
            return False, "ed25519_signature_invalid"

        if self._xmss_verifier is not None:
            if not manifest.xmss_signature_hex or not isinstance(manifest.xmss_signature_hex, str):
                return False, "xmss_signature_missing"
            try:
                xmss_sig_bytes = bytes.fromhex(manifest.xmss_signature_hex)
            except ValueError:
                return False, "xmss_signature_invalid_hex"
            try:
                xmss_sig_obj = XMSSSignature.from_bytes(xmss_sig_bytes)
                if int(xmss_sig_obj.seq) != int(manifest.seq):
                    return False, f"xmss_seq_mismatch:{xmss_sig_obj.seq}!={manifest.seq}"
                xmss_ok, xmss_reason = self._xmss_verifier.verify_and_advance(
                    message=payload,
                    xmss_sig=xmss_sig_obj,
                )
            except (ValueError, TypeError, KeyError) as exc:
                return False, f"xmss_signature_parse_error:{exc}"
            if not xmss_ok:
                return False, xmss_reason
            if commit_seq is None or commit_seq:
                self.current_seq = int(manifest.seq)
            return True, "verified"

        if not self.root_pq_pubkey_hex:
            if manifest.pq_pubkey_hex or manifest.pq_signature_hex:
                return False, "untrusted_pq_pubkey_when_root_pq_unset"
            return False, "pq_signature_and_pinned_root_required"

        if manifest.pq_pubkey_hex and manifest.pq_pubkey_hex.lower() != self.root_pq_pubkey_hex:
            return False, "pq_pubkey_mismatch_against_pinned_root"
        if not manifest.pq_signature_hex:
            return False, "pq_wots_signature_invalid"
        try:
            pq_ok = self._verify_wots_cached(
                bytes.fromhex(self.root_pq_pubkey_hex),
                payload,
                bytes.fromhex(manifest.pq_signature_hex),
            )
        except ValueError:
            pq_ok = False
        if not pq_ok:
            return False, "pq_wots_signature_invalid"

        if bool(commit_seq):
            self.commit_bundle(manifest)
        return True, "verified"

    def commit_bundle(self, manifest: SignedDHTManifest) -> int:
        """Advance `self.current_seq` after a verified bundle has been staged and accepted."""
        if isinstance(manifest.seq, bool) or not isinstance(manifest.seq, int) or manifest.seq <= self.current_seq:
            raise ValueError(f"invalid_or_non_monotonic_commit_seq:{manifest.seq}<={self.current_seq}")
        self.current_seq = int(manifest.seq)
        return self.current_seq

    def verify_and_apply_bundle(
        self,
        manifest: SignedDHTManifest,
        bundle_bytes: bytes,
        apply_fn: Any,
        canary_fn: Any | None = None,
        rollback_fn: Any | None = None,
    ) -> tuple[bool, str]:
        """Verify bundle, run `apply_fn` and optional `canary_fn`, and commit `current_seq` only on success."""
        ok, reason = self.verify_bundle(manifest, bundle_bytes, commit_seq=False)
        if not ok:
            return False, reason
        try:
            apply_res = apply_fn(bundle_bytes)
            if apply_res is False:
                if rollback_fn is not None:
                    rollback_fn()
                return False, "bundle_application_rejected"
            if canary_fn is not None:
                canary_ok = canary_fn()
                if not canary_ok:
                    if rollback_fn is not None:
                        rollback_fn()
                    return False, "canary_regression_rejected"
        except Exception as exc:
            if rollback_fn is not None:
                rollback_fn()
            return False, f"bundle_application_failed:{exc}"
        self.commit_bundle(manifest)
        return True, "verified_applied_and_committed"


ArmigerDHTRSIEngine = ArmigerDHTUpdater
