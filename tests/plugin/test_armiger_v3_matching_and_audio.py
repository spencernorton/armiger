"""Unit tests for Armiger v3.0 Matching & Audio Language Identification upgrades.

All test cases use strictly synthetic release names and public-domain fixtures.
"""
from __future__ import annotations

import pathlib
import sys

import unittest

from armiger.audio_language_guard import (
    apply_audio_language_guard,
    apply_audio_language_guard_v3,
    apply_title_language_guard,
    classify_audio_and_anime_dub,
    classify_audio_and_anime_dub_v3,
    detect_localized_foreign_title_stem,
    evaluate_owner_policy_rule3,
    is_localized_foreign_alias,
    split_and_score_title_slug,
)
from armiger.byte_span_extractor import (
    BIO_B_EP,
    BIO_B_TITLE,
    BIO_B_YEAR,
    BIO_I_EP,
    BIO_I_TITLE,
    BIO_I_YEAR,
    BIO_O,
    count_illegal_bio_transitions,
    decode_bio_logits_viterbi,
    decode_bio_tags_raw_neural,
    decode_bio_tags_to_spans,
    extract_byte_spans_v3,
    is_illegal_bio_transition,
    is_strict_clean_title,
    key_norm,
    snap_bio_span_boundaries,
    title_token_f1,
)
from armiger.zero_cue_plugin import (
    IDX_AUDIOBOOK,
    IDX_COMIC,
    IDX_COURSE,
    IDX_GAME,
    IDX_MOVIE,
    IDX_TV,
    hash_catalog_key,
    resolve_category_v3,
)


class ArmigerV3MatchingAndAudioTests(unittest.TestCase):
    def test_c2_strict_clean_title_rejects_unstripped_release_and_tech_tokens(self) -> None:
        raw = "Synthetic.Horizon.2024.1080p.WEB-DL.DDP5.1.H.264-SYNTHGRP.mkv"
        self.assertFalse(is_strict_clean_title(raw, raw))
        self.assertFalse(is_strict_clean_title("synthetic horizon 2024 1080p web dl", raw))
        self.assertFalse(is_strict_clean_title("synthetic horizon x264 mkv", raw))
        self.assertTrue(is_strict_clean_title("synthetic horizon", raw))
        self.assertEqual(title_token_f1("Synthetic Horizon", "synthetic horizon"), 1.0)

    def test_c3_raw_neural_bio_decoder_has_zero_heuristic_fallback(self) -> None:
        raw = "Synthetic.Starlight.Odyssey.2024.1080p.BluRay.x265-SYNTH.mkv"
        b = raw.encode("utf-8")
        tags = [BIO_O] * len(b)
        tags[0] = BIO_B_TITLE
        for i in range(1, 27):
            tags[i] = BIO_I_TITLE
        tags[28] = BIO_B_YEAR
        for i in range(29, 32):
            tags[i] = BIO_I_YEAR

        res_neural = decode_bio_tags_raw_neural(raw, tags)
        self.assertEqual(key_norm(res_neural["title"]), "synthetic starlight odyssey")
        self.assertEqual(res_neural["year"], 2024)
        self.assertTrue(res_neural["is_anchored"])
        self.assertEqual(res_neural["extractor_source"], "raw_neural_span_bio")

        all_o = [BIO_O] * len(b)
        res_empty = decode_bio_tags_raw_neural(raw, all_o)
        self.assertEqual(res_empty["title"], "")
        self.assertIsNone(res_empty["year"])
        self.assertFalse(res_empty["is_anchored"])

        res_hybrid = decode_bio_tags_to_spans(raw, all_o)
        self.assertEqual(key_norm(res_hybrid["title"]), "synthetic starlight odyssey")
        self.assertEqual(res_hybrid["year"], 2024)
        self.assertEqual(res_hybrid["extractor_source"], "structural_byte_span")

        # Over-extended neural span including '1080p' is NOT masked by heuristic filtering in raw decoder
        bad_tags = [BIO_O] * len(b)
        bad_tags[0] = BIO_B_TITLE
        for i in range(1, 38):
            bad_tags[i] = BIO_I_TITLE
        res_over = decode_bio_tags_raw_neural(raw, bad_tags)
        self.assertIn("1080p", key_norm(res_over["title"]))
        self.assertFalse(res_over["is_anchored"])
        res_over_hybrid = decode_bio_tags_to_spans(raw, bad_tags)
        self.assertEqual(res_over_hybrid["extractor_source"], "structural_byte_span")
        self.assertEqual(key_norm(res_over_hybrid["title"]), "synthetic starlight odyssey")

    def test_extract_byte_spans_v3_handles_cjk_fansub_and_numeric_titles(self) -> None:
        cjk_raw = "[SynthSub] 幻影星辰记 / Phantom Starlight Chronicle - 08 [1080p][HEVC-10bit].mkv"
        sp = extract_byte_spans_v3(cjk_raw)
        self.assertIsNone(sp["year"])
        self.assertIn("phantom starlight chronicle", key_norm(sp["title"]))

        cjk_pure = "幻影星辰记.2024.1080p.WEB-DL.mkv"
        sp_cjk_pure = extract_byte_spans_v3(cjk_pure)
        self.assertEqual(sp_cjk_pure["title"], "幻影星辰记")
        self.assertEqual(sp_cjk_pure["year"], 2024)
        self.assertTrue(sp_cjk_pure["is_anchored"])

        num_raw = "2049.2023.2160p.UHD.BluRay.REMUX.HEVC.TrueHD.7.1.Atmos-SYNTH"
        sp_num = extract_byte_spans_v3(num_raw)
        self.assertEqual(key_norm(sp_num["title"]), "2049")
        self.assertEqual(sp_num["year"], 2023)
        self.assertTrue(sp_num["is_anchored"])

    def test_audio_language_guard_v3_distinguishes_subtitles_and_multi_packs(self) -> None:
        p_lektor = apply_audio_language_guard_v3(
            "Synthetic.Chronicle.2024.PL.1080p.WEB-DL.[Lektor PL].[Napisy ENG-PL].mkv",
            p_en=0.78,
            p_title_foreign=0.20,
        )
        self.assertLessEqual(p_lektor, 0.15)

        p_dual_lektor = apply_audio_language_guard_v3(
            "Synthetic.Chronicle.2024.1080p.WEB-DL.[ENG-Lektor PL].mkv",
            p_en=0.82,
            p_title_foreign=0.10,
        )
        self.assertGreaterEqual(p_dual_lektor, 0.80)

        p_lektor_eng_aac = apply_audio_language_guard_v3(
            "Synthetic.Chronicle.2024.1080p.WEB-DL.[Lektor PL].[ENG AAC].mkv",
            p_en=0.35,
            p_title_foreign=0.30,
        )
        self.assertGreaterEqual(p_lektor_eng_aac, 0.85)

        p_lektor_dual = apply_audio_language_guard_v3(
            "Synthetic.Chronicle.2024.1080p.WEB-DL.[Lektor PL].[Dual Audio].mkv",
            p_en=0.40,
            p_title_foreign=0.30,
        )
        self.assertGreaterEqual(p_lektor_dual, 0.85)

        p_multi = apply_audio_language_guard(
            "Synthetic.Odyssey.2023.BDRip.1080p.[RUS + JAP + ENG].mkv",
            p_en=0.35,
            p_title_foreign=0.60,
        )
        self.assertGreaterEqual(p_multi, 0.85)

        anime_sub_only = classify_audio_and_anime_dub_v3(
            "[SynthSub] Synthetic Starlight - 04 [1080p][Eng Subs].mkv",
            p_en=0.72,
            p_anime=0.95,
            p_title_foreign=0.88,
        )
        self.assertTrue(anime_sub_only["is_anime"])
        self.assertFalse(anime_sub_only["is_anime_dub_or_dual"])

        anime_dual = classify_audio_and_anime_dub(
            "[SynthSub] Synthetic Starlight (2024) [1080p][Dual Audio][HEVC].mkv",
            p_en=0.42,
            p_anime=0.95,
            p_title_foreign=0.40,
        )
        self.assertTrue(anime_dual["is_anime"])
        self.assertTrue(anime_dual["is_english_compatible"])
        self.assertTrue(anime_dual["is_anime_dub_or_dual"])

    def test_resolve_category_v3_disambiguates_minority_and_zero_cue_classes(self) -> None:
        ambig_video_probs = [0.46, 0.44, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02]

        cat_map = {
            "title_year_map": {
                hash_catalog_key("synthetic starlight odyssey|2024"): "tv_show",
            }
        }
        raw_zc = "[SynthGroup] Synthetic Starlight Odyssey (2024) [1080p].mkv"
        spans = extract_byte_spans_v3(raw_zc)
        idx, src, _ = resolve_category_v3(
            raw_zc,
            ambig_video_probs,
            catalog=cat_map,
            extra_spans=[spans["title"]],
        )
        self.assertEqual(idx, IDX_TV)
        self.assertEqual(src, "v3_zero_cue_title_year")

        idx_comic, _, _ = resolve_category_v3("Synthetic_Chronicles_004_(2024)_(Digital).cbz", ambig_video_probs)
        self.assertEqual(idx_comic, IDX_COMIC)

        idx_ab, _, _ = resolve_category_v3("Synthetic Empire - Book 1 (2023) [Unabridged].m4b", ambig_video_probs)
        self.assertEqual(idx_ab, IDX_AUDIOBOOK)

        idx_course, _, _ = resolve_category_v3("Synthetic_Systems_Architecture_Lecture_Series_2024", ambig_video_probs)
        self.assertEqual(idx_course, IDX_COURSE)

        idx_game, _, _ = resolve_category_v3("Synthetic.Quest.v1.04.Plus.12.Trainer-SYNTH", ambig_video_probs)
        self.assertEqual(idx_game, IDX_GAME)

        idx_concert, _, _ = resolve_category_v3(
            "Synthetic.Philharmoniker.Live.in.Vienna.2023.1080i.BluRay.x264.mkv",
            [0.20, 0.05, 0.01, 0.65, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.03],
        )
        self.assertEqual(idx_concert, IDX_MOVIE)

        for check_name in (
            "Synthetic_Chronicles_004_(2024)_(Digital).cbz",
            "Synthetic.Odyssey.Movies.Collection.2024.1080p.BluRay.mkv",
            "Synthetic.Show.S01E02.1080p.WEB-DL.mkv",
        ):
            _, _, p_vec = resolve_category_v3(check_name, [0.45, 0.45, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02])
            self.assertAlmostEqual(sum(p_vec), 1.0, places=4)
            self.assertTrue(all(0.0 <= x <= 1.0 for x in p_vec))


    def test_encode_torrent_features_sorts_by_size_and_filters_padding(self) -> None:
        import math
        from armiger import armiger_inference as mod

        files = [{"path": f"__ia_thumb_{i:03d}.jpg", "size": 1024} for i in range(70)]
        files.extend(
            [{"path": f".____padding_file/{i}", "size": 4_000_000} for i in range(15)]
        )
        files.append({"path": "lecture_01.mp4", "size": 500_000_000})
        feats = mod.encode_torrent_features(
            {"name": "Synthetic_Course_18_06", "size": 560_071_680, "files": files}
        )
        self.assertEqual(int(feats["ext"][0]), mod._get_vocab_ext_map()["mp4"])
        self.assertGreaterEqual(int(feats["size"][0]), 28)
        # 71 non-padding files -> log1p(71)
        self.assertAlmostEqual(float(feats["stats"][0]), math.log1p(71), places=4)

        # Verify _apply_v3_post_neural_guard handles subtitle-trap and external .srt
        raw_item = {
            "category": {"label": "movie", "confidence": 0.91},
            "adult": {"label": "non_adult", "confidence": 0.99},
            "anime": {"label": "non_anime", "confidence": 0.60},
            "lang": {"label": "english", "confidence": 0.62},
            "english_subs": {"label": "no_eng_subs", "confidence": 0.58},
            "multi_audio": {"label": "single_audio", "confidence": 0.90},
            "audio_dubbed": {"label": "original_audio", "confidence": 0.90},
        }
        guarded = mod._apply_v3_post_neural_guard(
            {
                "name": "Synthetic-Asuka-Gumi-1990-english-subbed",
                "files": [{"path": "asuka_ep01.mp4", "size": 250_000_000}],
            },
            raw_item,
        )
        self.assertEqual(guarded["anime"]["label"], "anime")
        self.assertEqual(guarded["english_subs"]["label"], "has_eng_subs")
        self.assertEqual(guarded["lang"]["label"], "non_english")
        self.assertIn("matching_identity", guarded)
        self.assertEqual(guarded["matching_identity"]["release_year"], 1990)

        # Verify non-anime English release with Eng Subs is NOT forced to non_english
        en_with_subs = mod._apply_v3_post_neural_guard(
            {
                "name": "Synthetic.Horizon.2024.1080p.BluRay.ENG.AAC.Eng.Subs.mkv",
                "files": [{"path": "Synthetic.Horizon.2024.mkv", "size": 1_500_000_000}],
            },
            {
                "category": {"label": "movie", "confidence": 0.95},
                "adult": {"label": "non_adult", "confidence": 0.99},
                "anime": {"label": "non_anime", "confidence": 0.95},
                "lang": {"label": "english", "confidence": 0.92},
                "english_subs": {"label": "has_eng_subs", "confidence": 0.90},
                "title_language": {"label": "english_title", "confidence": 0.95},
            },
        )
        self.assertEqual(en_with_subs["lang"]["label"], "english")
        self.assertEqual(en_with_subs["english_subs"]["label"], "has_eng_subs")

    def test_c5_ultra_bridge_wires_v3_span_category_and_audio_plugins(self) -> None:
        import json
        import pathlib
        import tempfile
        try:
            from ultra_bridge import packet_from_input, run_v2_plugins
        except ImportError:
            self.skipTest("ultra_bridge is internal-only")
        from armiger.zero_cue_plugin import ZeroCueCatalogPlugin

        with tempfile.TemporaryDirectory() as tmp:
            cat_file = pathlib.Path(tmp) / "zc.json"
            cat_file.write_text(json.dumps({"format": "blake2b64", "title_year_map": {}}), encoding="utf-8")
            zc = ZeroCueCatalogPlugin(cat_file)

            raw_name = "Synthetic.Quest.v1.04.Plus.12.Trainer-SYNTH"
            rec = packet_from_input({"name": raw_name, "size": 500_000_000, "files_complete": False, "files": []})
            b = raw_name.encode("utf-8")
            tags = [BIO_O] * len(b)
            tags[0] = BIO_B_TITLE
            for i in range(1, 15):
                tags[i] = BIO_I_TITLE
            rec["bio_tags"] = tags

            base_probs = {
                "category": [0.46, 0.44, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02],
                "adult": [0.99, 0.01],
                "junk": [0.99, 0.01],
                "owner_scope": [0.90, 0.10],
                "lang": [0.85, 0.15, 0.0],
                "english_subs": [0.80, 0.20],
                "anime": [0.95, 0.05],
                "title_language": [0.95, 0.05],
            }
            v2 = run_v2_plugins(rec, base_probs, zero_cue=zc)
            self.assertEqual(v2["plugin_trace"]["zero_cue_rule"], "v3_structural_game")
            self.assertEqual(v2["plugin_trace"]["span_extractor"], "raw_neural_span_bio")
            self.assertEqual(v2["extraction"]["title"], "synthetic quest")
            self.assertGreaterEqual(v2["probs"]["category"][IDX_GAME], 0.88)

    def test_wave8_constrained_bio_viterbi_decoder_and_boundary_snapping(self) -> None:
        try:
            import numpy as np
        except ImportError:
            np = None  # type: ignore[assignment]

        raw_name = "[SynthGrp].Synthetic.Starlight.Odyssey.(2024).S01E02.1080p.WEB-DL.mkv"
        raw_bytes = raw_name.encode("utf-8")
        n = len(raw_bytes)
        logits_py = [[-4.0] * 7 for _ in range(n)]
        for row in logits_py:
            row[BIO_O] = 2.0

        # Inject greedy-illegal transitions:
        # 1) Start at t=0 with I-TITLE (illegal at t=0)
        logits_py[0][BIO_I_TITLE] = 5.0
        # 2) Leading delimiter '.' at index 10 and title span 11..38 with trailing '.('
        logits_py[10][BIO_B_TITLE] = 5.0
        for idx in range(11, 39):
            logits_py[idx][BIO_I_TITLE] = 5.0
        # 3) Partial year prediction on "(2024)" at bytes 40..44 (only 2 of 4 digits tagged with O -> I-YEAR)
        logits_py[40][BIO_I_YEAR] = 5.5
        logits_py[41][BIO_I_YEAR] = 5.5
        # 4) Cross-entity I-YEAR -> I-EP at 46..52 ("S01E02")
        for idx in range(46, 52):
            logits_py[idx][BIO_I_EP] = 5.0

        logits = np.asarray(logits_py, dtype=np.float64) if np is not None else logits_py
        greedy_tags = [max(range(7), key=lambda k: r[k]) for r in logits_py]
        self.assertGreater(count_illegal_bio_transitions(greedy_tags), 0)
        self.assertTrue(is_illegal_bio_transition(BIO_O, BIO_I_TITLE, is_start=False))
        self.assertTrue(is_illegal_bio_transition(0, BIO_I_TITLE, is_start=True))
        self.assertFalse(is_illegal_bio_transition(BIO_I_TITLE, BIO_B_TITLE, is_start=False))
        self.assertFalse(is_illegal_bio_transition(BIO_B_TITLE, BIO_B_TITLE, is_start=False))
        self.assertTrue(is_illegal_bio_transition(BIO_I_YEAR, BIO_I_EP, is_start=False))

        viterbi_tags = decode_bio_logits_viterbi(logits, raw_bytes)
        if np is not None:
            self.assertEqual(viterbi_tags.shape, (n,))
        self.assertEqual(len(viterbi_tags), n)
        self.assertEqual(count_illegal_bio_transitions(viterbi_tags), 0)

        # Verify leading/trailing delimiter '.' and '(' are stripped to BIO_O and B-TITLE starts at 'S' (index 11)
        self.assertEqual(int(viterbi_tags[10]), BIO_O)
        self.assertEqual(int(viterbi_tags[11]), BIO_B_TITLE)
        self.assertEqual(int(viterbi_tags[38]), BIO_O)

        # Verify 4-digit year "2024" at bytes 40..44 is snapped cleanly to [B-YEAR, I-YEAR, I-YEAR, I-YEAR]
        self.assertEqual(
            [int(x) for x in viterbi_tags[40:44]],
            [BIO_B_YEAR, BIO_I_YEAR, BIO_I_YEAR, BIO_I_YEAR],
        )

        decoded = decode_bio_tags_raw_neural(raw_name, viterbi_tags)
        self.assertEqual(key_norm(decoded["title"]), "synthetic starlight odyssey")
        self.assertEqual(decoded["year"], 2024)
        self.assertEqual(decoded["episode_spec"], "S01E02")

        # Verify Round 11-12 preservation:
        # (a) Release-metadata lookahead after language words so titles containing 'French' are not truncated
        fr_title = extract_byte_spans_v3("The.French.Dispatch.2021.1080p.mkv")
        self.assertEqual(key_norm(fr_title["title"]), "the french dispatch")
        self.assertEqual(fr_title["year"], 2021)

        # (b) Byte-exact year span offset even when a BIO year run starts on a UTF-8 continuation byte
        utf8_name = "Étoile.2024.1080p.mkv"
        utf8_bytes = utf8_name.encode("utf-8")
        yr_pos = utf8_bytes.index(b"2024")
        utf8_tags = [BIO_O] * len(utf8_bytes)
        utf8_tags[0] = BIO_B_TITLE
        for k in range(1, yr_pos - 1):
            utf8_tags[k] = BIO_I_TITLE
        # Start B-YEAR on the UTF-8 continuation byte of 'É' (index 1) or delimiter before 2024
        utf8_tags[1] = BIO_B_YEAR
        for k in range(2, yr_pos + 4):
            utf8_tags[k] = BIO_I_YEAR
        utf8_res = decode_bio_tags_raw_neural(utf8_name, utf8_tags)
        self.assertEqual(utf8_res["year"], 2024)
        self.assertEqual(utf8_res["year_byte_span"], (yr_pos, yr_pos + 4))

        # (c) Verify UltraBridge.run_v2_plugins accepts a 2D NumPy array in record["bio_logits"] without TruthValueError
        if np is not None:
            try:
                from ultra_bridge import run_v2_plugins
            except ImportError:
                run_v2_plugins = None  # type: ignore[assignment]
            if run_v2_plugins is not None:
                np_logits = np.asarray(logits, dtype=np.float32)
                base_probs = {
                    "category": [0.90] + [0.01] * 10,
                    "adult": [0.99, 0.01],
                    "junk": [0.98, 0.02],
                    "anime": [0.99, 0.01],
                    "english_subs": [0.10, 0.90],
                    "lang": [0.95, 0.04, 0.01],
                    "owner_scope": [0.95, 0.05],
                    "title_language": [0.96, 0.04],
                }
                v2_res = run_v2_plugins(
                    {"row": {"name": raw_name, "size": 1_500_000_000, "files": []}, "bio_logits": np_logits},
                    base_probs,
                )
                self.assertEqual(key_norm(v2_res["extraction"]["title"]), "synthetic starlight odyssey")
                self.assertEqual(v2_res["extraction"]["year"], 2024)

        # (d) Partial title word boundaries expand to full token boundaries without overwriting adjacent entities
        snapped_left = snap_bio_span_boundaries([0, 1, 2, 2, 2, 2, 2, 2, 2], b"Synthetic")
        self.assertEqual([int(x) for x in snapped_left], [1, 2, 2, 2, 2, 2, 2, 2, 2])
        snapped_right = snap_bio_span_boundaries([1, 2, 2, 2, 2, 0, 0, 0, 0], b"Synthetic")
        self.assertEqual([int(x) for x in snapped_right], [1, 2, 2, 2, 2, 2, 2, 2, 2])
        snapped_adj = snap_bio_span_boundaries(
            [0, 1, 2, 2, 0, 0, 0, 3, 4, 4, 4],
            b"Synthetic2024"[2:],  # b"nthetic2024" -> title stops before 2024
        )
        self.assertEqual([int(x) for x in snapped_adj[:7]], [1, 2, 2, 2, 2, 2, 2])
        self.assertEqual([int(x) for x in snapped_adj[7:]], [3, 4, 4, 4])

        # (e) Hard -inf transition constraints hold even under extreme finite logits (2e9) and NumPy/Python parity holds
        import armiger.byte_span_extractor as bse_mod
        extreme_logits = [[0.0, 0.0, 2e9, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 2e9, 0.0, 0.0]]
        np_extreme = [int(x) for x in decode_bio_logits_viterbi(extreme_logits)]
        self.assertEqual(count_illegal_bio_transitions(np_extreme), 0)
        orig_np = bse_mod.np
        try:
            bse_mod.np = None
            py_extreme = [int(x) for x in decode_bio_logits_viterbi(extreme_logits)]
        finally:
            bse_mod.np = orig_np
        self.assertEqual(np_extreme, py_extreme)
        self.assertEqual(count_illegal_bio_transitions(py_extreme), 0)

        # (f) Relative scores of large finite logits (e.g. O=100, B-TITLE=200) are preserved in both NumPy and Python
        rel_logits = [[100.0, 200.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        self.assertEqual([int(x) for x in decode_bio_logits_viterbi(rel_logits)], [BIO_B_TITLE])
        try:
            bse_mod.np = None
            self.assertEqual([int(x) for x in decode_bio_logits_viterbi(rel_logits)], [BIO_B_TITLE])
        finally:
            bse_mod.np = orig_np

        # (g) Adjacent same-type entities (I-TITLE -> B-TITLE) are permitted as the optimal legal path
        adj_same_type_logits = [
            [0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # B-TITLE
            [0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0],  # I-TITLE
            [0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # B-TITLE (adjacent second entity)
            [0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0],  # I-TITLE
        ]
        adj_decoded = [int(x) for x in decode_bio_logits_viterbi(adj_same_type_logits, b"AbCd")]
        self.assertEqual(adj_decoded, [BIO_B_TITLE, BIO_I_TITLE, BIO_B_TITLE, BIO_I_TITLE])
        self.assertEqual(count_illegal_bio_transitions(adj_decoded), 0)

    def test_wave8_owner_policy_rule3_localized_foreign_alias_guard(self) -> None:
        # 1. Localized Romance/Germanic alias stem without English audio/subs -> Rule 3 disallow (including accented variants)
        is_stem, reason = detect_localized_foreign_title_stem("Haus.Des.Geldes.S01E01.1080p.WEB-DL.mkv")
        self.assertTrue(is_stem)
        self.assertEqual(reason, "localized_romance_or_germanic_stem")
        for accented, unaccented in (
            ("El.Señor.De.Los.Anillos.2001.1080p.mkv", "El.Senor.De.Los.Anillos.2001.1080p.mkv"),
            ("La.Guerre.Des.Étoiles.1977.1080p.mkv", "La.Guerre.Des.Etoiles.1977.1080p.mkv"),
            ("Pirates.Des.Caraïbes.2003.1080p.mkv", "Pirates.Des.Caraibes.2003.1080p.mkv"),
        ):
            r_acc = evaluate_owner_policy_rule3(accented, p_en=0.62, p_title_foreign=0.80, p_anime=0.01, p_eng_subs_yes=0.05)
            r_unacc = evaluate_owner_policy_rule3(unaccented, p_en=0.62, p_title_foreign=0.80, p_anime=0.01, p_eng_subs_yes=0.05)
            self.assertTrue(r_acc["is_localized_foreign_alias"])
            self.assertTrue(r_acc["rule3_disallow"])
            self.assertEqual(r_acc["rule3_disallow"], r_unacc["rule3_disallow"])

        r3_disallow = evaluate_owner_policy_rule3(
            "El.Juego.Del.Calamar.S01E01.1080p.WEB-DL.mkv",
            p_en=0.62,
            p_title_foreign=0.80,
            p_anime=0.01,
            p_eng_subs_yes=0.05,
        )
        self.assertTrue(r3_disallow["is_localized_foreign_alias"])
        self.assertFalse(r3_disallow["has_verified_english_support"])
        self.assertTrue(r3_disallow["rule3_disallow"])

        # 2. Localized alias with explicit English subs or Dual-Audio -> Rule 3 allowed
        r3_with_subs = evaluate_owner_policy_rule3(
            "El.Juego.Del.Calamar.S01E01.ENG.SUBS.1080p.WEB-DL.mkv",
            p_en=0.25,
            p_title_foreign=0.80,
            p_anime=0.01,
            p_eng_subs_yes=0.92,
        )
        self.assertTrue(r3_with_subs["is_localized_foreign_alias"])
        self.assertTrue(r3_with_subs["has_verified_english_support"])
        self.assertFalse(r3_with_subs["rule3_disallow"])

        # 3. Canonical foreign-original works (including La Casa de Papel, La Haine, Cyrillic originals, and unlisted originals) are shielded
        for whitelisted_orig in ("La.Casa.De.Papel.S01E01.1080p.mkv", "La.Haine.1995.1080p.BluRay.x264.mkv", "Сталкер.1979.1080p.mkv"):
            is_stem_orig, stem_orig_reason = detect_localized_foreign_title_stem(whitelisted_orig)
            self.assertFalse(is_stem_orig)
            self.assertEqual(stem_orig_reason, "foreign_original_work")
            is_orig_alias, orig_reason = is_localized_foreign_alias(whitelisted_orig, p_en=0.90, p_title_foreign=0.10)
            self.assertFalse(is_orig_alias)
            self.assertEqual(orig_reason, "none")
            r3_orig = evaluate_owner_policy_rule3(whitelisted_orig, p_en=0.90, p_title_foreign=0.10)
            self.assertFalse(r3_orig["is_localized_foreign_alias"])
            self.assertFalse(r3_orig["rule3_disallow"])
            rep_w = classify_audio_and_anime_dub_v3(whitelisted_orig, p_en=0.90, p_anime=0.01, p_title_foreign=0.10)
            self.assertFalse(rep_w["is_localized_foreign_alias"])
            self.assertTrue(rep_w["is_english_compatible"])
        for orig_title in ("Le.Salaire.De.La.Peur.1953.1080p.BluRay.mkv", "La.Guerre.Des.Boutons.1962.1080p.mkv", "Иди.и.смотри.1985.1080p.mkv"):
            is_unlisted_alias, _ = is_localized_foreign_alias(orig_title, p_en=0.90, p_title_foreign=0.10)
            self.assertFalse(is_unlisted_alias)
            r3_unlisted = evaluate_owner_policy_rule3(orig_title, p_en=0.90, p_title_foreign=0.10)
            self.assertFalse(r3_unlisted["is_localized_foreign_alias"])
            self.assertFalse(r3_unlisted["rule3_disallow"])
            rep_u = classify_audio_and_anime_dub_v3(orig_title, p_en=0.90, p_anime=0.01, p_title_foreign=0.10)
            self.assertFalse(rep_u["is_localized_foreign_alias"])
            self.assertTrue(rep_u["is_english_compatible"])

        # 4. classify_audio_and_anime_dub_v3 integrates Rule 3 and suppresses unverified localized alias audio
        rep = classify_audio_and_anime_dub_v3(
            "Le.Seigneur.Des.Anneaux.2022.S01E01.1080p.mkv",
            p_en=0.72,
            p_anime=0.02,
            p_title_foreign=0.85,
            p_eng_subs_yes=0.05,
        )
        self.assertTrue(rep["is_localized_foreign_alias"])
        self.assertTrue(rep["owner_policy_rule3"]["rule3_disallow"])
        self.assertLessEqual(rep["p_english_compatible_audio"], 0.15)
        self.assertFalse(rep["is_english_compatible"])

        # 5. Underscore-delimited _ENG_ audio tag is recognized without subtitle evidence, and 5-arg positional plugin_rules works
        rep_underscore = classify_audio_and_anime_dub_v3(
            "El.Juego.Del.Calamar.S01E01_ENG_1080p.mkv",
            0.78,
            0.01,
            0.80,
            {"en_threshold": 0.50},
        )
        self.assertTrue(rep_underscore["owner_policy_rule3"]["has_explicit_eng_audio"])
        self.assertFalse(rep_underscore["owner_policy_rule3"]["rule3_disallow"])
        self.assertTrue(rep_underscore["is_english_compatible"])

        # 6. Excluded non-video audio/book payloads (.mp3, .epub) are shielded across all alias fallbacks
        r3_mp3 = evaluate_owner_policy_rule3(
            "Synthetic.Foreign.Title.mp3",
            p_en=0.60,
            p_title_foreign=0.80,
            p_anime=0.01,
            p_eng_subs_yes=0.0,
        )
        self.assertFalse(r3_mp3["is_localized_foreign_alias"])
        self.assertFalse(r3_mp3["rule3_disallow"])
        rep_mp3 = classify_audio_and_anime_dub_v3(
            "Synthetic.Foreign.Title.mp3",
            p_en=0.60,
            p_anime=0.01,
            p_title_foreign=0.80,
            p_eng_subs_yes=0.0,
        )
        self.assertTrue(rep_mp3["is_english_compatible"])

    def test_wave9_structural_file_tree_disambiguation(self) -> None:
        from armiger.zero_cue_plugin import IDX_EBOOK, _extract_file_surface, apply_structural_and_zero_cue_disambiguation

        uniform = [1.0 / 11.0] * 11
        # 1. Multi-file structured course tree resolves to course before tv_show
        course_files = [
            {"path": "01 Introduction/01_01_welcome.mp4"},
            {"path": "01 Introduction/01_02_setup.mp4"},
            {"path": "02 Core Concepts/module_02_slides.pdf"},
        ]
        surf, exts, cnt = _extract_file_surface(course_files)
        self.assertEqual(cnt, 3)
        self.assertIn("mp4", exts)
        self.assertIn("pdf", exts)
        idx_c, src_c = apply_structural_and_zero_cue_disambiguation(
            "Synthetic.System.Design.Bootcamp.2024",
            uniform,
            p_adult=0.01,
            p_anime=0.01,
            files=course_files,
        )
        self.assertEqual(idx_c, IDX_COURSE)
        self.assertEqual(src_c, "structural_course_book")

        # 1b. Underscore-delimited module/lecture filenames also match _COURSE_FILE_TREE_RE
        underscore_course_files = [
            {"path": "module_01_intro.mp4"},
            {"path": "module_02_basics.mp4"},
            {"path": "module_03_advanced.mp4"},
        ]
        idx_cu, src_cu, _ = resolve_category_v3(
            "Synthetic.Architecture.Series.2024",
            uniform,
            p_adult=0.01,
            p_anime=0.01,
            files=underscore_course_files,
        )
        self.assertEqual(idx_cu, IDX_COURSE)
        self.assertEqual(src_cu, "v3_structural_course")

        # 2. LibriVox / chapter MP3 file tree resolves to audiobook
        ab_files = [
            {"path": "chapter_01.mp3"},
            {"path": "chapter_02.mp3"},
        ]
        idx_ab, src_ab, _ = resolve_category_v3(
            "synthetic_classic_novel_2103",
            uniform,
            p_adult=0.01,
            p_anime=0.01,
            files=ab_files,
        )
        self.assertEqual(idx_ab, IDX_AUDIOBOOK)
        self.assertEqual(src_ab, "v3_structural_audiobook")

        # 3. Scoped _V3_SOFTWARE_RE does not hijack multilingual EPUB/PDF eBooks
        idx_eb, src_eb, _ = resolve_category_v3(
            "Synthetic.Multilingual.Dictionary.2023.epub",
            uniform,
            p_adult=0.01,
            p_anime=0.01,
            files=[{"path": "Synthetic.Multilingual.Dictionary.2023.epub"}],
        )
        self.assertEqual(idx_eb, IDX_EBOOK)
        self.assertEqual(src_eb, "v3_structural_ebook")

        # 4. Mixed game repack + bonus digital comic (.cbz) stays game in both resolvers
        game_probs = [0.01] * 11
        game_probs[IDX_GAME] = 0.90
        mixed_game_files = [
            {"path": "setup.exe", "size": 5_000_000},
            {"path": "fg-01.bin", "size": 4_000_000_000},
            {"path": "Bonus/Digital_Comic.cbz", "size": 15_000_000},
        ]
        idx_gm1, src_gm1 = apply_structural_and_zero_cue_disambiguation(
            "Synthetic.Cyber.Quest.v2.1.FitGirl.Repack",
            game_probs,
            p_adult=0.01,
            p_anime=0.01,
            files=mixed_game_files,
        )
        self.assertEqual(idx_gm1, IDX_GAME)
        idx_gm2, src_gm2, _ = resolve_category_v3(
            "Synthetic.Cyber.Quest.v2.1.FitGirl.Repack",
            game_probs,
            p_adult=0.01,
            p_anime=0.01,
            files=mixed_game_files,
        )
        self.assertEqual(idx_gm2, IDX_GAME)
        self.assertEqual(src_gm2, "v3_structural_game")

        # 5. Mixed game repack + bonus audiobook (.m4b) stays game in both resolvers
        mixed_game_m4b_files = [
            {"path": "setup.exe", "size": 5_000_000},
            {"path": "fg-01.bin", "size": 4_000_000_000},
            {"path": "Bonus/Novel.m4b", "size": 45_000_000},
        ]
        idx_gm_ab1, _ = apply_structural_and_zero_cue_disambiguation(
            "Synthetic.Cyber.Quest.v2.1.FitGirl.Repack",
            game_probs,
            p_adult=0.01,
            p_anime=0.01,
            files=mixed_game_m4b_files,
        )
        self.assertEqual(idx_gm_ab1, IDX_GAME)
        idx_gm_ab2, src_gm_ab2, _ = resolve_category_v3(
            "Synthetic.Cyber.Quest.v2.1.FitGirl.Repack",
            game_probs,
            p_adult=0.01,
            p_anime=0.01,
            files=mixed_game_m4b_files,
        )
        self.assertEqual(idx_gm_ab2, IDX_GAME)
        self.assertEqual(src_gm_ab2, "v3_structural_game")

        # 6. Disqualifying payload after entry 64 is still collected in file_exts and blocks book/comic override
        late_exe_files = [{"path": f"Bonus/Page_{i:03d}.cbz", "size": 1_000_000} for i in range(70)] + [
            {"path": "setup.exe", "size": 50_000_000}
        ]
        _, late_exts, late_cnt = _extract_file_surface(late_exe_files)
        self.assertEqual(late_cnt, 71)
        self.assertIn("exe", late_exts)
        idx_late, _ = apply_structural_and_zero_cue_disambiguation(
            "Synthetic.Generic.Bundle.v1.0",
            game_probs,
            p_adult=0.01,
            p_anime=0.01,
            files=late_exe_files,
        )
        self.assertEqual(idx_late, IDX_GAME)

        # 7. Mixed game repack + bonus module/lecture PDF stays game in both resolvers
        mixed_game_course_files = [
            {"path": "setup.exe", "size": 5_000_000},
            {"path": "fg-01.bin", "size": 4_000_000_000},
            {"path": "Bonus/module_01_slides.pdf", "size": 2_000_000},
        ]
        idx_gm_c1, _ = apply_structural_and_zero_cue_disambiguation(
            "Synthetic.Cyber.Quest.v2.1.FitGirl.Repack",
            game_probs,
            p_adult=0.01,
            p_anime=0.01,
            files=mixed_game_course_files,
        )
        self.assertEqual(idx_gm_c1, IDX_GAME)
        idx_gm_c2, src_gm_c2, _ = resolve_category_v3(
            "Synthetic.Cyber.Quest.v2.1.FitGirl.Repack",
            game_probs,
            p_adult=0.01,
            p_anime=0.01,
            files=mixed_game_course_files,
        )
        self.assertEqual(idx_gm_c2, IDX_GAME)
        self.assertEqual(src_gm_c2, "v3_structural_game")

    def test_wave10_concat_english_slug_and_stalled_swarm_honeypot_shield(self) -> None:
        from armiger.swarm_survival import SwarmResidualAdapter
        from armiger.zero_cue_plugin import parse_clean_candidates

        # 1. Concatenated LibriVox/Gutenberg English slug segmentation + non-English language marker guard
        slug_res = split_and_score_title_slug("thepaintedveil_2103_librivox")
        self.assertTrue(slug_res["is_english_slug"] and slug_res["found_concat_slug"])
        self.assertEqual(slug_res["segmented_words"][:3], ["the", "painted", "veil"])
        self.assertFalse(split_and_score_title_slug("germinal_french_librivox")["is_english_slug"])
        self.assertFalse(split_and_score_title_slug("faust_german_librivox")["is_english_slug"])
        tl_fr = apply_title_language_guard("germinal_french_librivox", [0.20, 0.80], [0.15, 0.85])
        self.assertLess(tl_fr[0], 0.50)

        # 2. apply_title_language_guard + parse_clean_candidates for English slugs
        tl_probs = apply_title_language_guard("thesecretgarden_1908_librivox", [0.35, 0.65], [0.20, 0.80])
        self.assertGreaterEqual(tl_probs[0], 0.88)
        cands, _ = parse_clean_candidates("thepaintedveil_2103_librivox")
        self.assertIn("the painted veil", [c[0] for c in cands])

        # 3. Stalled legitimate swarm is not blocked; high neural junk on seeded swarm still blocks
        adapter = SwarmResidualAdapter()
        stalled_eval = adapter.evaluate_swarm(
            title="Synthetic.Documentary.2024.1080p.WEB-DL.x264.mkv", seeders=0, leechers=85, completed=0,
            total_size_bytes=2_400_000_000, files=[{"path": "Synthetic.Documentary.2024.1080p.WEB-DL.x264.mkv", "size": 2_400_000_000}],
            category_pred="movie", p_junk_neural=0.04,
        )
        self.assertEqual(stalled_eval["category"], "movie")
        self.assertLess(stalled_eval["trap_score"], 0.45)
        self.assertFalse(stalled_eval["is_honeypot_blocked"])

        junk_eval = adapter.evaluate_swarm(
            title="Synthetic.Suspicious.Release.2024", seeders=18, leechers=3, completed=45,
            total_size_bytes=800_000_000, files=[{"path": "Synthetic.Suspicious.Release.2024.dat", "size": 800_000_000}],
            category_pred="software", p_junk_neural=0.95,
        )
        self.assertLess(junk_eval["trap_score"], 0.35)
        self.assertGreaterEqual(junk_eval["p_honeypot_or_junk"], 0.90)
        self.assertTrue(junk_eval["is_honeypot_blocked"])


if __name__ == "__main__":
    unittest.main()
