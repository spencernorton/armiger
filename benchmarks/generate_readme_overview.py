"""Render the plain-language README overview from the canonical benchmark manifest."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "benchmarks/CHARTS_MANIFEST.json"
IMAGE = "docs/assets/armiger_overview.png"
RECORD = "benchmarks/README_OVERVIEW.json"
ULTRA = "Armiger Ultra (Fair Hybrid)"
CATEGORIES = [
    ("Movie", "Movies"), ("TV Show", "TV shows"), ("Music", "Music"),
    ("Audiobook", "Audiobooks"), ("Ebook", "eBooks"), ("Comic", "Comics"),
    ("Course", "Courses"), ("Game", "Games"), ("Software", "Software"),
    ("Adult / XXX", "Adult content"),
]
CLASSIFIERS = [
    (ULTRA, "armiger_ultra_pct"),
    ("MIMMO BERT (360M)", "mimmo_bert_pct"),
    ("Bitmagnet Built-In", "bitmagnet_pct"),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_source(root: Path = ROOT) -> dict:
    manifest = json.loads((root / SOURCE).read_text())
    for name, expected in manifest["provenance_sha256"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Invalid provenance path: {name}")
        path = root / "benchmarks" / relative
        if not path.is_file():
            path = root / relative
        if sha256(path) != expected:
            raise ValueError(f"Canonical provenance mismatch: {name}")
    return manifest


def score(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 100:
        raise ValueError("Benchmark score must be finite and between 0 and 100")
    return value


def panel(source: dict, name: str, metric: str, models: list[str]) -> dict:
    if source["metric_type"] != metric:
        raise ValueError(f"Unexpected metric for {name}")
    n = source["n"]
    if not isinstance(n, int) or n <= 0:
        raise ValueError(f"Invalid sample count for {name}")
    values = {}
    for model in models:
        result = source["models"][model]
        if result["n"] != n:
            raise ValueError(f"Comparison denominators differ for {name}")
        values[model] = score(result["point_pct"])
    return {"metric": metric, "n": n, "scores": values}


def build_data(manifest: dict) -> dict:
    charts = manifest["charts"]
    classifiers = charts["armiger_vs_torrent_classifiers_6dims.png"]
    teachers = charts["armiger_vs_llms_overall_and_6dims.png"]
    categories = charts["armiger_categorical_breakdown.png"]["left_oct15_confirmation_n2835"]
    rows = []
    for key, label in CATEGORIES:
        row = categories[key]
        if not isinstance(row["n"], int) or row["n"] <= 0:
            raise ValueError(f"Invalid sample count for {label}")
        rows.append({"label": label, "n": row["n"],
                     "scores": {name: score(row[field]) for name, field in CLASSIFIERS}})
    overall = panel(classifiers["panel_1_category_11class_oct15"], "overall", "mean",
                    [name for name, _ in CLASSIFIERS])
    other_n = overall["n"] - sum(row["n"] for row in rows)
    if other_n < 0:
        raise ValueError("Category counts exceed the overall cohort")
    comparison = [ULTRA, "GuessIt", "RTN (rank-torrent-name)"]
    return {
        "model_version": "3.0.0", "overall": overall, "categories": rows,
        "other_category_n": other_n,
        "title_matching": panel(classifiers["panel_2_catalog_matching_oct15"],
                                "title matching", "mean", comparison),
        "audio_language": panel(classifiers["panel_3_audio_strict_oct15"],
                                "audio language", "mean", comparison),
        "junk": panel(teachers["panel_2e_junk_balanced_977"], "junk", "balanced", [ULTRA]),
        "seeds": panel(teachers["panel_2f_swarm_warehouse_10278"], "seeds", "auroc",
                       [ULTRA, "Raw Seeder Count"]),
    }


def render(data: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    from matplotlib.patches import FancyBboxPatch
    from PIL import Image

    bg, surface, border = "#0d1117", "#151c25", "#2b3745"
    white, muted, green = "#edf4fa", "#9caec0", "#3edba8"
    with plt.rc_context({"font.family": "DejaVu Sans", "text.color": white}):
        fig = plt.figure(figsize=(16, 10), dpi=120, facecolor=bg)
        ax = fig.add_axes((0, 0, 1, 1))
        ax.set(xlim=(0, 1600), ylim=(1000, 0))
        ax.axis("off")

        def box(x, y, w, h, color=surface, edge=border, radius=14):
            ax.add_patch(FancyBboxPatch((x, y), w, h,
                         boxstyle=f"round,pad=0,rounding_size={radius}",
                         facecolor=color, edgecolor=edge, linewidth=0.8))

        def text(x, y, value, size=14, color=white, weight="normal", align="left"):
            ax.text(x, y, value, fontsize=size, color=color, weight=weight,
                    ha=align, va="top", linespacing=1.25)

        def rule(y):
            ax.plot((72, 866), (y, y), color=border, linewidth=0.6)

        text(50, 35, "ARMIGER ULTRA  /  MODEL v3.0.0", 11, muted, "bold")
        text(50, 75, "How Armiger Ultra compares", 31, weight="bold")
        text(50, 135, "Content recognition, title matching and extra metadata checks", 16, muted)
        box(45, 190, 850, 704)
        text(73, 214, "Correct content category", 20, weight="bold")
        text(73, 257, "Same labelled items for each classifier", 12, muted)
        centres = [409, 589, 769]
        for x, label, color in zip(centres, ["Armiger\nUltra", "MIMMO\nBERT", "Bitmagnet\nbuilt-in"],
                                   [green, muted, muted]):
            text(x, 294, label, 13, color, "bold", "center")
        text(297, 309, "ITEMS", 8.5, muted, align="right")
        rule(342)
        table_rows = [{"label": "Overall", **data["overall"]}, *data["categories"]]
        for index, row in enumerate(table_rows):
            y = 365 + index * 43
            text(74, y - 9, row["label"], 15.4, weight="bold" if index == 0 else "normal")
            text(297, y - 4, f"{row['n']:,}", 10.5, muted, align="right")
            for column, ((model, _), x) in enumerate(zip(CLASSIFIERS, centres)):
                value = row["scores"][model]
                base = to_rgb(green if column == 0 else "#80a7db")
                dark = to_rgb(surface)
                strength = 0.08 + 0.25 * value / 100
                color = tuple(a * (1 - strength) + b * strength for a, b in zip(dark, base))
                box(x - 78, y - 14, 156, 35, color, color, 7)
                text(x, y - 9, f"{value:.0f}%", 16.1, green if column == 0 else white,
                     "bold" if column == 0 else "normal", "center")
            if index == 0:
                rule(y + 28)
        text(74, 849, "0% = no correct labels in that row; tools cover different content types.", 9.5, muted)
        text(74, 873, f"Overall also includes {data['other_category_n']} other / unknown items.", 9.5, muted)

        text(930, 214, "Ultra’s extra checks", 20, weight="bold")
        text(930, 257, "Separate tests for each job", 12, muted)

        def card(x, y, title, key, unit, comparisons, note="", suffix="%"):
            entry = data[key]
            box(x, y, 292, 277)
            text(x + 22, y + 21, title, 16.5, weight="bold")
            text(x + 22, y + 62, f"{entry['scores'][ULTRA]:.0f}{suffix}", 39, green, "bold")
            text(x + 22, y + 132, unit, 11.3)
            for line, value in enumerate(comparisons):
                text(x + 22, y + 173 + line * 24, value, 11.0, muted)
            if note:
                text(x + 22, y + 219, note, 9.2, muted)
            text(x + 22, y + 251, f"{entry['n']:,} test items", 10, muted)

        title = data["title_matching"]["scores"]
        audio = data["audio_language"]["scores"]
        card(930, 302, "Title matching", "title_matching", "correct movie / show matches", [
            f"GuessIt {title['GuessIt']:.0f}%   ·   RTN {title['RTN (rank-torrent-name)']:.0f}%"])
        card(1247, 302, "Audio language", "audio_language", "English vs other audio", [
            f"GuessIt {audio['GuessIt']:.0f}%   ·   RTN {audio['RTN (rank-torrent-name)']:.0f}%"])
        card(930, 605, "Junk & malware cues", "junk", "junk / clean detection score", [
            "Other tools: no comparable", "score in this test"], "Metadata cues, not a file scan")
        card(1247, 605, "Seed availability", "seeds", "live-swarm ranking score", [
            f"Seed count alone: {data['seeds']['scores']['Raw Seeder Count']:.0f}/100"],
            suffix="/100")
        text(50, 923, "Rounded benchmark results, not guarantees. Seed scores measure ranking quality, not percentage correct.", 11, muted)
        text(50, 953, "Junk scoring weighs clean and junk items equally. Full test sets, methods and limitations are in the evaluation guide.", 11, muted)
        output = io.BytesIO()
        fig.savefig(output, format="png", dpi=120, facecolor=bg)
        plt.close(fig)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.open(io.BytesIO(output.getvalue())).convert("RGB").save(path, optimize=False, compress_level=9)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root
    data = build_data(load_source(root))
    render(data, root / IMAGE)
    record = {
        "schema_version": 1, "source_manifest": SOURCE,
        "source_manifest_sha256": sha256(root / SOURCE),
        "renderer": "benchmarks/generate_readme_overview.py",
        "renderer_sha256": sha256(Path(__file__)),
        "image": IMAGE, "image_sha256": sha256(root / IMAGE), "data": data,
    }
    (root / RECORD).write_text(json.dumps(record, indent=2) + "\n")
    print(f"Wrote {IMAGE} and {RECORD}")


if __name__ == "__main__":
    main()
