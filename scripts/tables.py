import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

SETTINGS = ["vit_b32_lora", "vit_l14_lora", "llama3_8b_lora", "vit_b32_full", "vit_l14_full", "llama32_1b_full"]


def mean_std(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 1:
        return f"{values[0]:.2f}"
    return f"{values.mean():.2f} ± {values.std(ddof=1):.2f} (n={len(values)})"


def table(title, header, rows):
    lines = [f"\n### {title}\n", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def collect(folder: Path, key):
    groups = defaultdict(list)
    if folder.exists():
        for metrics in sorted(folder.rglob("metrics.json")):
            groups[key(metrics)].append(json.load(open(metrics)))
    return groups


def experiment_of(metrics: Path) -> str:
    return metrics.parents[1].name


def report(out: Path) -> str:
    parts = ["# Results (normalised test accuracy, %; mean ± sample std over seeds where several runs exist)"]

    main = collect(out / "main", lambda p: experiment_of(p).replace("experiments_", ""))
    for name, ms in collect(out / "zeroshot", lambda p: experiment_of(p).replace("experiments_", "").rsplit("_", 1)[0] + "_zeroshot").items():
        main[name] = ms
    rows = defaultdict(list)
    for name, ms in sorted(main.items()):
        for setting in SETTINGS:
            if name.startswith(setting + "_"):
                method = name[len(setting) + 1:]
                rows[setting].append([method, mean_std([m["avg/norm_acc"] for m in ms]), mean_std([m["avg/acc"] for m in ms])])
    for setting in SETTINGS:
        if rows[setting]:
            parts.append(table(f"Main results: {setting} (Tab. 1, 2, 12)", ["method", "norm", "abs"], rows[setting]))

    for name, folder, label in (("ablation", out / "ablation", "Ablation (Tab. 3, 4)"), ("sample_sweep", out / "sample_sweep", "Calibration examples per task (Tab. 6)"),
                                ("calibration_sensitivity", out / "calibration_sensitivity", "Calibration set (Tab. 5)")):
        groups = collect(folder, lambda p: (p.parents[3].name, p.parents[2].name))
        by = defaultdict(list)
        for (setting, variant), ms in sorted(groups.items()):
            by[setting].append([variant, mean_std([m["avg/norm_acc"] for m in ms])])
        for setting, r in by.items():
            parts.append(table(f"{label}: {setting}", ["variant", "norm"], r))

    for setting_dir in sorted((out / "synthetic").glob("*")) if (out / "synthetic").exists() else []:
        r = []
        for f in sorted(setting_dir.glob("*.json")):
            d = json.load(open(f))
            test = d.get("test", d)
            r.append([f.stem, f"{test['avg/norm_acc']:.2f}", d.get("selected", "")])
        parts.append(table(f"Synthetic calibration data (Tab. 7): {setting_dir.name}", ["method", "norm", "selected on synthetic images"], r))

    for setting_dir in sorted((out / "matched").glob("*")) if (out / "matched").exists() else []:
        cells = defaultdict(lambda: defaultdict(list))
        ns = set()
        for kind in ("merge", "finetune"):
            for f in (setting_dir / kind).glob("*.json"):
                d = json.load(open(f))
                cells[(d["method"], kind)][d["n"]].append(d["test"]["avg/norm_acc"])
                ns.add(d["n"])
        ns = sorted(ns)
        methods = sorted({m for m, _ in cells}, key=lambda m: (m != "zeroshot", m))
        r = []
        for m in methods:
            r.append([m] + [f"{np.mean(cells[(m, 'merge')][n]):.1f}" if cells[(m, "merge")][n] else "–" for n in ns]
                     + [f"{np.mean(cells[(m, 'finetune')][n]):.1f}" if cells[(m, "finetune")][n] else "–" for n in ns])
        parts.append(table(f"Merging vs fine-tuning with the same n examples (Tab. 8): {setting_dir.name}",
                           ["method"] + [f"merged n={n}" for n in ns] + [f"+FT n={n}" for n in ns], r))

    for f in sorted((out / "correlation").glob("*.json")) if (out / "correlation").exists() else []:
        d = json.load(open(f))
        parts.append(table(f"Task importance vs performance gap (Fig. 6, 10): {f.stem} (Pearson {d['pearson']:.3f}, Spearman {d['spearman']:.3f})",
                           ["task", "correlation importance", "accuracy gap"],
                           [[t, f"{w:.4f}", f"{g:.2f}"] for t, w, g in zip(d["tasks"], d["correlation_importance"], d["performance_gap"])]))
    return "\n".join(parts) + "\n"


def main():
    ap = argparse.ArgumentParser(description="Collect every result under outputs/ into markdown tables.")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--write", default="outputs/RESULTS.md")
    a = ap.parse_args()
    text = report(Path(a.out))
    Path(a.write).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
