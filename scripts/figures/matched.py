import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, NullLocator
from style import RED, BLUE, GRAY, LINE, apply, grid, frame

out_pdf, out_png = sys.argv[1:3]
matched = Path(sys.argv[3] if len(sys.argv) > 3 else "outputs/matched")


def series(setting):
    cells = defaultdict(list)
    for kind in ("merge", "finetune"):
        for f in (matched / setting / kind).glob("*.json"):
            r = json.load(open(f))
            cells[(r["method"], kind, r["n"])].append(r["test"]["avg/norm_acc"])
    mean = {k: float(np.mean(v)) for k, v in cells.items()}
    ns = sorted({k[2] for k in mean})
    baselines = sorted({k[0] for k in mean if k[0] not in ("com", "zeroshot")})
    return {"n": ns, "res": {"CoM merged": [mean[("com", "merge", n)] for n in ns], "CoM + FT": [mean[("com", "finetune", n)] for n in ns],
                             "Best baseline merged": [max(mean[(m, "merge", n)] for m in baselines) for n in ns],
                             "Best baseline + FT": [max(mean[(m, "finetune", n)] for m in baselines) for n in ns],
                             "Pretrained + FT": [mean[("zeroshot", "finetune", n)] for n in ns]}}


data = {"ViT-B/32": series("vit_b32_lora"), "Llama-3.2-1B": series("llama32_1b_lora")}
d = {"n": data["ViT-B/32"]["n"], "res": {k: v["res"] for k, v in data.items()}}
n = d["n"][1:]
res = {m: {k: v[1:] for k, v in r.items()} for m, r in d["res"].items()}

spec = [
    ("CoM + FT", "CoM merged", "CoM + fine-tuning", RED, "o", "-"),
    ("Best baseline + FT", "Best baseline merged", "Best baseline + fine-tuning", BLUE, "s", "--"),
    ("Pretrained + FT", None, "Pretrained + fine-tuning", GRAY, "^", ":"),
]
label_pos = {("ViT-B/32", RED): ("ft", (-3, 2), "bottom"), ("ViT-B/32", BLUE): ("merged", (-2, -2), "top"),
             ("Llama-3.2-1B", RED): ("ft", (-3, 4), "bottom"), ("Llama-3.2-1B", BLUE): ("merged", (-2, -2), "top")}
apply()
fig, axes = plt.subplots(1, 2, figsize=(3.08, 2.15), gridspec_kw={"wspace": 0.3})
for ax, model in zip(axes, ("ViT-B/32", "Llama-3.2-1B")):
    r = res[model]
    for ft, mg, label, color, mk, ls in spec:
        if mg is not None:
            ax.fill_between(n, r[mg], r[ft], color=color, alpha=0.15, linewidth=0, zorder=1)
        ax.plot(n, r[ft], label=label, color=color, linestyle=ls, marker=mk, zorder=3, **LINE)
        if mg is not None:
            where, off, va = label_pos[(model, color)]
            ax.annotate(f"{r[ft][-1] - r[mg][-1]:+.1f}", (n[-1], r[ft][-1] if where == "ft" else r[mg][-1]),
                        xytext=off, textcoords="offset points", va=va, ha="right", fontsize=6.3,
                        color=color, fontweight="bold")
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(n))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([str(v) for v in n])
    ax.set_xlim(38, 650)
    ax.set_title(model, fontsize=7, pad=2)
    grid(ax)
axes[0].set_ylim(70, 98)
axes[0].set_yticks([70, 80, 90])
axes[1].set_ylim(60, 102)
axes[1].set_yticks([60, 70, 80, 90])
axes[0].set_ylabel("Avg. normalized acc. (%)", labelpad=1.5)
fig.supxlabel("Labelled examples per task ($n$)", fontsize=7, y=0.175)
handles = [Line2D([], [], color=c, linestyle=ls, marker=mk, label=lb, **LINE) for _, _, lb, c, mk, ls in spec]
handles.append(Patch(facecolor="0.6", alpha=0.3, linewidth=0, label="Change vs. merged"))
frame(fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=5.6, handlelength=2.0,
                 columnspacing=0.9, labelspacing=0.3, handletextpad=0.5, bbox_to_anchor=(0.53, 0.0)))
fig.subplots_adjust(left=0.13, right=0.975, top=0.92, bottom=0.33)
fig.savefig(out_pdf)
fig.savefig(out_png, dpi=220)
