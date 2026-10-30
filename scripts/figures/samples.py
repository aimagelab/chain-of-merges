import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator
from style import GREEN, ORANGE, BLUE, LINE, apply, grid, frame

out_pdf, out_png = sys.argv[1], sys.argv[2]
sweep = Path(sys.argv[3] if len(sys.argv) > 3 else "outputs/sample_sweep")

samples = [5, 10, 50, 100, 200, 300, 400, 500]
names = {"ViT-B/32": "vit_b32_lora", "ViT-L/14": "vit_l14_lora", "Llama3-8B": "llama3_8b_lora"}
acc = {label: [np.mean([json.load(open(f))["avg/norm_acc"] for f in (sweep / setting / f"n{n}").rglob("metrics.json")]) for n in samples]
       for label, setting in names.items()}
style = {
    "ViT-B/32": dict(color=GREEN, marker="s", linestyle="--"),
    "ViT-L/14": dict(color=ORANGE, marker="o", linestyle="-"),
    "Llama3-8B": dict(color=BLUE, marker="^", linestyle="-"),
}

apply()
fig, (axv, axt) = plt.subplots(2, 1, figsize=(2.3, 2.15), sharex=True,
                               gridspec_kw={"height_ratios": [2.1, 1], "hspace": 0.12})
panels = [(axv, ["ViT-B/32", "ViT-L/14"], (75, 96), [75, 80, 85, 90, 95]),
          (axt, ["Llama3-8B"], (96.5, 100.5), [97, 99])]
for ax, names, ylim, yticks in panels:
    for name in names:
        ax.plot(samples, acc[name], label=name, **LINE, **style[name])
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([5, 10, 50, 100, 500]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(4, 650)
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    grid(ax)
frame(axv.legend(loc="lower right", fontsize=6, handlelength=2.0, labelspacing=0.3))
frame(axt.legend(loc="lower right", fontsize=6, handlelength=2.0, labels=["Llama3-8B (text)"]))
axt.set_xticks([5, 10, 50, 100, 500])
axt.set_xticklabels(["5", "10", "50", "100", "500"])
axt.set_xlabel("Examples per task", labelpad=1.5)
fig.supylabel("Avg. normalized acc. (%)", fontsize=7, x=0.03)

fig.subplots_adjust(left=0.22, right=0.97, top=0.97, bottom=0.17)
fig.savefig(out_pdf)
fig.savefig(out_png, dpi=220)
