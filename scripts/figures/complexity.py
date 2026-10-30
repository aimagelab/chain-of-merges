import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from style import apply

BLUE, RED = "#435183", "#B03A2E"

out_dir = sys.argv[1]

num_samples_train = 838429
max_num_samples_com = 500
min_num_samples_com = 5
R = 16
settings = {
    "vitb": dict(forward=4.3e9, M=48, N=8, D=768, seq_len=50),
    "vitl": dict(forward=6.2e10, M=96, N=8, D=1024, seq_len=257),
    "llama": dict(forward=1.23e12, M=128, N=8, D=4096, seq_len=77),
}
labels = ["TA", "TIES", "DARE_ties", "Cons_TA", "LiNeS", "FisherAVG", "RegMean", "MaTS", "TSV", "Iso-C",
          "KnOTS_ties", "CORE_tsv", "CORE_iso", "CoM_fast", "CoM_best"]
titles = {"vitl": "ViT-L/14", "llama": "Llama3-8B"}


def costs(forward, M, N, D, seq_len):
    backward = 2 * forward
    core = 2 * M * D**2 + D**2 * N * R + N**4 * R**3
    flops = [
        2 * M * D**2,
        3 * M * D**2,
        3 * M * D**2 * 2,
        3.1 * M * D**2,
        2.3 * M * N * D**2 + 64.3 * M * D**2,
        num_samples_train * (forward + backward) + 2 * M * N * D**2,
        2 * N * M * num_samples_train * D**2 + M * ((2 * N + 8 / 3) * D**3 + N * D**2) + forward * num_samples_train,
        (num_samples_train + 40) * (forward + backward) + 2 * M * N * D**2,
        1.125 * M * N * D**3 + 2.125 * M * N * D**2,
        M * D**3 + 2 * M * N * D**2,
        M * N * D**3 + 2 * M * N * D**2 + 3 * M * N * D**2,
        core,
        core,
        2 * N * M * min_num_samples_com * D**2 + M * ((2 * N + 8 / 3) * D**3 / 3 + N * D**2) + forward * min_num_samples_com,
        2 * N * M * max_num_samples_com * D**2 + M * ((2 * N + 8 / 3) * D**3 / 3 + N * D**2) + forward * max_num_samples_com,
    ]
    mem = [2 * M * D**2] * 5 + [3 * M * D**2, 2 * M * D**2, 3 * M * D**2, 2 * M * D**2, 2 * M * D**2,
                                15 * M * N * D**2, 2 * M * D**2, 2 * M * D**2,
                                2 * M * D**2 + min_num_samples_com * D * seq_len,
                                2 * M * D**2 + max_num_samples_com * D * seq_len]
    return np.log10(np.array(mem, dtype=np.float64)), np.log10(np.array(flops, dtype=np.float64))


def plot(mem, flops, size, out, fontsize, gap, margins):
    apply()
    plt.rcParams.update({"font.size": fontsize})
    colors = [BLUE] * (len(labels) - 2) + [RED] * 2
    y = np.arange(len(labels))[::-1]
    fig, (axm, axl, axf) = plt.subplots(1, 3, figsize=size, sharey=True,
                                        gridspec_kw={"width_ratios": [1, gap, 1], "wspace": 0.0})
    fig.subplots_adjust(**margins)
    m_lo, f_lo = np.floor(mem.min() - 0.5), np.floor(flops.min() - 2)
    for ax, v, lo, hi in ((axm, mem, m_lo, mem.max() + 0.25), (axf, flops, f_lo, flops.max() + 0.6)):
        ax.barh(y, v - lo, left=lo, height=0.72, color=colors, alpha=0.9, edgecolor="white", linewidth=0.5, zorder=3)
        ax.set_xlim(lo, hi)
        ax.grid(axis="x", alpha=0.3, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0, labelleft=False)
    axm.set_ylim(-0.6, len(labels) - 0.4)
    axm.invert_xaxis()
    axl.axis("off")
    for yi, lb, c in zip(y, labels, colors):
        axl.text(0.5, yi, lb, ha="center", va="center", fontsize=fontsize)
    axm.set_title("Peak Memory", fontsize=fontsize + 0.5, pad=3)
    axf.set_title("Computation", fontsize=fontsize + 0.5, pad=3)
    axm.set_xlabel("log10(GB)", labelpad=1.5)
    axf.set_xlabel("log10(FLOPs)", labelpad=1.5)
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"), dpi=220)
    plt.close(fig)


def plot_vertical(mem, flops, model, out):
    apply()
    plt.rcParams.update({"font.size": 12})
    colors = [BLUE] * (len(labels) - 2) + [RED] * 2
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 1, figsize=(9, 6))
    panels = ((axes[0], mem, "log10(GB)", f"Peak Memory Usage ({model})", (mem.min() * 0.902, mem.max() * 1.02)),
              (axes[1], flops, "log10(FLOPs)", f"FLOPs Cost ({model})", (flops.min() * 0.64, flops.max() * 1.05)))
    for ax, v, ylabel, title, ylim in panels:
        ax.bar(x, v, width=0.7, color=colors, alpha=0.9, edgecolor="white", linewidth=0.8, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_title(title, fontsize=13, weight="bold")
        ax.set_ylim(*ylim)
        ax.grid(axis="y", alpha=0.3, linewidth=0.8)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


for name, cfg in settings.items():
    mem, flops = costs(**cfg)
    if name == "vitb":
        plot(mem, flops, (2.45, 2.0), f"{out_dir}/complexity_bars_formal_horizontal.pdf", 5.8, 0.85,
             dict(left=0.03, right=0.97, top=0.93, bottom=0.135))
    else:
        plot_vertical(mem, flops, titles[name], f"{out_dir}/complexity_bars_{name}.pdf")
