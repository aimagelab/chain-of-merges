import matplotlib

GREEN, BLUE, ORANGE, RED, GRAY = "#2ca02c", "#1f77b4", "#ff7f0e", "#d62728", "#7f7f7f"
RC = {"pdf.fonttype": 42, "font.size": 7, "axes.linewidth": 0.6,
      "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5}
LINE = dict(linewidth=1.3, alpha=0.9, markersize=4.2, markeredgecolor="white", markeredgewidth=0.5)


def apply():
    matplotlib.rcParams.update(RC)


def grid(ax, **kw):
    ax.grid(alpha=0.3, linewidth=0.6, **kw)
    ax.set_axisbelow(True)


def frame(leg):
    f = leg.get_frame()
    f.set_facecolor("lightgray")
    f.set_alpha(0.5)
    f.set_linewidth(0.6)
    return leg
