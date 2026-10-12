"""House chart style: one validated categorical palette, thin marks, recessive grid."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .data import FIG  # noqa: E402

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.figsize": (10, 5), "figure.dpi": 110, "savefig.dpi": 160,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 13, "axes.titleweight": "semibold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.prop_cycle": matplotlib.cycler(color=PALETTE),
    "lines.linewidth": 1.6, "xtick.color": INK2, "ytick.color": INK2,
    "legend.frameon": False, "font.size": 10,
})


def save(fig, name: str):
    fig.tight_layout()
    path = FIG / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    try:  # show inline when running inside Jupyter so notebooks render on GitHub
        get_ipython  # noqa: B018
        from IPython.display import Image, display
        display(Image(filename=str(path)))
    except NameError:
        pass
    return path
