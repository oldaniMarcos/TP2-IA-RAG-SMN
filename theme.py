import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(
    style="whitegrid",
    context="notebook",
    font="Trebuchet MS",
    font_scale=1.05,
    rc={
        "figure.figsize": (10, 6),
        "figure.dpi": 100,
        "savefig.dpi": 100,

        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#FFFFFF",

        "axes.titlesize": 17,
        "axes.titleweight": "bold",
        "axes.titlepad": 16,

        "axes.labelsize": 12,
        "axes.labelweight": "medium",

        "axes.grid": True,
        "grid.color": "#D9D9D9",
        "grid.linestyle": "--",
        "grid.linewidth": 0.7,
        "grid.alpha": 0.6,

        # "axes.spines.top": False,
        # "axes.spines.right": False,

        "xtick.labelsize": 10,
        "ytick.labelsize": 10,

        "legend.frameon": True,
        "legend.fontsize": 10,

        "lines.linewidth": 2,
        "lines.markersize": 6,
    }
)


# -----------------------------
# Palettes
# -----------------------------

PALETTE_MAIN = "deep"

PALETTE_CATEGORICAL = [
    "#4C72B0",
    "#55A868",
    "#C44E52",
    "#8172B2",
    "#CCB974",
    "#64B5CD",
]

PALETTE_TREATMENT = {
    "Yes": "#4C72B0",
    "No": "#C44E52",
}


sns.set_palette(PALETTE_CATEGORICAL)