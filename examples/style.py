import matplotlib
import matplotlib.pyplot as plt
from cycler import cycler

_my_style = {
    "figure.figsize": (3.5, 3),
    "axes.spines.right": False,
    "axes.spines.top": False,
    "legend.frameon": False,
    "legend.fancybox": False,
    "savefig.transparent": False,
    "axes.prop_cycle": cycler(
        "color",
        [
            "k",  # black
            "#C5091F",  # red
            "#1F77B4",  # blue
        ],
    ),
}
plt.rcParams.update(_my_style)


def move_axes(
    ax: matplotlib.axes.Axes | None = None,
    which: str = "both",
):
    """
    Move the axes outward.

    Parameters
    ----------
    ax
        The matplotlib axes object to modify. If None, the current axes are
        used.

    which
        The axes to move. Options are "both", "x", "y", "bottom", "top", "left",
        and "right
    """

    ax: plt.Axes = ax or plt.gca()
    if which in ["both", "x", "bottom"]:
        ax.spines["bottom"].set_position(("outward", 10))

    if which in ["both", "y", "left"]:
        ax.spines["left"].set_position(("outward", 10))
