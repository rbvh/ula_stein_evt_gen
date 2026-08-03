import matplotlib as mpl
import numpy as np
from matplotlib import ticker


COLORS = [
    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#ffff33",
    "#a65628",
    "#f781bf",
]
TEXTWIDTH = 7.2
FONTSIZE = 10.95


def configure_matplotlib(use_tex=True):
    mpl.rcParams["text.usetex"] = use_tex
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.size"] = FONTSIZE

    mpl.rcParams["xtick.direction"] = "in"
    mpl.rcParams["xtick.major.size"] = 4.0
    mpl.rcParams["xtick.minor.size"] = 2.0
    mpl.rcParams["xtick.major.width"] = 0.5
    mpl.rcParams["xtick.minor.width"] = 0.5

    mpl.rcParams["ytick.direction"] = "in"
    mpl.rcParams["ytick.major.size"] = 4.0
    mpl.rcParams["ytick.minor.size"] = 2.0
    mpl.rcParams["ytick.major.width"] = 0.5
    mpl.rcParams["ytick.minor.width"] = 0.5

    mpl.rcParams["lines.linewidth"] = 1.2
    mpl.rcParams["lines.markersize"] = 2.8
    mpl.rcParams["legend.labelspacing"] = 0.1
    mpl.rcParams["legend.frameon"] = False
    mpl.rcParams["legend.handletextpad"] = 0.5
    mpl.rcParams["legend.columnspacing"] = 0.9
    mpl.rcParams["xtick.minor.visible"] = True
    mpl.rcParams["ytick.minor.visible"] = True


def multiplicity_color(n_jets):
    return COLORS[n_jets % len(COLORS)]


def style_axes(ax, labelsize=None):
    ax.tick_params(direction="in", top=True, right=True, labelsize=labelsize)
    ax.tick_params(which="minor", direction="in", top=True, right=True)


def symlog_minor_ticks(ymin, ymax, linthresh):
    ticks = []
    max_abs = max(abs(ymin), abs(ymax))
    if max_abs > linthresh:
        max_power = int(np.ceil(np.log10(max_abs)))
        min_power = -int(np.ceil(-np.log10(linthresh)))
        for power in range(min_power, max_power + 1):
            decade = 10.0**power
            for sub in range(2, 10):
                tick = sub * decade
                ticks.extend([-tick, tick])
    linear_step = linthresh / 5.0
    ticks.extend(np.arange(-linthresh, linthresh + 0.5 * linear_step, linear_step))
    ticks = np.array(sorted(set(np.round(ticks, 14))))
    return ticks[(ticks > ymin) & (ticks < ymax)]


def style_symlog_axis(ax, linthresh, ymin=None, ymax=None):
    ax.set_yscale("symlog", linthresh=linthresh)
    ax.yaxis.set_major_locator(ticker.SymmetricalLogLocator(base=10, linthresh=linthresh))
    if ymin is not None or ymax is not None:
        current_ymin, current_ymax = ax.get_ylim()
        ax.set_ylim(
            current_ymin if ymin is None else ymin,
            current_ymax if ymax is None else ymax,
        )
    ax.yaxis.set_minor_locator(
        ticker.FixedLocator(symlog_minor_ticks(*ax.get_ylim(), linthresh))
    )
