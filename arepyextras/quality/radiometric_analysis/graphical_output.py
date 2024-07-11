# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Graphical Module to generate plots for Radiometric Analysis"""
from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

import arepyextras.quality.radiometric_analysis.custom_dataclasses as rdt

# syncing with logger
log = logging.getLogger("quality_analysis")


class PlotModes(Enum):
    """Overall profile computation mode"""

    MEAN = "mean"
    MIN = "min"


def radiometric_2D_hist_plot(
    data: rdt.RadiometricProfilesOutput,
    out_dir: str | Path,
    title: str | None = None,
    plot_mode: str | PlotModes = PlotModes.MEAN,
):
    """Radiometric profiles 2D histogram plot.

    Parameters
    ----------
    data : rdt.RadiometricProfilesOutput
        radiometric profiles output data
    out_dir : str | Path
        output directory
    title : str | None
        plot title
    plot_mode : str | PlotModes, optional
        overall profile extraction mode, by default PlotModes.MEAN
    """
    graphs_dir = Path(out_dir).joinpath("graphs")
    graphs_dir.mkdir(exist_ok=True)

    plot_mode = PlotModes(plot_mode)

    # figure plot
    if title is None:
        title = f"Radiometric Profile Histogram {data.swath} {data.polarization.name}"
    log.info(f"Generating {title}")

    fig, ax = plt.subplots(figsize=(8, 6), dpi=180)

    cs = ax.imshow(
        data.hist_2d,
        cmap="binary",
        vmin=1,
        extent=[
            data.hist_x_bins_axis.min(),
            data.hist_x_bins_axis.max(),
            data.hist_y_bins_axis.max(),
            data.hist_y_bins_axis.min(),
        ],
    )
    ax.invert_yaxis()
    if plot_mode == PlotModes.MEAN:
        mean_profile = np.nanmean(data.profiles, 0)
    elif plot_mode == PlotModes.MIN:
        np.ma.set_fill_value(data.profiles, np.nan)
        mean_profile = np.nanmin(data.profiles.data, 0)
    if data.look_angles is not None:
        mean_profile_axis = np.nanmean(data.look_angles, 0)
    else:
        mean_profile_axis = np.nanmean(data.block_azimuth_times, 0)
    smoothed_profile = savgol_filter(mean_profile, polyorder=3, window_length=mean_profile.size // 10)

    # forcing equal aspect ratio
    aspect = 8 / 6
    im = ax.get_images()
    extent = im[0].get_extent()
    ax.set_aspect(abs((extent[1] - extent[0]) / (extent[3] - extent[2])) / aspect)

    # ax.invert_yaxis()
    plt.plot(mean_profile_axis, smoothed_profile, color="#63B6E3")

    plt.locator_params(axis="x", nbins=10)
    plt.locator_params(axis="y", nbins=10)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    if data.look_angles is not None:
        plt.xlabel("Elevation Angle [deg]", fontdict={"size": 12})
    else:
        plt.xlabel("Azimuth Block Times [s]", fontdict={"size": 12})
    plt.ylabel("Power [dB]", fontdict={"size": 12})
    plt.title(title, fontdict={"size": 16, "weight": "bold"})
    plt.grid(color="#7EB4B4", alpha=0.4)

    fig.savefig(graphs_dir.joinpath(title.lower().replace(" ", "_")).with_suffix(".png"))
    plt.close("all")
