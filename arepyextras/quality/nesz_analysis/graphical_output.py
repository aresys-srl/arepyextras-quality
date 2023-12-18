# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
Noise Equivalent Sigma Zero (NESZ) graphical output
---------------------------------------------------
"""

from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from arepyextras.quality.core.signal_processing import convert_to_db
from arepyextras.quality.nesz_analysis.custom_dataclasses import NESZOutput


def nesz_graphs(data: NESZOutput, output_dir: Union[str, Path]) -> None:
    """Creating graphical output for NESZ analysis.

    Parameters
    ----------
    data : NESZOutput
        NESZ computation output
    output_dir : Union[str, Path]
        path to folder where to save graphs
    """
    output_dir = Path(output_dir).joinpath("graphs")
    output_dir.mkdir(exist_ok=True)

    # computing averages
    profile_db = convert_to_db(data.nesz_profiles)

    center_intensity = convert_to_db(np.nanmean(data.nesz_profiles))
    intensity_bins = np.linspace(center_intensity - 20, center_intensity + 20, 301)
    elevation_angles_bins = data.axis_deg[::5]

    # 2D histogram generation
    hist, _, _ = np.histogram2d(
        x=data.elevation_angles_deg.ravel(),
        y=profile_db.ravel(),
        bins=[elevation_angles_bins, intensity_bins],
    )
    hist = hist.T

    # figure plot
    title = f"NESZ Profile Histogram {data.swath} {data.polarization.name}"
    fig = plt.figure(figsize=(8, 6), dpi=180)
    sns.heatmap(
        data=hist,
        xticklabels=np.round(elevation_angles_bins, 1),
        yticklabels=np.round(intensity_bins, 1),
        cmap="YlOrBr",
        cbar=False,
    )
    plt.gca().invert_yaxis()
    plt.locator_params(axis="x", nbins=10)
    plt.locator_params(axis="y", nbins=10)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.xlabel("Elevation Angle [deg]", fontdict={"size": 12})
    plt.ylabel("Power [dB]", fontdict={"size": 12})
    plt.title(title, fontdict={"size": 16, "weight": "bold"})

    fig.savefig(output_dir.joinpath(title.lower().replace(" ", "_")).with_suffix(".png"))
    plt.close("all")
