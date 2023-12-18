# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Graphical Module to generate plots for Radiometric Analysis"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import arepyextras.quality.core.generic_dataclasses as gdt
import arepyextras.quality.radiometric_analysis.custom_dataclasses as rdt


def radiometric_profiles(
    data: list[rdt.RadiometricAnalysisOutput],
    out_dir: Path,
    config: rdt.RadiometricAnalysisConfig,
    projection: gdt.SARProjection,
) -> None:
    """Generation of the Radiometric Analysis graphical output.

    Parameters
    ----------
    data : list[rdt.RadiometricAnalysisOutput]
        list of RadiometricAnalysisOutput results dataclass
    out_dir : Path
        output directory where to save the graphs
    config : rdt.RadiometricAnalysisConfig
        configuration RadiometricAnalysisConfig dataclass
    projection : gdt.SARProjection
        data projection
    """

    color_set = ["#7E5920", "#FFA737", "#5F5449", "#5999D9"]

    # re-organizing input data
    times = list({p.time for p in data})

    for time_id, time in enumerate(times):
        selected_data = [p for p in data if p.time == time]
        direction = list({p.direction for p in selected_data})[0]
        out_type = list({p.value_type for p in selected_data})[0]
        smoothed_profiles = np.ma.masked_invalid(np.concatenate([p.smoothed_profile for p in selected_data]))
        original_profiles = np.ma.masked_invalid(np.concatenate([p.profile for p in selected_data]))
        axes = np.concatenate([p.axis for p in selected_data])

        # initializing figure
        fig, axs = plt.subplots(figsize=(14, 10))
        fig.suptitle(f"Radiometric Profiles @ {time}", fontsize=16, fontweight="bold")
        axs.set_title(f"{direction.name.capitalize()} direction", fontsize=12)

        for num, channel in enumerate(selected_data):
            if direction == rdt.RadiometricAnalysisDirection.RANGE:
                x_label = "Slant Range time [s]"
                if projection == gdt.SARProjection.GROUND_RANGE:
                    x_label = "Ground Range distance [m]"

                if config.axis == rdt.RadiometricAnalysisAxes.INCIDENCE_ANGLE:
                    x_label = "Incidence Angle [deg]"
                elif config.axis == rdt.RadiometricAnalysisAxes.LOOK_ANGLE:
                    x_label = "Look Angle [deg]"
            else:
                x_label = "Azimuth time (relative) [s]"

            if out_type == rdt.RadiometricAnalysisValue.AMPLITUDE:
                nought = " ".join([s.capitalize() for s in config.output_type.name.split("_")])
                y_label = f"{config.value.name.capitalize()} {nought} [dB]"
            else:
                y_label = "Phase [rad]"

            _plotting_profiles(axis=axs, channel=channel, color=color_set[num], x_label=x_label, y_label=y_label)

        # customizing data for histogram 2D plot
        profile_mean = smoothed_profiles.mean()
        x_lim = [np.min(axes), np.max(axes)]
        y_lim = [smoothed_profiles.min(), smoothed_profiles.max()]
        y_lim[0] = y_lim[0] * 1.1 if y_lim[0] < 0 else y_lim[0] * 0.9
        y_lim[1] = y_lim[1] * 0.9 if y_lim[1] < 0 else y_lim[1] * 1.1

        # x_edges = np.linspace(x_lim[0], x_lim[1], 101)
        # y_edges = np.linspace(profile_mean-6, profile_mean+6, 51)
        y_values = original_profiles[~original_profiles.mask].data.copy()
        x_values = axes[~original_profiles.mask].copy()
        display_cond = np.logical_and(y_values > profile_mean - 6, y_values < profile_mean + 6)

        # plotting histogram
        axs.hist2d(x_values[display_cond], y_values[display_cond], (101, 51), cmin=1, cmap="bone", alpha=0.7, label="_")

        # setting axes limits
        axs.set_xlim(x_lim)
        axs.set_ylim(y_lim)

        axs.grid(visible=True, linestyle="--", alpha=0.6)

        plt.legend()

        brst_info = ("_burst" + str(selected_data[0].burst)) if selected_data[0].burst is not None else ""
        plot_name = (
            selected_data[0].swath + brst_info + "_radiometric_analysis_" + direction.name.lower() + "_" + str(time_id)
        )

        plt.tight_layout()
        plt.savefig(out_dir.joinpath(plot_name).with_suffix(".png"), dpi=200)
        plt.close(fig)


def _plotting_profiles(axis: plt.Axes, channel: rdt.RadiometricAnalysisOutput, color: str, x_label: str, y_label: str):
    """Function to plot radiometric analysis output profiles.

    Parameters
    ----------
    axis : plt.Axes
        axes where to plot the profiles
    channel : rdt.RadiometricAnalysisOutput
        radiometric analysis output dataclass
    color : str
        color of the plot line
    x_label : str
        x axis label
    y_label : str
        y axis label
    """

    brst_label = (" Burst " + str(channel.burst)) if channel.burst is not None else ""
    label = channel.swath + brst_label + " " + channel.polarization

    axis.plot(channel.axis, channel.smoothed_profile, color=color, label=label)

    axis.set_xlabel(x_label, fontsize=12, fontweight="bold")
    axis.set_ylabel(y_label, fontsize=12, fontweight="bold")
