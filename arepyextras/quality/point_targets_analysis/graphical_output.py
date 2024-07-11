# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Graphical Module to generate plots for IRF and RCS analyses"""
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib.font_manager import FontProperties
from plotly.subplots import make_subplots

import arepyextras.quality.core.generic_dataclasses as gdt
import arepyextras.quality.point_targets_analysis.custom_dataclasses as ptdt
from arepyextras.quality.core.signal_processing import convert_to_db


def irf_graphs(data_graph: ptdt.IRFGraphDataOutput, data_values: dict, label: str, out_dir: Path) -> None:
    """Function to generate the graphical output after IRF analysis.

    Parameters
    ----------
    data_graph : ptdt.IRFGraphDataOutput
        dataclass instance containing all relevant data for plotting results
    data_values : dict
        dictionary of IRF results
    label : str
        label of point target in exam
    out_dir : Path
        output folder path
    """

    # figure init
    fig = plt.figure(figsize=(9, 9))
    gs = fig.add_gridspec(3, 2)
    ax1 = fig.add_subplot(gs[:2, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[2, 0])
    ax5 = fig.add_subplot(gs[2, 1])

    lobe_rng = data_graph.side_lobes_directions[0]
    lobe_az = data_graph.side_lobes_directions[1]

    rng_ax_m = data_graph.rng_axis * data_graph.rng_step_distance
    az_ax_m = data_graph.az_axis * data_graph.az_step_distance
    image_db = convert_to_db(np.abs(data_graph.image), mode=gdt.DecibelConversion.AMPLITUDE)

    # 1st plot: interpolated irf target area
    axes_ratio = az_ax_m.max() / rng_ax_m.max()
    extent = [az_ax_m[0], az_ax_m[-1], rng_ax_m[-1], rng_ax_m[0]]
    ax1.imshow(image_db, vmin=image_db.max() - 40, cmap="jet", extent=extent, aspect=axes_ratio)
    ax1.plot(data_values["azimuth_localization_error_[m]"], data_values["slant_range_localization_error_[m]"], "ro")
    ax1.plot(az_ax_m, lobe_az * rng_ax_m)
    if np.isinf(lobe_rng):
        ax1.vlines(0, rng_ax_m[0], rng_ax_m[-1])
    else:
        ax1.plot(az_ax_m, lobe_rng * rng_ax_m)

    # customization
    ax1.grid(alpha=0.3, linestyle="--")
    ax1.set_title("Interpolated Response", fontweight="bold")
    ax1.set_xlabel(
        "Azimuth [m]",
        fontweight="bold",
    )
    ax1.set_ylabel("Range [m]", fontweight="bold")

    # 2nd plot: summary table
    ax2.axis("off")
    # ax2.axis('tight')
    tbl_data = np.round(
        np.array(
            [
                [data_values["range_resolution_[m]"], data_values["azimuth_resolution_[m]"]],
                [data_values["range_pslr_[dB]"], data_values["azimuth_pslr_[dB]"]],
                [data_values["range_islr_[dB]"], data_values["azimuth_islr_[dB]"]],
            ]
        ),
        5,
    )
    tbl = ax2.table(
        cellText=tbl_data,
        colLabels=["Range", "Azimuth"],
        rowLabels=["Resolution [m]", "PSLR [dB]", "ISLR [dB]"],
        bbox=[0.45, 0.1, 0.6, 0.45],
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.auto_set_column_width(col=[0, 1])

    for (row, col), cell in tbl.get_celld().items():
        if (row == 0) or (col == -1):
            cell.set_text_props(fontproperties=FontProperties(weight="bold"))

    # 3rd plot: localization table
    ax3.axis("off")
    tbl_data1 = np.round(
        np.array(
            [
                [data_values["slant_range_localization_error_[m]"]],
                [data_values["ground_range_localization_error_[m]"]],
                [data_values["azimuth_localization_error_[m]"]],
            ]
        ),
        5,
    )
    tbl1 = ax3.table(
        cellText=tbl_data1,
        colLabels=["Localization Error"],
        rowLabels=["Slant Range [m]", "Ground Range [m]", "Azimuth [m]"],
        bbox=[0.55, 0.4, 0.55, 0.45],
        cellLoc="center",
    )
    tbl1.auto_set_font_size(False)
    tbl1.auto_set_column_width(col=[0])

    for (row, col), cell in tbl1.get_celld().items():
        if (row == 0) or (col == -1):
            cell.set_text_props(fontproperties=FontProperties(weight="bold"))

    # 4th plot: range profile
    prof = convert_to_db(np.abs(data_graph.rng_profile), mode=gdt.DecibelConversion.AMPLITUDE)

    if np.abs(lobe_rng * data_graph.image.shape[1] / data_graph.image.shape[0]) > 1:
        ax4.plot(rng_ax_m, prof)
        ax4.hlines(prof.max() - 3, rng_ax_m[0], rng_ax_m[-1], linestyle="--", color="r")
        x_lim_low = np.max([rng_ax_m[0], -5.5 * data_graph.rng_resolution * data_graph.rng_step_distance])
        x_lim_up = np.min([rng_ax_m[-1], 5.5 * data_graph.rng_resolution * data_graph.rng_step_distance])
    else:
        # with sidelobe dirs
        ax4.plot(data_graph.az_axis * data_graph.rng_step_distance, prof)
        ax4.hlines(
            prof.max() - 3,
            data_graph.az_axis[0] * data_graph.rng_step_distance,
            data_graph.az_axis[-1] * data_graph.rng_step_distance,
            linestyle="--",
            color="r",
        )
        x_lim_low = np.max(
            [
                data_graph.az_axis[0] * data_graph.rng_step_distance,
                -5.5 * data_graph.rng_resolution * data_graph.rng_step_distance,
            ]
        )
        x_lim_up = np.min(
            [
                data_graph.az_axis[-1] * data_graph.rng_step_distance,
                5.5 * data_graph.rng_resolution * data_graph.rng_step_distance,
            ]
        )

    ax4.set_xlim([x_lim_low, x_lim_up])
    ax4.set_ylim([-40, 0.5])
    ax4.grid(alpha=0.4)

    # labelling customization
    ax4.set_xlabel("Range (along cut) [m]", fontweight="bold")
    ax4.set_ylabel("Power [dB]", fontweight="bold")

    # 5th plot: azimuth profile
    prof = convert_to_db(np.abs(data_graph.az_profile), mode=gdt.DecibelConversion.AMPLITUDE)

    if np.abs(lobe_az * data_graph.image.shape[1] / data_graph.image.shape[0]) > 1:
        # with sidelobe dirs
        ax5.plot(data_graph.rng_axis * data_graph.az_step_distance, prof)
        ax5.hlines(
            prof.max() - 3,
            data_graph.rng_axis[0] * data_graph.az_step_distance,
            data_graph.rng_axis[-1] * data_graph.az_step_distance,
            linestyle="--",
            color="r",
        )
        x_lim_low = np.max(
            [
                data_graph.az_axis[0] * data_graph.az_step_distance,
                -5.5 * data_graph.az_resolution * data_graph.az_step_distance,
            ]
        )
        x_lim_up = np.min(
            [
                data_graph.az_axis[-1] * data_graph.az_step_distance,
                5.5 * data_graph.az_resolution * data_graph.az_step_distance,
            ]
        )
    else:
        ax5.plot(az_ax_m, prof)
        ax5.hlines(prof.max() - 3, az_ax_m[0], az_ax_m[-1], linestyle="--", color="r")
        x_lim_low = np.max([az_ax_m[0], -5.5 * data_graph.az_resolution * data_graph.az_step_distance])
        x_lim_up = np.min([az_ax_m[-1], 5.5 * data_graph.az_resolution * data_graph.az_step_distance])

    ax5.set_xlim([x_lim_low, x_lim_up])
    ax5.set_ylim([-40, 0.5])
    ax5.grid(alpha=0.4)

    # labelling customization
    ax5.set_xlabel("Azimuth (along cut) [m]", fontweight="bold")
    ax5.set_ylabel("Power [dB]", fontweight="bold")

    title = label + " IRF Analysis"
    fig.suptitle(title, fontsize=16, fontweight="bold")

    gs.update(wspace=0.3, hspace=0, top=0.97)

    fig.savefig(out_dir.joinpath(title).with_suffix(".png"), dpi=200)
    plt.close("all")


def rcs_graphs(data_graph: ptdt.RCSGraphDataOutput, label: str, out_dir: Path) -> None:
    """Function to generate the graphical output after RCS analysis.

    Parameters
    ----------
    data_graph : ptdt.RCSGraphDataOutput
        dataclass instance containing all relevant data for plotting results
    label : str
        label of point target in exam
    out_dir : Path
        output folder path
    """

    # figure init
    fig, ax_1 = plt.subplots(figsize=(6, 6))
    rng_axis = np.arange(-data_graph.roi_size[0] / 2, data_graph.roi_size[0] / 2) * data_graph.rng_step_distance
    az_axis = np.arange(-data_graph.roi_size[1] / 2, data_graph.roi_size[1] / 2) * data_graph.az_step_distance

    if data_graph.data_type == gdt.TargetDataType.DETECTED:
        im_db = convert_to_db(np.abs(data_graph.image))
    else:
        im_db = convert_to_db(np.abs(data_graph.image), mode=gdt.DecibelConversion.AMPLITUDE)

    axes_ratio = az_axis.max() / rng_axis.max()
    extent = [az_axis[0], az_axis[-1], rng_axis[-1], rng_axis[0]]

    # plotting image
    ax_1.imshow(im_db, vmin=im_db.max() - 40, cmap="jet", aspect=axes_ratio, extent=extent)

    # plotting peak roi rectangle
    roi_peak = np.asarray(data_graph.roi_peak)
    roi_rng = (roi_peak[:2] / data_graph.interp_factor - data_graph.roi_size[0] / 2) * data_graph.rng_step_distance
    roi_az = (roi_peak[2:] / data_graph.interp_factor - data_graph.roi_size[1] / 2) * data_graph.az_step_distance
    rect_peak = patches.Rectangle(
        (roi_az[0], roi_rng[0]),
        roi_az[1] - roi_az[0],
        roi_rng[1] - roi_rng[0],
        linewidth=3,
        edgecolor="r",
        facecolor="none",
    )
    ax_1.add_patch(rect_peak)

    # plotting background corner rectangles
    rect_corners = []
    for rect in data_graph.roi_background:
        rect = np.asarray(rect).astype(float)
        rect[:2] = (rect[:2] - data_graph.roi_size[0] / 2) * data_graph.rng_step_distance
        rect[2:] = (rect[2:] - data_graph.roi_size[1] / 2) * data_graph.az_step_distance

        rect_corners.append(
            patches.Rectangle(
                (rect[2], rect[0]), rect[3] - rect[2], rect[1] - rect[0], linewidth=2, edgecolor="m", facecolor="none"
            )
        )

    for rect in rect_corners:
        ax_1.add_patch(rect)

    # customizing labels
    ax_1.set_xlabel("Azimuth [m]", fontsize=13)
    ax_1.set_ylabel("Range [m]", fontsize=13)

    # adding title and subtitle
    title = label + " RCS Analysis"
    plt.title("$\sigma = $" + f"{np.round(data_graph.rcs_lin, 4)} = " + f"{np.round(data_graph.rcs_db, 4)} [dB]")
    plt.suptitle(title, fontsize=16, fontweight="bold", y=0.97)

    # saving fig and closing it
    fig.savefig(out_dir.joinpath(title).with_suffix(".png"), dpi=200)
    plt.close("all")


def interactive_graphs(
    irf_data_graph: ptdt.IRFGraphDataOutput,
    rcs_data_graph: ptdt.RCSGraphDataOutput,
    data_values: dict,
    label: str,
    out_dir: Path,
) -> None:
    """Interactive graph generation using plotly for Point Target Analysis (IRF + RCS)

    Parameters
    ----------
    irf_data_graph : ptdt.IRFGraphDataOutput
        dataclass instance containing all relevant data for plotting results of IRF
    rcs_data_graph : ptdt.RCSGraphDataOutput
        dataclass instance containing all relevant data for plotting results or RCS
    data_values : dict
        dictionary of full Point Target Analysis results
    label : str
        lable of current Point Target id
    out_dir : Path
        output folder for saving results
    """

    # subplots generation
    fig = make_subplots(
        rows=2,
        cols=4,
        specs=[[{}, {}, {}, {}], [{"type": "table", "t": 0.018, "colspan": 4}, None, None, None]],
        subplot_titles=(
            "<b>IRF Interpolated Response</b>",
            "<b>Range Cut (interpolated)</b>",
            "<b>Azimuth Cut (interpolated)</b>",
            "<b>Target RCS</b>",
            "<b>Summary Results</b>",
        ),
        row_heights=[0.6, 0.35],
        vertical_spacing=0.3,
        horizontal_spacing=0.08,
    )
    fig.update_annotations(yshift=15)

    lobe_rng = irf_data_graph.side_lobes_directions[0]
    lobe_az = irf_data_graph.side_lobes_directions[1]

    rng_ax_m = irf_data_graph.rng_axis * irf_data_graph.rng_step_distance
    az_ax_m = irf_data_graph.az_axis * irf_data_graph.az_step_distance
    image_db = convert_to_db(np.abs(irf_data_graph.image), mode=gdt.DecibelConversion.AMPLITUDE)
    extent = [az_ax_m[0], az_ax_m[-1], rng_ax_m[-1], rng_ax_m[0]]

    rng_prof = convert_to_db(np.abs(irf_data_graph.rng_profile), mode=gdt.DecibelConversion.AMPLITUDE).squeeze()
    az_prof = convert_to_db(np.abs(irf_data_graph.az_profile), mode=gdt.DecibelConversion.AMPLITUDE).squeeze()

    # first subplot: IRF
    # interpolated image
    fig.append_trace(
        go.Heatmap(
            x=np.linspace(extent[0], extent[1], image_db.shape[1]),
            y=np.linspace(extent[2], extent[3], image_db.shape[0]),
            z=image_db,
            zmin=image_db.max() - 40,
            zmax=image_db.max(),
            colorscale="Jet",
            showscale=False,
            name="Interpolated Target Response",
            hovertemplate="<br>".join(
                ["<b>Azimuth [m]</b>: %{x}", "<b>Range [m]</b>: %{y}", "<b>Power [dB]</b>: %{z}"]
            ),
        ),
        row=1,
        col=1,
    )
    # point target nominal position
    fig.append_trace(
        go.Scatter(
            x=[-data_values["azimuth_localization_error_[m]"]],
            y=[-data_values["slant_range_localization_error_[m]"]],
            mode="markers",
            marker={"symbol": "circle", "color": "#EE4B2B", "size": 10},
            showlegend=False,
            name="Target Nominal Position",
            hovertemplate="<br>".join(["<b>Azimuth [m]</b>: %{x}", "<b>Range [m]</b>: %{y}"]),
        ),
        row=1,
        col=1,
    )
    # range cut
    fig.append_trace(
        go.Scatter(
            x=az_ax_m,
            y=-lobe_az * rng_ax_m,
            mode="lines",
            line_color="white",
            opacity=0.5,
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )
    # azimuth cut
    if np.isinf(lobe_rng):
        fig.add_vline(x=0, line_color="white", opacity=0.5, row=1, col=1)
    else:
        fig.append_trace(
            go.Scatter(
                x=az_ax_m,
                y=-lobe_rng * rng_ax_m,
                mode="lines",
                line_color="white",
                opacity=0.5,
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )

    # customizing IRF plot
    fig.update_yaxes(
        title_text="<b>Range [m]</b>",
        # autorange="reversed",
        scaleratio=1,
        # dtick=2,
        row=1,
        col=1,
    )
    fig.update_xaxes(
        title_text="<b>Azimuth [m]</b>",
        # dtick=1,
        row=1,
        col=1,
    )

    # second subplot: IRF
    # range cut profile
    if np.abs(lobe_rng * irf_data_graph.image.shape[1] / irf_data_graph.image.shape[0]) > 1:
        fig.append_trace(
            go.Scatter(x=rng_ax_m, y=rng_prof, mode="lines", line_color="#003d5b", showlegend=False), row=1, col=2
        )
        x_lim_low = np.max([rng_ax_m[0], -5.5 * irf_data_graph.rng_resolution * irf_data_graph.rng_step_distance])
        x_lim_up = np.min([rng_ax_m[-1], 5.5 * irf_data_graph.rng_resolution * irf_data_graph.rng_step_distance])
    else:
        # with sidelobe dirs
        fig.append_trace(
            go.Scatter(
                x=irf_data_graph.az_axis * irf_data_graph.rng_step_distance,
                y=rng_prof,
                mode="lines",
                line_color="#003d5b",
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        x_lim_low = np.max(
            [
                irf_data_graph.az_axis[0] * irf_data_graph.rng_step_distance,
                -5.5 * irf_data_graph.rng_resolution * irf_data_graph.rng_step_distance,
            ]
        )
        x_lim_up = np.min(
            [
                irf_data_graph.az_axis[-1] * irf_data_graph.rng_step_distance,
                5.5 * irf_data_graph.rng_resolution * irf_data_graph.rng_step_distance,
            ]
        )

    # adding -3 dB line
    fig.add_hline(
        y=rng_prof.max() - 3,
        line_color="red",
        line_dash="dash",
        annotation_text="<b>-3 dB</b>",
        annotation_position="bottom right",
        row=1,
        col=2,
    )

    # labelling customization
    fig.update_yaxes(
        range=[-40, 0.5],
        title_text="<b>Power [dB]</b>",
        showgrid=True,
        zeroline=True,
        zerolinecolor="LightPink",
        gridcolor="LightPink",
        row=1,
        col=2,
    )
    fig.update_xaxes(
        range=[x_lim_low, x_lim_up],
        title_text="<b>Range (along cut) [m]</b>",
        showgrid=True,
        zeroline=True,
        zerolinecolor="LightPink",
        gridcolor="LightPink",
        row=1,
        col=2,
    )

    # third subplot: IRF
    # azimuth cut profile
    if np.abs(lobe_az * irf_data_graph.image.shape[1] / irf_data_graph.image.shape[0]) > 1:
        # with sidelobe dirs
        fig.append_trace(
            go.Scatter(
                x=irf_data_graph.rng_axis * irf_data_graph.az_step_distance,
                y=az_prof,
                mode="lines",
                line_color="#a4303f",
                showlegend=False,
            ),
            row=1,
            col=3,
        )
        x_lim_low = np.max(
            [
                irf_data_graph.az_axis[0] * irf_data_graph.az_step_distance,
                -5.5 * irf_data_graph.az_resolution * irf_data_graph.az_step_distance,
            ]
        )
        x_lim_up = np.min(
            [
                irf_data_graph.az_axis[-1] * irf_data_graph.az_step_distance,
                5.5 * irf_data_graph.az_resolution * irf_data_graph.az_step_distance,
            ]
        )
    else:
        fig.append_trace(
            go.Scatter(x=az_ax_m, y=az_prof, mode="lines", line_color="#a4303f", showlegend=False), row=1, col=3
        )
        x_lim_low = np.max([az_ax_m[0], -5.5 * irf_data_graph.az_resolution * irf_data_graph.az_step_distance])
        x_lim_up = np.min([az_ax_m[-1], 5.5 * irf_data_graph.az_resolution * irf_data_graph.az_step_distance])

    # adding -3 dB line
    fig.add_hline(
        y=az_prof.max() - 3,
        line_color="red",
        line_dash="dash",
        annotation_text="<b>-3 dB</b>",
        annotation_position="bottom right",
        row=1,
        col=3,
    )

    # labelling customization
    fig.update_yaxes(
        range=[-40, 0.5],
        title_text="<b>Power [dB]</b>",
        showgrid=True,
        zeroline=True,
        zerolinecolor="LightPink",
        gridcolor="LightPink",
        row=1,
        col=3,
    )
    fig.update_xaxes(
        range=[x_lim_low, x_lim_up],
        title_text="<b>Azimuth (along cut) [m]</b>",
        showgrid=True,
        zeroline=True,
        zerolinecolor="LightPink",
        gridcolor="LightPink",
        row=1,
        col=3,
    )

    # fourth subplot: RCS
    # rcs image
    rng_axis = (
        np.arange(-rcs_data_graph.roi_size[0] / 2, rcs_data_graph.roi_size[0] / 2) * rcs_data_graph.rng_step_distance
    )
    az_axis = (
        np.arange(-rcs_data_graph.roi_size[1] / 2, rcs_data_graph.roi_size[1] / 2) * rcs_data_graph.az_step_distance
    )

    if rcs_data_graph.data_type == gdt.TargetDataType.DETECTED:
        image_db = convert_to_db(np.abs(rcs_data_graph.image))
    else:
        image_db = convert_to_db(np.abs(rcs_data_graph.image), mode=gdt.DecibelConversion.AMPLITUDE)

    extent = [az_axis[0], az_axis[-1], rng_axis[-1], rng_axis[0]]

    # plotting image
    fig.append_trace(
        go.Heatmap(
            x=np.linspace(extent[0], extent[1], image_db.shape[1]),
            y=np.linspace(extent[2], extent[3], image_db.shape[0]),
            z=image_db,
            zmin=image_db.max() - 40,
            zmax=image_db.max(),
            colorscale="Jet",
            showscale=False,
            name="Target Response",
            hovertemplate="<br>".join(
                ["<b>Azimuth [m]</b>: %{x}", "<b>Range [m]</b>: %{y}", "<b>Power [dB]</b>: %{z}"]
            ),
        ),
        row=1,
        col=4,
    )
    # customizing IRF plot
    fig.update_yaxes(
        title_text="<b>Range [m]</b>",
        # autorange="reversed",
        scaleratio=1,
        row=1,
        col=4,
    )
    fig.update_xaxes(title_text="<b>Azimuth [m]</b>", row=1, col=4)

    # adding peak reactangle
    roi_peak = np.asarray(rcs_data_graph.roi_peak)
    roi_rng = (
        roi_peak[:2] / rcs_data_graph.interp_factor - rcs_data_graph.roi_size[0] / 2
    ) * rcs_data_graph.rng_step_distance
    roi_az = (
        roi_peak[2:] / rcs_data_graph.interp_factor - rcs_data_graph.roi_size[1] / 2
    ) * rcs_data_graph.az_step_distance

    fig.add_shape(
        type="rect", x0=roi_az[0], y0=roi_rng[0], x1=roi_az[1], y1=roi_rng[1], line={"color": "Red"}, row=1, col=4
    )

    # plotting background corner rectangles
    for rect in rcs_data_graph.roi_background:
        rect = np.asarray(rect).astype(float)
        rect[:2] = (rect[:2] - rcs_data_graph.roi_size[0] / 2) * rcs_data_graph.rng_step_distance
        rect[2:] = (rect[2:] - rcs_data_graph.roi_size[1] / 2) * rcs_data_graph.az_step_distance

        fig.add_shape(
            type="rect", x0=rect[2], y0=rect[0], x1=rect[3], y1=rect[1], line={"color": "Magenta"}, row=1, col=4
        )

    # plotting tables
    tbl_content = {
        "Range Resolution [m]": data_values["range_resolution_[m]"],
        "Azimuth Resolution [m]": data_values["azimuth_resolution_[m]"],
        "Range PSLR [dB]": data_values["range_pslr_[dB]"],
        "Azimuth PSLR [dB]": data_values["azimuth_pslr_[dB]"],
        "Range ISLR [dB]": data_values["range_islr_[dB]"],
        "Azimuth ISLR [dB]": data_values["azimuth_islr_[dB]"],
        "Slant Range Loc. Err. [m]": data_values["slant_range_localization_error_[m]"],
        "Azimuth Loc. Err. [m]": data_values["azimuth_localization_error_[m]"],
        "Gound Range Loc. Err. [m]": data_values["ground_range_localization_error_[m]"],
        "RCS [dB]": data_values["rcs_[dB]"],
        "Clutter [dB]": data_values["clutter_[dB]"],
        "Peak Phase Error [deg]": data_values["peak_phase_error_[deg]"],
    }
    fig.add_trace(
        go.Table(
            header=dict(
                values=["<b>" + c + "</b>" for c in list(tbl_content.keys())],
                fill_color="#8C8CBA",
                font=dict(color="black", size=11),
                align="center",
            ),
            cells=dict(
                values=[np.round(c, 4) for c in tbl_content.values()],
                fill_color="#D9D9E8",
                font=dict(color="black", size=12),
                align="center",
            ),
        ),
        row=2,
        col=1,
    )

    # final customization and saving
    fig.update_layout(
        clickmode="event+select",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        # height=800,
        # width=1500,
        title={
            "text": "<b>Point Target Analysis - Target #" + label + " </b>",
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 20},
        },
        title_font_color="black",
        title_x=0.5,
        title_y=0.98,
        legend={"itemsizing": "constant"},
    )

    fig.write_html(out_dir.joinpath(label + "_PointTargetAnalysisInteractive.html").as_posix(), include_plotlyjs="cdn")
