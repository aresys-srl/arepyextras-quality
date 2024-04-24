# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Interferometry graphical output module"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from arepyextras.quality.interferometric_analysis.config import InterferometricConfig
from arepyextras.quality.interferometric_analysis.custom_dataclasses import (
    InterferometricCoherence2DHistograms,
    InterferometricCoherenceOutput,
)

# syncing with logger
log = logging.getLogger("quality_analysis")


def generate_coherence_graphs(
    data: InterferometricCoherenceOutput, output_dir: str | Path, config: InterferometricConfig | None = None
) -> None:
    """Computing coherence graphs from Arepyextras Quality InterferometricCoherenceOutput coherence computation results.

    Parameters
    ----------
    data : InterferometricCoherenceOutput
        InterferometricCoherenceOutput dataclass with results from coherence computation
    output_dir : str | Path
        output directory where to save the graph
    config : InterferometricConfig | None, optional
        interferometric configuration, by default None
    """
    if config is None:
        config = InterferometricConfig()
        config.azimuth_blocks_number = 20
        config.range_blocks_number = 50
    tag = "_".join([data.swath, data.polarization.name])
    return coherence_graph_core(
        coherence=data.coherence, histograms=data.coherence_histograms, tag=tag, output_dir=Path(output_dir)
    )


def coherence_graph_core(
    coherence: np.ndarray,
    histograms: InterferometricCoherence2DHistograms,
    output_dir: Path,
    tag: str = "",
) -> None:
    """Generating interferogram coherence graph.

    Parameters
    ----------
    coherence : InterferometricCoherenceOutput
        coherence dataclass
    output_dir : Path
        output directory where to save the graph
    tag : str, optional
        string tag to be added to the plot title and filename, by default ""
    """

    coherence = np.abs(coherence)
    coherence_bins_number = histograms.coherence_bin_edges.size - 1

    filename = output_dir.joinpath("coherence_graph_" + tag + ".png")
    log.info(f"Generating {filename.name}")

    # graph
    fig = plt.figure(figsize=(15, 8), dpi=250)
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=(4, 1),
        height_ratios=(1, 2),
        left=0.05,
        right=0.95,
        bottom=0.1,
        top=0.9,
        wspace=0.05,
        hspace=0.15,
    )
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    # plot 0, upper left corner, coherence histogram along range direction
    ax0.imshow(histograms.range_histogram, aspect="auto", origin="lower")
    ax0.set_yticks(
        np.linspace(0, histograms.range_histogram.shape[0], coherence_bins_number)[:: coherence_bins_number // 4],
        histograms.coherence_bin_edges[: -1 : coherence_bins_number // 4],
    )
    ax0.set_xticks([])
    ax0.set_title("Coherence Histogram [Range]", fontsize=14)

    # plot 1, upper right corner, must not be shown
    ax1.set_visible(False)

    # plot 2, lower left corner, whole coherence array
    ax2.imshow(coherence, interpolation="nearest", aspect="auto")
    ax2.invert_yaxis()
    ax2.set_title("Coherence Map", fontsize=14)

    # plot 3, lower right corner, coherence histogram along azimuth direction
    ax3.imshow(histograms.azimuth_histogram.T, aspect="auto", origin="lower")
    ax3.set_xticks(
        np.linspace(0, histograms.azimuth_histogram.shape[0], coherence_bins_number)[:: coherence_bins_number // 4],
        histograms.coherence_bin_edges[: -1 : coherence_bins_number // 4],
    )
    ax3.set_yticks([])
    ax3.set_title("Coherence Histogram [Azimuth]", fontsize=14)

    fig.suptitle(f"Coherence Graph {tag.replace('_', ' ')}", fontweight="bold", fontsize=18)

    fig.savefig(filename)
    plt.close("all")
