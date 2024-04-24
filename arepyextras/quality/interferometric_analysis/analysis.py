# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""2D Coherence computation from interferogram"""
from __future__ import annotations

import logging

from arepyextras.quality.interferometric_analysis.config import InterferometricConfig
from arepyextras.quality.interferometric_analysis.custom_dataclasses import (
    InterferometricCoherenceOutput,
)
from arepyextras.quality.interferometric_analysis.support import (
    coherence_2d_histogram_computation_core,
    coherence_computation_core,
)
from arepyextras.quality.io.quality_input_protocol import QualityInputProduct

# syncing with logger
log = logging.getLogger("quality_analysis")


def interferometric_analysis(
    product: QualityInputProduct, config: InterferometricConfig | None = None
) -> list[InterferometricCoherenceOutput]:
    """Interferometric analysis of input product based on selected options in configuration.
    Input product can be an interferogram or a coherence map, it can be merged or separated into bursts.

    If the input product contains interferogram data, the coherence map must be computed and therefore the
    "enable_coherence_computation" flag in configuration should be set to True.

    If the input product contains coherence data, "enable_coherence_computation" flag in configuration should be left
    to its default value of False, to compute just coherence histograms.

    Parameters
    ----------
    product : QualityInputProduct
        object satisfying the QualityInputProduct protocol
    config : InterferometricConfig | None, optional
        InterferometricConfig configuration dataclass, by default None

    Returns
    -------
    list[InterferometricCoherenceOutput]
        an InterferometricCoherenceOutput dataclass for each channel
    """
    if config is None:
        config = InterferometricConfig()

    if config.enable_coherence_computation:
        log.info(f"Coherence computation enabled: product {product.name} contains interferogram data")
    else:
        log.info(f"Coherence computation disabled: product {product.name} contains coherence data")

    output = []
    for idx, channel in enumerate(product.channels_list):
        # loading channel data
        if config.enable_coherence_computation:
            log.info(f"Computing coherence for Channel {channel}  {idx+1}/{len(product.channels_list)}")
        else:
            log.info(f"Computing coherence histograms for Channel {channel}  {idx+1}/{len(product.channels_list)}")
        channel_data = product.get_channel_data(channel_id=channel)
        # for each burst
        for idx, burst_lines in enumerate(channel_data.lines_per_burst):
            cumulative_burst_lines = burst_lines // 2 + idx * burst_lines
            mid_range_pixel = int(channel_data.slant_range_axis.size // 2)
            # data transposes because they are needed (lines, samples) but read_data provided them (samples, lines)
            data = channel_data.read_data(
                azimuth_index=cumulative_burst_lines,
                range_index=mid_range_pixel,
                cropping_size=(mid_range_pixel * 2, burst_lines),
            ).T
            if config.enable_coherence_computation:
                # computing coherence
                data = coherence_computation_core(data, kernel_size=config.coherence_kernel)

            histograms = coherence_2d_histogram_computation_core(coherence=data, config=config)
            output.append(
                InterferometricCoherenceOutput(
                    channel_name=channel,
                    swath=channel_data.swath_name,
                    burst=idx,
                    polarization=channel_data.polarization,
                    coherence=data,
                    coherence_histograms=histograms,
                )
            )
    return output
