# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Interferometry auxiliary functions module"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from netCDF4 import Dataset
from scipy.signal import convolve2d

from arepyextras.quality.interferometric_analysis.config import InterferometricConfig
from arepyextras.quality.interferometric_analysis.custom_dataclasses import (
    InterferometricCoherence2DHistograms,
    InterferometricCoherenceOutput,
)

# syncing with logger
log = logging.getLogger("quality_analysis")

# ignoring invalid divide by 0 warning
np.seterr(invalid="ignore")


def coherence_histograms_to_netcdf(data: InterferometricCoherenceOutput, output_dir: str | Path) -> None:
    """Saving Coherence 2D histograms to NetCDF4 file.

    Parameters
    ----------
    data : InterferometricCoherenceOutput
        InterferometricCoherenceOutput dataclass
    output_dir : str | Path
        path where to save the NetCDF file
    """
    output_dir = Path(output_dir)

    out_name = "coherence_histograms_" + data.swath + "_" + data.polarization.name
    log.info(f"Saving {out_name} data to NetCDF file")

    root = Dataset(output_dir.joinpath(out_name).with_suffix(".nc"), "w", format="NETCDF4")
    root.swath = data.swath
    root.channel = data.channel_name
    root.burst = data.burst
    root.polarization = data.polarization.name

    # creating common dimensions
    root.createDimension("coherence_bins", data.coherence_histograms.coherence_bin_edges.size - 1)
    root.createDimension("azimuth_blocks", data.coherence_histograms.azimuth_histogram.shape[1])
    root.createDimension("range_blocks", data.coherence_histograms.range_histogram.shape[1])

    # creating coherence bins variable
    coherence_bins = root.createVariable(
        "coherence_bins", data.coherence_histograms.coherence_bin_edges.dtype, "coherence_bins"
    )
    coherence_bins.unit = ""
    coherence_bins[:] = data.coherence_histograms.coherence_bin_edges[:-1]

    # creating azimuth histogram variable
    az_hist = root.createVariable(
        "azimuth_histogram", data.coherence_histograms.azimuth_histogram.dtype, ("coherence_bins", "azimuth_blocks")
    )
    az_hist.unit = ""
    az_hist[:] = data.coherence_histograms.azimuth_histogram

    # creating range histogram variable
    rng_hist = root.createVariable(
        "range_histogram", data.coherence_histograms.range_histogram.dtype, ("coherence_bins", "range_blocks")
    )
    rng_hist.unit = ""
    rng_hist[:] = data.coherence_histograms.range_histogram

    root.close()


def coherence_2d_histogram_computation_core(
    coherence: np.ndarray, config: InterferometricConfig
) -> InterferometricCoherence2DHistograms:
    """Computing 2D coherence histograms along range and azimuth directions

    These discrete histograms are computed by splitting the input coherence array along a direction (azimuth or range) in
    several blocks and then computing the histogram of coherence for that block with a fixed number of coherence bins in
    a fixed range (so that all sub-blocks will match the same coherence intervals).

    Parameters
    ----------
    coherence : np.ndarray
        coherence map array
    config : InterferometricConfig
        InterferometricConfig configuration dataclass

    Returns
    -------
    InterferometricCoherence2DHistograms
        2D histogram interferometric coherence data
    """
    coherence = np.abs(coherence)
    if config.azimuth_blocks_number is None:
        # automatically defining number of azimuth blocks
        config.azimuth_blocks_number = int(1.5 * np.sqrt(coherence.shape[0]))

    if config.range_blocks_number is None:
        # automatically defining number of range blocks
        config.range_blocks_number = int(1.5 * np.sqrt(coherence.shape[1]))

    # bin edges
    bin_edges = np.histogram_bin_edges(coherence, bins=config.coherence_bins_number, range=(0, 1))

    # splitting coherence array along azimuth direction in blocks to compute histograms on
    cor_az_blocks = np.array_split(coherence, config.azimuth_blocks_number, axis=0)
    histograms_az = [
        np.histogram(np.ma.masked_invalid(c).compressed(), bins=config.coherence_bins_number, range=(0, 1))
        for c in cor_az_blocks
    ]
    az_counts = np.stack([h[0] for h in histograms_az], axis=1)

    # splitting coherence array along range direction in blocks to compute histograms on
    cor_rng_blocks = np.array_split(coherence, config.range_blocks_number, axis=1)
    histograms_rng = [
        np.histogram(np.ma.masked_invalid(c).compressed(), bins=config.coherence_bins_number, range=(0, 1))
        for c in cor_rng_blocks
    ]
    rng_counts = np.stack([h[0] for h in histograms_rng], axis=1)

    return InterferometricCoherence2DHistograms(
        coherence_bin_edges=bin_edges,
        azimuth_histogram=az_counts,
        range_histogram=rng_counts,
    )


def coherence_computation_interferogram_core(data: np.ndarray, kernel_size: int | tuple[int, int] = 15) -> np.ndarray:
    """Core algorithm to compute coherence by 2D convolution of input interferogram data with a boxcar filter.

    Coherence is defined as the ratio between the interferogram convolution with a boxcar filter of the complex data
    (with phase information) and the same convolution performed on the absolute of the input data.

    .. math::

        \\hat\\gamma = \\frac{\\sum_{i=1}^{N} u_i}{\\sum_{i=1}^{N} |u_i|}

    Parameters
    ----------
    data : np.ndarray
        interferogram data, with shape (lines, samples)
    kernel_size : int | tuple[int, int], optional
        size of the boxcar kernel, if an integer is provided, kernel will be a square with a side of that size, while
        if the input is a tuple, that is the shape of the final kernel, by default 15

    Returns
    -------
    np.ndarray
        coherence array
    """
    # creating kernel "boxcar" from kernel_size
    kernel_norm = boxcar_kernel_setup(kernel_size)

    # computing the convolution on the phase of the image
    filtered_image = convolve2d(
        data,
        kernel_norm,
        mode="same",
    )
    filtered_abs_image = convolve2d(
        np.abs(data),
        kernel_norm,
        mode="same",
    )

    # coherence is obtained by dividing the phase convolution by the absolute convolution
    coherence = np.ma.masked_invalid(filtered_image / filtered_abs_image)

    # masking where phase > abs, aka ratio > 1
    return np.ma.masked_where(np.abs(coherence) > 1, coherence)


def coherence_computation_co_registered_core(
    data_1: np.ndarray, data_2: np.ndarray, kernel_size: int | tuple[int, int] = 15
) -> np.ndarray:
    """Core algorithm to compute coherence by 2D convolution of input co-registered products with a boxcar filter.

    Coherence is defined as the ratio between the interferogram convolution with a boxcar filter of the complex data
    (with phase information) and the square root of the product of the two input data squared and convoluted.

    .. math::

        \\hat\\gamma = \\frac{\\sum_{i=1}^{N} u_i v_i^*}{\\sqrt{\\sum_{i=1}^{N} |u_i|^2 \\sum_{i=1}^{N} |v_i|^2}}

    where :math:`\\hat\\gamma` is the coherence, :math:`u` is the first product data and :math:`v` is the second co-registered
    data.

    Reference: `https://www.esa.int/esapub/tm/tm19/TM-19_ptC.pdf <https://www.esa.int/esapub/tm/tm19/TM-19_ptC.pdf>`_

    Parameters
    ----------
    data_1 : np.ndarray
        first co-registered product data
    data_2 : np.ndarray
        second co-registered product data
    kernel_size : int | tuple[int, int], optional
        size of the boxcar kernel, if an integer is provided, kernel will be a square with a side of that size, while
        if the input is a tuple, that is the shape of the final kernel, by default 15

    Returns
    -------
    np.ndarray
        coherence array
    """

    # creating kernel "boxcar" from kernel_size
    kernel_norm = boxcar_kernel_setup(kernel_size)

    # computing the convolution on the phase of the image
    filtered_image = convolve2d(
        data_1 * data_2.conj(),
        kernel_norm,
        mode="same",
    )
    filtered_abs_image_1 = convolve2d(
        np.abs(data_1) ** 2,
        kernel_norm,
        mode="same",
    )
    filtered_abs_image_2 = convolve2d(
        np.abs(data_2) ** 2,
        kernel_norm,
        mode="same",
    )

    # coherence is obtained by dividing the phase convolution by the absolute convolution
    coherence = np.ma.masked_invalid(filtered_image / np.sqrt(filtered_abs_image_1 * filtered_abs_image_2))

    # masking where phase > abs, aka ratio > 1
    return np.ma.masked_where(np.abs(coherence) > 1, coherence)


def boxcar_kernel_setup(kernel_size: int | tuple[int, int]) -> np.ndarray:
    """Creating the normalized boxcar kernel from its size.

    Parameters
    ----------
    kernel_size : int | tuple[int, int]
        kernel size

    Returns
    -------
    np.ndarray
        normalized boxcar kernel of the given size
    """

    # creating kernel "boxcar" from kernel_size
    if isinstance(kernel_size, int):
        # if a single value is provided, kernel is a square of side equal to kernel_size
        kernel = np.ones((kernel_size, kernel_size))
    else:
        # otherwise creating a custom rectangular kernel from input shape
        assert len(kernel_size) == 2
        kernel = np.ones(kernel_size)

    # normalizing kernel to remove gain
    return kernel / np.sum(kernel)
