# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Support functions and parameters for Unit Testing"""

from typing import Optional

import numpy as np
from scipy.fft import fft2, ifft2

from arepyextras.quality.core.signal_processing import locate_max_2d_interp
from arepyextras.quality.point_targets_analysis.custom_dataclasses import IRFDataOutput

ref_data_irf_results = IRFDataOutput(
    range_resolution=0.9831505486028405,
    azimuth_resolution=0.9831505486028405,
    pslr_2d=-13.272021869964892,
    islr_2d=-7.208789038956241,
    sslr_2d=-23.005687315280902,
    azimuth_pslr=-13.272021869964906,
    azimuth_islr=-10.219088995596044,
    azimuth_sslr=-23.005687315280902,
    range_pslr=-13.272021869964892,
    range_islr=-10.219088995596062,
    range_sslr=-23.005687315280916,
    azimuth_localization_error=-1.1392089049877541e-07,
    ground_range_localization_error=-1.1392089049877541e-07,
    slant_range_localization_error=-1.1392089049877541e-07,
)

ref_data_rcs_results = {"clutter": -91.7106822164182, "rcs": 77.228675375588, "scr": 91.712713592383}

default_input_data_generation = {
    "lines": 128,
    "samples": 128,
    "lines_step": 0.00016420361247947455,
    "samples_step": 3.57142857142857e-09,
    "samples_start": 0.00400438831877932,
    "fc_hz": 9.6e9,
}


def generate_target_data(
    bandwidths: tuple[float, float],
    axes: tuple[np.ndarray, np.ndarray],
    target_relative_pos: tuple[float, float],
    swath_range_start: float,
    fc_hz: float,
    window: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Generating theorical target response data matrix from input swath axes and raster parameters. It also applies a
    weighting window if provided.

    Parameters
    ----------
    bandwidths : tuple[float, float]
        range [0] and azimuth [1] signal bandwidths
    axes : tuple[np.ndarray, np.ndarray]
        range [0] and azimuth [1] axes
    target_relative_pos : tuple[float, float]
        range [0] and azimuth [1] relative times in the swath (start times not included), both floats
    swath_range_start : float
        range start time of the swath
    fc_hz : int
        carrier frequency in Hz
    window : Optional[np.ndarray], optional
        weighting window for generated data, by default None

    Returns
    -------
    np.ndarray
        data 2D array containing the target response
    """

    # creating theoretical response data given input parameters
    array = np.outer(
        np.sinc(bandwidths[0] * (axes[0] - target_relative_pos[0])),
        np.sinc(bandwidths[1] * (axes[1] - target_relative_pos[1])),
    )
    array = array * np.exp(2j * np.pi * fc_hz * (target_relative_pos[0] + swath_range_start))

    # performing fft of data and then inverse fft on windowed data
    if window is not None:
        return ifft2(fft2(array) * window)

    return array


def generate_data_for_test(
    lines: int,
    samples: int,
    lines_step: float,
    samples_step: float,
    samples_start: float,
    fc_hz: float,
    perc: float = 0.9,
    window: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, tuple[float, float], tuple[float, float]]:
    """Generating testing data from generic setup info.

    Parameters
    ----------
    lines : int
        number of azimuth lines
    samples : int
        number of range samples
    lines_step : float
        azimuth step
    samples_step : float
        range step
    samples_start : float
        initial sample range time
    fc_hz : float
        carrier frequency
    perc : float, optional
        optional percentage value, by default 0.9
    window :  Optional[np.ndarray], optional
        2D wighting window

    Returns
    -------
    tuple[np.ndarray, tuple[float, float], tuple[float, float]]
        2D generated data array
        peak position in the generated array, row[0] and col[0] subpixel precision
        target position in the generated array, row[0] and col[0]
    """

    # computing bandwidths
    azimuth_bandwidth = perc / lines_step
    range_bandwidth = perc / samples_step

    # computing axes
    az_axis = np.arange(0, lines) * lines_step
    rng_axis = np.arange(0, samples) * samples_step

    # computing target position (set at center of the image)
    target_az_rel = (np.ceil(lines / 2) + 0.5) * lines_step
    target_rng_rel = (np.ceil(samples / 2) + 0.5) * samples_step

    # generating theorical data
    data = generate_target_data(
        bandwidths=(range_bandwidth, azimuth_bandwidth),
        axes=(rng_axis, az_axis),
        target_relative_pos=(target_rng_rel, target_az_rel),
        swath_range_start=samples_start,
        fc_hz=fc_hz,
        window=window,
    )

    # find main lobe peak with subpixel precision
    _, *peak_pos = locate_max_2d_interp(data)
    target_pos = np.array([data.shape[0] / 2 + 0.5, data.shape[1] / 2 + 0.5])

    return data, peak_pos, target_pos
