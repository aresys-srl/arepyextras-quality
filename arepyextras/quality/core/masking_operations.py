# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Isolating masking operations for PSLR, ISLR and SSLR"""

from __future__ import annotations

import numpy as np
from scipy import interpolate as sp_interp
from scipy.signal import convolve2d

import arepyextras.quality.core.signal_processing as sp
from arepyextras.quality.core.generic_dataclasses import (
    DecibelConversion,
    MaskingMethod,
)


def _masking_lobes_from_lines_intersections(
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    x_values: list,
    y_values: list,
    side_lobes_directions: tuple[float, float],
) -> np.ndarray:
    """Masking data from input data by creating straight lines using side lobes directions as angular coefficients.
    y = side_lobes_dir * x + q

    Parameters
    ----------
    x_axis : np.ndarray
        x axis
    y_axis : np.ndarray
        y axis
    x_values : list
        x values of the straight lines
    y_values : list
        y values of the straight lines
    side_lobes_directions : tuple[float, float]
        range and azimuth cuts angular coefficients in samples.

    Returns
    -------
    np.ndarray
        masked array from input axis and lines intersections
    """

    # building the q values, lines interecpts
    line_intercepts = [
        y_values[0] - side_lobes_directions[1] * x_values[0],
        y_values[1] - side_lobes_directions[0] * x_values[1],
        y_values[2] - side_lobes_directions[1] * x_values[2],
        y_values[3] - side_lobes_directions[0] * x_values[3],
    ]

    x_grid, y_grid = np.meshgrid(x_axis, y_axis)

    # assembling lines from each constituent
    lines = [
        side_lobes_directions[1] * x_grid + line_intercepts[0],
        side_lobes_directions[0] * x_grid + line_intercepts[1],
        side_lobes_directions[1] * x_grid + line_intercepts[2],
        side_lobes_directions[0] * x_grid + line_intercepts[3],
    ]

    # determine masking comparing y values to lines
    if line_intercepts[0] < line_intercepts[2]:
        mask_1 = (y_grid > lines[0]) & (y_grid < lines[2])
    else:
        mask_1 = (y_grid < lines[0]) & (y_grid > lines[2])

    if line_intercepts[1] < line_intercepts[3]:
        mask_2 = (y_grid > lines[1]) & (y_grid < lines[3])
    else:
        mask_2 = (y_grid < lines[1]) & (y_grid > lines[3])

    # boolean mask intersection
    mask = mask_1 & mask_2

    return mask.astype("int")


def generate_peak_mask(data: np.ndarray) -> np.ndarray:
    """Generating a mask of the input array to isolate the main lobe from the background, filling with 1s where the
    main lobe region lies. Masking centered on main lobe and covering it whole up to first zeroes.

    Parameters
    ----------
    data : np.ndarray
        2D array

    Returns
    -------
    np.ndarray
        2D mask array of 0s and 1s
    """

    # finding the peak position
    max_row, max_col = sp.locate_max_2d(np.abs(data))

    # converting absolute of input data to decibel and derive it
    def func(arg):
        return np.diff(sp.convert_to_db(np.abs(arg), mode=DecibelConversion.AMPLITUDE))

    # selecting all decreasing part of sinc lobes, for each direction
    rg_id_dx = np.argwhere(func(data[:max_row, max_col]) < 0).squeeze()
    rg_id_sx = np.argwhere(func(data[max_row:, max_col]) > 0).squeeze()
    az_id_dx = np.argwhere(func(data[max_row, :max_col]) < 0).squeeze()
    az_id_sx = np.argwhere(func(data[max_row, max_col:]) > 0).squeeze()

    break_cond = np.logical_or.reduce((rg_id_dx.size == 0, rg_id_sx.size == 0, az_id_dx.size == 0, az_id_sx.size == 0))

    if break_cond:
        mask = np.ones_like(data)
        return mask

    # initializing the mask
    mask = np.zeros_like(data)
    # filling in mask with 1s where the main lobe lies
    # finding the first zeroes of the main lobe in all directions and filling inside them (towards the peak center)
    mask[rg_id_dx[-1] + 1 : max_row + rg_id_sx[0] + 1, az_id_dx[-1] + 1 : max_col + az_id_sx[0] + 1] = 1

    return mask


def generate_peak_mask_lobes(
    data: np.ndarray, side_lobes_directions: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generating a mask of the input array to isolate the main lobe from the background, filling with 1s where the
    main lobe region lies. Masking centered on main lobe and covering it whole up to first zeroes.

    Parameters
    ----------
    data : np.ndarray
        array with a peak to be masked
    side_lobes_directions : tuple[float, float]
        range and azimuth cuts angular coefficients in samples

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        range filled indexes,
        azimuth filled indexes,
        2D mask array of 0s and 1s
    """
    max_row, max_col = sp.locate_max_2d(np.abs(data))

    # Compute data size and axes
    block_dim_y, block_dim_x = data.shape
    dim_ratio = block_dim_x / block_dim_y
    a_x = np.arange(block_dim_x) - max_col
    a_y = np.arange(block_dim_y) - max_row

    # lobes conditions
    rng_lobes_cond = np.abs(side_lobes_directions[0] * dim_ratio) > 1
    az_lobes_cond = np.abs(side_lobes_directions[1] * dim_ratio) > 1

    # Find mask borders using data cuts
    interp_func = sp_interp.RegularGridInterpolator(
        points=(a_y, a_x), method="linear", values=data, fill_value=0, bounds_error=False
    )

    # managing range direction
    if rng_lobes_cond:
        data_rng_cut = interp_func((a_y, a_y / side_lobes_directions[0]))
        data_rng_cut_db = sp.convert_to_db(np.abs(data_rng_cut), mode=DecibelConversion.AMPLITUDE)

        rng_id_dx = np.argwhere(np.diff(data_rng_cut_db[:max_row]) < 0).squeeze()
        rng_id_sx = np.argwhere(np.diff(data_rng_cut_db[max_row:]) > 0).squeeze()

        ml_ind_rng = np.zeros(data.shape[0])
        ml_ind_rng[rng_id_dx[-1] + 1 : max_row + rng_id_sx[0] + 1] = 1

        y_1 = a_y[rng_id_dx[-1] + 1]
        x_1 = y_1 / side_lobes_directions[0]
        y_3 = a_y[max_row + rng_id_sx[0]]
        x_3 = y_3 / side_lobes_directions[0]

    else:
        data_rng_cut = interp_func((a_x * side_lobes_directions[0], a_x))
        data_rng_cut_db = sp.convert_to_db(np.abs(data_rng_cut), mode=DecibelConversion.AMPLITUDE)

        rng_id_dx = np.argwhere(np.diff(data_rng_cut_db[:max_col]) < 0).squeeze()
        rng_id_sx = np.argwhere(np.diff(data_rng_cut_db[max_col:]) > 0).squeeze()

        ml_ind_rng = np.zeros(data.shape[1])
        ml_ind_rng[rng_id_dx[-1] + 1 : max_col + rng_id_sx[0] + 1] = 1

        x_1 = a_x[rng_id_dx[-1] + 1]
        y_1 = side_lobes_directions[0] * x_1
        x_3 = a_x[max_col + rng_id_sx[0]]
        y_3 = side_lobes_directions[0] * x_3

    # managing azimuth direction
    if az_lobes_cond:
        data_az_cut = interp_func((a_y, a_y / side_lobes_directions[1]))
        data_az_cut_db = sp.convert_to_db(np.abs(data_az_cut), mode=DecibelConversion.AMPLITUDE)

        az_id_dx = np.argwhere(np.diff(data_az_cut_db[:max_row]) < 0).squeeze()
        az_id_sx = np.argwhere(np.diff(data_az_cut_db[max_row:]) > 0).squeeze()

        ml_ind_az = np.zeros(data.shape[0])
        ml_ind_az[az_id_dx[-1] + 1 : max_row + az_id_sx[0] + 1] = 1

        y_2 = a_y[az_id_dx[-1] + 1]
        x_2 = y_2 / side_lobes_directions[1]
        y_4 = a_y[max_row + az_id_sx[0]]
        x_4 = y_4 / side_lobes_directions[1]
    else:
        data_az_cut = interp_func((a_x * side_lobes_directions[1], a_x))
        data_az_cut_db = sp.convert_to_db(np.abs(data_az_cut), mode=DecibelConversion.AMPLITUDE)

        az_id_dx = np.argwhere(np.diff(data_az_cut_db[:max_col]) < 0).squeeze()
        az_id_sx = np.argwhere(np.diff(data_az_cut_db[max_col:]) > 0).squeeze()

        ml_ind_az = np.zeros(data.shape[1])
        ml_ind_az[az_id_dx[-1] + 1 : max_col + az_id_sx[0] + 1] = 1

        x_2 = a_x[az_id_dx[-1] + 1]
        y_2 = side_lobes_directions[1] * x_2
        x_4 = a_x[max_col + az_id_sx[0]]
        y_4 = side_lobes_directions[1] * x_4

    x_vals = [x_1, x_2, x_3, x_4]
    y_vals = [y_1, y_2, y_3, y_4]

    return (
        ml_ind_rng,
        ml_ind_az,
        _masking_lobes_from_lines_intersections(
            x_axis=a_x, y_axis=a_y, x_values=x_vals, y_values=y_vals, side_lobes_directions=side_lobes_directions
        ),
    )


def generate_rectangular_mask(x_axis: np.ndarray, y_axis: np.ndarray, size_x: float, size_y: float) -> np.ndarray:
    """Generate rectangular mask of specified size starting from given axes.

    Parameters
    ----------
    x_axis : np.ndarray
        x axis
    y_axis : np.ndarray
        y axis
    size_x : float
        mask size along x
    size_y : float
        mask size along y

    Returns
    -------
    np.ndarray
        2D mask of 1s of the given size
    """

    y_indexes_in_halfwindow = np.abs(y_axis) <= size_y / 2
    x_indexes_in_halfwindow = np.abs(x_axis) <= size_x / 2
    mask = np.zeros((y_axis.size, x_axis.size))
    mask[np.ix_(y_indexes_in_halfwindow, x_indexes_in_halfwindow)] = 1

    return mask


def generate_resolution_mask(
    x_axis: np.ndarray, y_axis: np.ndarray, res_x: float, res_y: float, multiplier_x: float, multiplier_y: float
) -> np.ndarray:
    """Generate a rectangular mask with size equal to res * multiplier.

    Parameters
    ----------
    x_axis : np.ndarray
        x axis
    y_axis : np.ndarray
        y axis
    res_x : float
        resolution along x axis
    res_y : float
        resolution along y axis
    multiplier_x : float
        size of the mask for x axis in number of resolution cells
    multiplier_y : float
        size of the mask for y axis in number of resolution cells

    Returns
    -------
    np.ndarray
        2D mask array
    """

    window_x = multiplier_x * res_x
    window_y = multiplier_y * res_y

    return generate_rectangular_mask(x_axis, y_axis, window_x, window_y)


def generate_resolution_mask_lobes(
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    res_x: float,
    res_y: float,
    multiplier_x: float,
    multiplier_y: float,
    side_lobes_directions: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a mask containing the data to the first resolution cell in both directions taking into account the side
    lobes directions.

    Parameters
    ----------
    x_axis : np.ndarray
        x axis
    y_axis : np.ndarray
        y axis
    res_x : float
        resolution along x axis
    res_y : float
        resolution along y axis
    multiplier_x : float
        size of the mask for x axis in number of resolution cells
    multiplier_y : float
        size of the mask for y axis in number of resolution cells
    side_lobes_directions : tuple[float, float]
        range and azimuth cuts angular coefficients in samples

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        range filled index,
        azimuth filled indexes,
        2D mask array
    """

    # initialize mask parameters
    y_dim = len(y_axis)
    x_dim = len(x_axis)
    dim_ratio = x_dim / y_dim
    subpixel_x = (multiplier_x * res_x) / 2
    subpixel_y = (multiplier_y * res_y) / 2

    # lobes conditions
    rng_lobes_cond = np.abs(side_lobes_directions[0] * dim_ratio) > 1
    az_lobes_cond = np.abs(side_lobes_directions[1] * dim_ratio) > 1

    # managing range direction
    if rng_lobes_cond:
        ml_rng_id = np.abs(y_axis) <= subpixel_y
        y_1 = -subpixel_y
        x_1 = y_1 / side_lobes_directions[0]
        y_3 = subpixel_y
        x_3 = y_3 / side_lobes_directions[0]
    else:
        ml_rng_id = np.abs(x_axis) <= subpixel_y
        x_1 = -subpixel_y
        y_1 = side_lobes_directions[0] * x_1
        x_3 = subpixel_y
        y_3 = side_lobes_directions[0] * x_3

    # managing azimuth direction
    if az_lobes_cond:
        ml_az_id = np.abs(y_axis) <= subpixel_x
        y_2 = -subpixel_x
        x_2 = y_2 / side_lobes_directions[1]
        y_4 = subpixel_x
        x_4 = y_4 / side_lobes_directions[1]
    else:
        ml_az_id = np.abs(x_axis) <= subpixel_x
        x_2 = -subpixel_x
        y_2 = side_lobes_directions[1] * x_2
        x_4 = subpixel_x
        y_4 = side_lobes_directions[1] * x_4

    x_vals = [x_1, x_2, x_3, x_4]
    y_vals = [y_1, y_2, y_3, y_4]

    return (
        ml_rng_id,
        ml_az_id,
        _masking_lobes_from_lines_intersections(
            x_axis=x_axis, y_axis=y_axis, x_values=x_vals, y_values=y_vals, side_lobes_directions=side_lobes_directions
        ),
    )


def get_interpolated_lobes_cuts(
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    values: np.ndarray,
    side_lobes_directions: tuple[float, float],
    method: str = "linear",
) -> tuple[np.ndarray, np.ndarray]:
    """Extract range and azimuth cut/profiles by interpolation of input data for IRF analysis in case of squinted data,
    a.k.a. side lobes directions.

    Parameters
    ----------
    x_axis : np.ndarray
        x axis array
    y_axis : np.ndarray
        y axis array
    values : np.ndarray
        2D array where to extract the profiles
    side_lobes_directions : tuple[float, float]
        range and azimuth cuts angular coefficients in samples
    method : str, optional
        interpolation method using RegularGridInterpolator, by default "linear"

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        range cut,
        azimuth cut
    """

    # lobes conditions
    dim_ratio = values.shape[1] / values.shape[0]
    rng_lobes_cond = np.abs(side_lobes_directions[0] * dim_ratio) > 1
    az_lobes_cond = np.abs(side_lobes_directions[1] * dim_ratio) > 1

    # interpolating input data using a regulat grid generated by input axes
    interp_func = sp_interp.RegularGridInterpolator(
        points=(y_axis, x_axis), method=method, values=values, fill_value=0, bounds_error=False
    )

    if rng_lobes_cond:
        rng_cut = interp_func((y_axis, y_axis / side_lobes_directions[0]))
    else:
        rng_cut = interp_func((side_lobes_directions[0] * x_axis, x_axis))

    if az_lobes_cond:
        az_cut = interp_func((y_axis, y_axis / side_lobes_directions[1]))
    else:
        az_cut = interp_func((side_lobes_directions[1] * x_axis, x_axis))

    return rng_cut, az_cut


def pslr_masking(
    data: np.ndarray,
    mask_flag: MaskingMethod,
    resolution: tuple[float, float],
    peak_pos: tuple[int, int],
    side_lobes_directions: tuple[float, float],
    interp_factor: int = 16,
    number_res_cells_az: int = 10,
    number_res_cells_rng: int = 10,
) -> np.ndarray:
    """Masking input data to select only side lobes (up to a given extent) and remove the rest (main lobe and furthest
    lobes) for PSLR computation.

    Parameters
    ----------
    data : np.ndarray
        input image 2d array
    mask_flag : MaskingMethod
        masking flag from one of MaskingMethod options
    resolution : tuple[float, float]
        resolution along range [0] and azimuth [1]
    peak_pos : tuple[int, int]
        peak position as max row [0] and max col [0] in input array
    side_lobes_directions : tuple[float, float]
        range and azimuth cuts angular coefficients in samples
    interp_factor : int, optional
        interpolation factor, by default 16
    number_res_cells_az : int, optional
        number of resolution cells to be taken for the mask size. this determines the extent of the mask and therefore
        the number of lobes selected in azimuth direction, by default 10
    number_res_cells_rng : int, optional
        number of resolution cells to be taken for the mask size. this determines the extent of the mask and therefore
        the number of lobes selected in range direction, by default 10

    Returns
    -------
    np.ndarray
        2D data mask
    """

    rng_res, az_res = resolution

    lobes_flag = bool(~np.isinf(side_lobes_directions[0]))
    # Compute pixel axes of the interpolated data
    a_x = (np.arange(0, data.shape[1]) - peak_pos[1]) / interp_factor
    a_y = (np.arange(0, data.shape[0]) - peak_pos[0]) / interp_factor

    # Compute data masks
    if mask_flag == MaskingMethod.PEAK:
        # Peak case
        if lobes_flag:
            _, _, main_lobe_mask = generate_peak_mask_lobes(data, side_lobes_directions)
        else:
            main_lobe_mask = generate_peak_mask(data)

    elif mask_flag == MaskingMethod.RESOLUTION:
        # Resolution case
        if lobes_flag:
            _, _, main_lobe_mask = generate_resolution_mask_lobes(
                x_axis=a_x,
                y_axis=a_y,
                res_x=az_res,
                res_y=rng_res,
                multiplier_x=2,
                multiplier_y=2,
                side_lobes_directions=side_lobes_directions,
            )
        else:
            main_lobe_mask = generate_resolution_mask(
                x_axis=a_x,
                y_axis=a_y,
                res_x=az_res,
                res_y=rng_res,
                multiplier_x=2,
                multiplier_y=2,
            )

    # generating a bigger mask centered on the main lobe extending up to a given number of resolution cells in both
    # directions, i.e. selecting only a given number of side lobes when it is applied to input data
    if lobes_flag:
        _, _, oversized_peak_mask = generate_resolution_mask_lobes(
            x_axis=a_x,
            y_axis=a_y,
            res_x=az_res,
            res_y=rng_res,
            multiplier_x=number_res_cells_az,
            multiplier_y=number_res_cells_rng,
            side_lobes_directions=side_lobes_directions,
        )
    else:
        oversized_peak_mask = generate_resolution_mask(
            x_axis=a_x,
            y_axis=a_y,
            res_x=az_res,
            res_y=rng_res,
            multiplier_x=number_res_cells_az,
            multiplier_y=number_res_cells_rng,
        )

    # subtracting masks
    side_lobes_mask = oversized_peak_mask - main_lobe_mask

    return side_lobes_mask


def pslr_profile_cutting(
    masked_data: np.ndarray,
    peak_pos: tuple[int, int],
    side_lobes_directions: tuple[float, float],
    interp_factor: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Extracting profile cuts both for range and azimuth from the masked input 2D array for PSLR computation.

    Parameters
    ----------
    masked_data : np.ndarray
        masked 2D array
    peak_pos : tuple[int, int]
        peak position as max row [0] and max col [0] in input array
    side_lobes_directions : tuple[float, float]
        range and azimuth cuts angular coefficients in samples
    interp_factor : int, optional
        interpolation factor, by default 16

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        range profile cut,
        azimuth profile cut
    """

    # Compute pixel axes of the interpolated data
    a_x = (np.arange(0, masked_data.shape[1]) - peak_pos[1]) / interp_factor
    a_y = (np.arange(0, masked_data.shape[0]) - peak_pos[0]) / interp_factor

    # extracting range and azimuth cuts
    if np.isinf(side_lobes_directions[0]):
        rng_cut = masked_data[:, peak_pos[1]]
        az_cut = masked_data[peak_pos[0], :]

    else:
        rng_cut, az_cut = get_interpolated_lobes_cuts(
            x_axis=a_x, y_axis=a_y, values=masked_data, side_lobes_directions=side_lobes_directions
        )

    return np.abs(rng_cut), np.abs(az_cut)


def islr_masking(
    data: np.ndarray,
    mask_flag: MaskingMethod,
    resolution: tuple[float, float],
    peak_pos: tuple[int, int],
    side_lobes_directions: tuple[float, float],
    interp_factor: int = 16,
    number_res_cells_az: int = 20,
    number_res_cells_rng: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Masking input data to extract only data relevant to ISLR computation. Two masks are built: one extracting just
    the main lobe up to the first zeroes, and a second one to isolate only side lobes up to a given extent in terms of
    resolution cells both in range and azimuth.

    Parameters
    ----------
    data : np.ndarray
        input 2D array
    mask_flag : MaskingMethod
        masking flag from one of MaskingMethod options
    resolution : tuple[float, float]
        resolution along range [0] and azimuth [1]
    peak_pos : tuple[int, int]
        peak position as max row [0] and max col [0] in input array
    side_lobes_directions : tuple[float, float]
        range and azimuth cuts angular coefficients in samples
    interp_factor : int, optional
        interpolation factor, by default 16
    number_res_cells_az : int, optional
        number of resolution cells to be taken for the mask size. this determines the extent of the mask and therefore
        the number of lobes selected in azimuth direction, by default 20
    number_res_cells_rng : int, optional
        number of resolution cells to be taken for the mask size. this determines the extent of the mask and therefore
        the number of lobes selected in range direction, by default 20

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        main lobe mask 2D array,
        side lobes mask 2D array
    """

    rng_res, az_res = resolution
    lobes_flag = bool(~np.isinf(side_lobes_directions[0]))

    # Compute pixel axes of the interpolated data
    a_x = (np.arange(0, data.shape[1]) - peak_pos[1]) / interp_factor
    a_y = (np.arange(0, data.shape[0]) - peak_pos[0]) / interp_factor

    # Compute data masks
    if mask_flag == MaskingMethod.PEAK:
        if lobes_flag:
            rng_idx, az_idx, main_lobe_mask = generate_peak_mask_lobes(data, side_lobes_directions)
            # determine the width and length of the main lobe masking portion in terms of resolution cells
            main_lobe_rng_size = (np.sum(rng_idx) - 1) / (rng_res * interp_factor)
            main_lobe_az_size = (np.sum(az_idx) - 1) / (az_res * interp_factor)
        else:
            main_lobe_mask = generate_peak_mask(data)
            # finding all columns with 1s and same for rows
            rows_filled = np.where(main_lobe_mask.any(axis=1))[0].size
            cols_filled = np.where(main_lobe_mask.any(axis=0))[0].size
            # determine the width and length of the main lobe masking portion in terms of resolution cells
            main_lobe_rng_size = np.squeeze((rows_filled - 1) / (rng_res * interp_factor))
            main_lobe_az_size = np.squeeze((cols_filled - 1) / (az_res * interp_factor))
            # generating strip masks along both directions covering side lobes from the first to the nth
            # nth is 20-dependent (20 resolution cells in that direction)

    elif mask_flag == MaskingMethod.RESOLUTION:
        # associating a default number of cell resolution corresponding to the main lobe width/length
        if lobes_flag:
            rng_idx, az_idx, main_lobe_mask = generate_resolution_mask_lobes(
                x_axis=a_x,
                y_axis=a_y,
                res_x=az_res,
                res_y=rng_res,
                multiplier_x=2,
                multiplier_y=2,
                side_lobes_directions=side_lobes_directions,
            )
            main_lobe_rng_size = (np.sum(rng_idx) - 1) / (rng_res * interp_factor)
            main_lobe_az_size = (np.sum(az_idx) - 1) / (az_res * interp_factor)
        else:
            main_lobe_mask = generate_resolution_mask(
                x_axis=a_x, y_axis=a_y, res_x=az_res, res_y=rng_res, multiplier_x=2, multiplier_y=2
            )
            main_lobe_rng_size = 2
            main_lobe_az_size = 2

    if lobes_flag:
        _, _, side_lobes_mask_az = generate_resolution_mask_lobes(
            x_axis=a_x,
            y_axis=a_y,
            res_x=az_res,
            res_y=rng_res,
            multiplier_x=number_res_cells_az,
            multiplier_y=main_lobe_rng_size,
            side_lobes_directions=side_lobes_directions,
        )
        _, _, side_lobes_mask_rng = generate_resolution_mask_lobes(
            x_axis=a_x,
            y_axis=a_y,
            res_x=az_res,
            res_y=rng_res,
            multiplier_x=main_lobe_az_size,
            multiplier_y=number_res_cells_rng,
            side_lobes_directions=side_lobes_directions,
        )
    else:
        side_lobes_mask_az = generate_resolution_mask(
            x_axis=a_x,
            y_axis=a_y,
            res_x=az_res,
            res_y=rng_res,
            multiplier_x=number_res_cells_az,
            multiplier_y=main_lobe_rng_size,
        )
        side_lobes_mask_rng = generate_resolution_mask(
            x_axis=a_x,
            y_axis=a_y,
            res_x=az_res,
            res_y=rng_res,
            multiplier_x=main_lobe_az_size,
            multiplier_y=number_res_cells_rng,
        )

    islr_mask = side_lobes_mask_az + side_lobes_mask_rng - main_lobe_mask * 2
    islr_mask[(islr_mask < 0) | (islr_mask > 1)] = 1

    return main_lobe_mask, islr_mask


def islr_profile_cutting(
    data: np.ndarray,
    main_lobe_mask: np.ndarray,
    islr_mask: np.ndarray,
    peak_pos: tuple[int, int],
    side_lobes_directions: tuple[float, float],
    interp_factor: int = 16,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Extracting profile cuts both for range and azimuth divided by main lobe and side lobe coverage after applying
    the masks to the input 2D array for ISLR computation.

    Parameters
    ----------
    data : np.ndarray
        input 2D array
    main_lobe_mask : np.ndarray
        main lobe boolean 2D mask
    islr_mask : np.ndarray
        islr boolean 2D mask
    peak_pos : tuple[int, int]
        peak position as max row [0] and max col [0] in input array
    side_lobes_directions : tuple[float, float]
        range and azimuth cuts angular coefficients in samples
    interp_factor : int, optional
        interpolation factor, by default 16

    Returns
    -------
    tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]
        main lobe range and azimuth cuts,
        side lobes range and azimuth cuts
    """

    # Compute pixel axes of the interpolated data
    a_x = (np.arange(0, data.shape[1]) - peak_pos[1]) / interp_factor
    a_y = (np.arange(0, data.shape[0]) - peak_pos[0]) / interp_factor

    if np.isinf(side_lobes_directions[0]):
        az_main_lobe_cut = data[peak_pos[0], :] * main_lobe_mask[peak_pos[0], :]
        rng_main_lobe_cut = data[:, peak_pos[1]] * main_lobe_mask[:, peak_pos[1]]

    else:
        rng_main_lobe_cut, az_main_lobe_cut = get_interpolated_lobes_cuts(
            x_axis=a_x, y_axis=a_y, values=data * main_lobe_mask, side_lobes_directions=side_lobes_directions
        )

    az_main_lobe_cut = np.abs(az_main_lobe_cut)
    rng_main_lobe_cut = np.abs(rng_main_lobe_cut)

    # storing peak cuts for export
    main_lobe_cuts = (rng_main_lobe_cut, az_main_lobe_cut)

    if np.isinf(side_lobes_directions[0]):
        az_side_lobes_cut = data[peak_pos[0], :] * islr_mask[peak_pos[0], :]
        rng_side_lobes_cut = data[:, peak_pos[1]] * islr_mask[:, peak_pos[1]]

    else:
        rng_side_lobes_cut, az_side_lobes_cut = get_interpolated_lobes_cuts(
            x_axis=a_x, y_axis=a_y, values=data * islr_mask, side_lobes_directions=side_lobes_directions
        )

    az_side_lobes_cut = np.abs(az_side_lobes_cut)
    rng_side_lobes_cut = np.abs(rng_side_lobes_cut)

    # storing side cuts for export
    side_lobes_cuts = (rng_side_lobes_cut, az_side_lobes_cut)

    return main_lobe_cuts, side_lobes_cuts


def sslr_masking(
    data: np.ndarray,
    resolution: tuple[float, float],
    peak_pos: tuple[int, int],
    side_lobes_directions: tuple[float, float],
    interp_factor: int = 16,
    number_res_cells_az: int = 10,
    number_res_cells_rng: int = 10,
) -> np.ndarray:
    """Masking input data 2D array to select only intermediate intensity lobes, removing most of the energy inside the
    image for SSLR computation.

    Parameters
    ----------
    data : np.ndarray
        input 2D array
    resolution : tuple[float, float]
        resolution along range [0] and azimuth [1]
    peak_pos : tuple[int, int]
        peak position as max row [0] and max col [0] in input array
    side_lobes_directions : np.ndarray
        range and azimuth cuts angular coefficients in samples
    interp_factor : int, optional
        interpolation factor, by default 16
    number_res_cells_az : int, optional
        number of resolution cells to be taken for the mask size. this determines the extent of the mask and therefore
        the number of lobes selected in azimuth direction, by default 10
    number_res_cells_rng : int, optional
        number of resolution cells to be taken for the mask size. this determines the extent of the mask and therefore
        the number of lobes selected in range direction, by default 10

    Returns
    -------
    np.ndarray
        2D intermediate lobes mask array
    """

    rng_res, az_res = resolution
    lobes_flag = bool(~np.isinf(side_lobes_directions[0]))

    # Compute pixel axes of the interpolated data
    a_x = (np.arange(0, data.shape[1]) - peak_pos[1]) / interp_factor
    a_y = (np.arange(0, data.shape[0]) - peak_pos[0]) / interp_factor

    # Compute mask centered on main lobe and extending a given number of resolution cells covering also higher side
    # lobes
    if lobes_flag:
        _, _, higher_intensity_mask = generate_resolution_mask_lobes(
            x_axis=a_x,
            y_axis=a_y,
            res_x=az_res,
            res_y=rng_res,
            multiplier_x=number_res_cells_az,
            multiplier_y=number_res_cells_rng,
            side_lobes_directions=side_lobes_directions,
        )
    else:
        higher_intensity_mask = generate_resolution_mask(
            x_axis=a_x,
            y_axis=a_y,
            res_x=az_res,
            res_y=rng_res,
            multiplier_x=number_res_cells_az,
            multiplier_y=number_res_cells_rng,
        )
    # oversized mask equal to the previous but bigger
    if lobes_flag:
        _, _, oversized_higher_intensity_mask = generate_resolution_mask_lobes(
            x_axis=a_x,
            y_axis=a_y,
            res_x=az_res,
            res_y=rng_res,
            multiplier_x=2 * number_res_cells_az,
            multiplier_y=2 * number_res_cells_rng,
            side_lobes_directions=side_lobes_directions,
        )
    else:
        oversized_higher_intensity_mask = generate_resolution_mask(
            x_axis=a_x,
            y_axis=a_y,
            res_x=az_res,
            res_y=rng_res,
            multiplier_x=2 * number_res_cells_az,
            multiplier_y=2 * number_res_cells_rng,
        )
    # difference between the two masks to take only the frame of intermediate intensity lobes around the main energy
    # in the image
    intermediate_intensity_lobes_mask = oversized_higher_intensity_mask - higher_intensity_mask

    return intermediate_intensity_lobes_mask


def sslr_profile_cutting(
    masked_data: np.ndarray,
    peak_pos: tuple[int, int],
    side_lobes_directions: tuple[float, float],
    interp_factor: int = 16,
) -> tuple[np.ndarray | float, np.ndarray | float]:
    """Extracting profile cuts both for range and azimuth from the masked input 2D array for SSLR computation.

    Parameters
    ----------
    masked_data : np.ndarray
        masked 2D array
    side_lobe_directions : np.ndarray
        range and azimuth cuts angular coefficients in samples
    peak_pos : tuple[int, int]
        peak position as max row [0] and max col [0] in input array
    interp_factor : int, optional
        interpolation factor, by default 16

    Returns
    -------
    tuple[np.ndarray | float, np.ndarray | float]
        range profile cut (or its max in case of side lobes),
        azimuth profile cut (or its max in case of side lobes),
    """

    # Compute pixel axes of the interpolated data
    a_x = (np.arange(0, masked_data.shape[1]) - peak_pos[1]) / interp_factor
    a_y = (np.arange(0, masked_data.shape[0]) - peak_pos[0]) / interp_factor

    if np.isinf(side_lobes_directions[0]):
        az_cut = np.abs(masked_data[peak_pos[0], :])
        rng_cut = np.abs(masked_data[:, peak_pos[1]])

    else:
        rng_cut, az_cut = get_interpolated_lobes_cuts(
            x_axis=a_x, y_axis=a_y, values=masked_data, side_lobes_directions=side_lobes_directions
        )
        rng_cut = np.max(np.abs(rng_cut))
        az_cut = np.max(np.abs(az_cut))

    return rng_cut, az_cut


def masking_outliers(
    data: np.ndarray,
    kernel_size: tuple[int, int] = (5, 5),
    filter_size: tuple[int, int] = (10, 10),
    percentile_boundaries: tuple[float, float] = (20, 90),
) -> np.ndarray:
    """Masking operation or removing outliers by specified percentile thresholds from input data.
    Values outside boundaries are set to nan.

    Parameters
    ----------
    data : np.ndarray
        input 2D array to be masked
    kernel_size : tuple[int, int], optional
        size of convolution kernel, by default (5, 5)
    filter_size : tuple[int, int], optional
        size of filter convolution kernel, by default (10, 10)
    percentile_boundaries : tuple[float, float], optional
        percentile boundaries above and below which data are to be removed, by default (20, 90)

    Returns
    -------
    np.ndarray
        masked 2D array containing nans where values have been removed
    """

    array = data.copy()
    # creating kernels
    kernel_array = np.ones(kernel_size) / np.prod(kernel_size)
    filter_kernel = np.ones(filter_size)

    # convolving data with first kernel
    data_conv = np.sqrt(convolve2d(data, kernel_array, mode="same"))
    data_conv = data_conv - np.tile(np.nanmedian(data_conv, 1), (data_conv.shape[1], 1)).T

    # masking data by percentiles
    masking_cond = np.logical_or(
        data_conv < np.percentile(data_conv.ravel(), percentile_boundaries[0]),
        data_conv > np.percentile(data_conv.ravel(), percentile_boundaries[1]),
    ).astype("int64")

    # convolving data with filter kernel
    mask = np.round(convolve2d(masking_cond, filter_kernel, mode="same") / np.sum(filter_kernel))

    # masking out data
    array[np.where(mask)] = np.nan

    return array
