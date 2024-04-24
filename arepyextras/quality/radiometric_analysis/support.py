# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Radiometric Analysis support functionalities module"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import numpy as np
from arepytools.geometry.conversions import llh2xyz, xyz2llh
from arepytools.geometry.curve_protocols import TwiceDifferentiable3DCurve
from arepytools.geometry.direct_geocoding import (
    GeocodingSide,
    direct_geocoding_monostatic,
)
from arepytools.timing.precisedatetime import PreciseDateTime
from netCDF4 import Dataset
from scipy.signal import convolve2d

from arepyextras.quality.radiometric_analysis.config import (
    Radiometric2DHistogramParameters,
)
from arepyextras.quality.radiometric_analysis.custom_dataclasses import (
    RadiometricProfilesOutput,
)

# syncing with logger
log = logging.getLogger("quality_analysis")


def radiometric_profiles_to_netcdf(
    data: RadiometricProfilesOutput, out_path: Union[str, Path], tag: str | None = None
) -> None:
    """Saving Radiometric Profiles output data to NetCDF4 file.

    Parameters
    ----------
    data : RadiometricProfilesOutput
        RadiometricProfilesOutput dataclass
    out_path : Union[str, Path]
        path where to save the NetCDF file
    tag : str | None, optional
        tag string to be added to the output filename, by default None
    """
    out_path = Path(out_path)
    tag = "radiometric" if tag is None else tag

    out_name = tag + "_profiles_" + data.swath + "_" + data.polarization.name
    log.info(f"Saving {out_name} data to NetCDF file.")

    root = Dataset(out_path.joinpath(out_name).with_suffix(".nc"), "w", format="NETCDF4")
    root.swath = data.swath
    root.channel = data.channel
    root.polarization = data.polarization.name
    root.direction = data.direction.name.lower()
    root.output_radiometric_quantity = data.output_radiometric_quantity.name
    root.azimuth_blocks_num = data.blocks_num
    root.azimuth_block_centers = [str(d) for d in data.azimuth_block_centers]
    root.range_block_centers = data.range_block_centers

    # creating common dimensions
    root.createDimension("samples", data.profiles.shape[1])
    root.createDimension("azimuth_blocks", data.blocks_num)

    # creating elevation angles variable
    if data.look_angles is not None:
        data_axis = root.createVariable("look_angles", data.look_angles.dtype, ("azimuth_blocks", "samples"))
        data_axis.unit = "deg"
        data_axis[:] = data.look_angles

    if data.block_azimuth_times is not None:
        data_axis = root.createVariable("azimuth_times", data.block_azimuth_times.dtype, ("azimuth_blocks", "samples"))
        data_axis.unit = "s"
        data_axis[:] = data.block_azimuth_times

    # creating nesz profile variable
    profs = root.createVariable("radiometric_profiles", data.profiles.dtype, ("azimuth_blocks", "samples"))
    profs.unit = "dB"
    profs[:] = data.profiles

    root.close()


def angles_computation_setup(
    trajectory: TwiceDifferentiable3DCurve,
    azimuth_time: PreciseDateTime,
    range_values: np.ndarray,
    look_direction: Union[str, GeocodingSide],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Setting up the stage to compute incidence and look angles by computing sensor position, ground points and nadir
    direction.

    Parameters
    ----------
    trajectory : TwiceDifferentiable3DCurve
        sensor trajectory
    azimuth_time : PreciseDateTime
        azimuth time at which compute the output
    range_values : np.ndarray
        range values for which compute values
    look_direction : Union[str, GeocodingSide]
        sensor look direction

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        sensor position,
        ground points,
        nadir direction
    """
    look_direction = GeocodingSide(look_direction)
    sensor_pos = trajectory.evaluate(azimuth_time)
    sensor_vel = trajectory.evaluate_first_derivatives(azimuth_time)

    ground_points = direct_geocoding_monostatic(
        sensor_positions=sensor_pos,
        sensor_velocities=sensor_vel,
        range_times=range_values,
        geocoding_side=look_direction.value,
        frequencies_doppler_centroid=0,
        wavelength=1,
        geodetic_altitude=0,
    )

    sensor_position_ground = xyz2llh(sensor_pos)
    sensor_position_ground[2] = 0.0
    sensor_position_ground = llh2xyz(sensor_position_ground).squeeze()

    nadir = sensor_position_ground - sensor_pos
    return sensor_pos, ground_points, nadir


def blocks_definition(
    azimuth_axis: np.ndarray,
    range_axis: np.ndarray,
    lines_per_burst: np.ndarray,
    default_block_size: int,
) -> tuple[int, int, list[tuple[int, int]]]:
    """Defining the blocks partitioning of the whole scene.

    Parameters
    ----------
    azimuth_axis : np.ndarray
        azimuth axis of the whole scene
    range_axis : np.ndarray
        range axis of the whole scene
    lines_per_burst : np.ndarray
        lines per burst array
    default_block_size : int
        default block size value, needed for stripmap case

    Returns
    -------
    tuple[int, int, list[tuple[int, int]]]
        size of each block,
        number of partitioning blocks,
        pixel coordinates of blocks centers (azimuth and range pixel values)
    """
    block_size = default_block_size
    blocks_num = int(np.floor(azimuth_axis.size / block_size))
    mid_range_pixel = int(range_axis.size // 2)

    if lines_per_burst.size > 1:
        # TOPSAR/SCANSAR case: blocks set using bursts
        block_size = lines_per_burst[0]
        blocks_num = lines_per_burst.size  # number of bursts

    blocks_centers_px = np.arange(block_size // 2, block_size * blocks_num, block_size).tolist()
    blocks_centers_px = [(px, mid_range_pixel) for px in blocks_centers_px]

    return block_size, blocks_num, blocks_centers_px


def compute_2d_histogram(
    x_data: np.ndarray, y_data: np.ndarray, x_axis: np.ndarray, config: Radiometric2DHistogramParameters
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute 2D histogram from input data.

    Parameters
    ----------
    x_data : np.ndarray
        data along the selected x axis
    y_data : np.ndarray
        data along the selected y axis
    x_axis : np.ndarray
        histogram x axis
    config : Radiometric2DHistogramParameters
        configuration parameters for the 2D histogram

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        2D histogram,
        x bins axis,
        y bins axis
    """

    assert config.y_bins_center_margin is not None
    assert config.y_bins_num is not None
    assert config.x_bins_step is not None

    # bins axis generation
    y_bins_center = np.nanmean(y_data)

    y_bins = np.linspace(
        start=y_bins_center - config.y_bins_center_margin,
        stop=y_bins_center + config.y_bins_center_margin,
        num=config.y_bins_num,
    )
    x_bins = x_axis[:: config.x_bins_step]

    # 2D histogram generation
    hist, _, _ = np.histogram2d(
        x=x_data.ravel(),
        y=y_data.ravel(),
        bins=[x_bins, y_bins],
    )
    hist = hist.T

    return hist, x_bins, y_bins


def masking_outliers_by_percentiles(
    data: np.ndarray, kernel: tuple[int, int], percentile_boundaries: tuple[int, int]
) -> np.ndarray:
    """Masking outliers outside of provided percentile boundaries setting them to NaN.

    Parameters
    ----------
    data : np.ndarray
        input 2D array
    kernel : tuple[int, int]
        kernel size, height and width in pixels
    percentile_boundaries : tuple[int, int]
        data below percentile_boundaries[0] and above percentile_boundaries[1] are set to NaN

    Returns
    -------
    np.ndarray
        input array with NaN where outliers lie
    """
    filter_kernel = np.ones(kernel)
    masking_cond = np.logical_or(
        data < np.nanpercentile(data.ravel(), percentile_boundaries[0]),
        data > np.nanpercentile(data.ravel(), percentile_boundaries[1]),
    ).astype("int64")

    # convolving data with filter kernel
    mask = np.round(convolve2d(masking_cond, filter_kernel, mode="same") / np.sum(filter_kernel))

    # masking out data
    data[np.where(mask)] = np.nan

    return data
