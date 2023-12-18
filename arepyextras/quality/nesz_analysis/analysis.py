# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
Noise Equivalent Sigma Zero (NESZ) computation
----------------------------------------------
"""
import logging
from pathlib import Path
from typing import Union

import numpy as np
from arepytools.geometry.conversions import llh2xyz, xyz2llh
from arepytools.geometry.direct_geocoding import direct_geocoding_monostatic
from netCDF4 import Dataset
from scipy.signal import convolve2d

from arepyextras.quality.core.signal_processing import convert_to_db
from arepyextras.quality.io.quality_input_from_product_folder import (
    ProductFolderManager,
)
from arepyextras.quality.io.quality_input_protocol import QualityInputProduct
from arepyextras.quality.nesz_analysis.custom_dataclasses import NESZConfig, NESZOutput

# syncing with logger
log = logging.getLogger("quality_analysis")


def nesz_productfolder_wrapper(
    pf_path: Union[str, Path],
    config: NESZConfig = None,
) -> list[NESZOutput]:
    """NESZ analysis wrapper for Aresys Product Folder input type.

    Parameters
    ----------
    pf_path : Union[str, Path]
        Path to product folder
    noise_path : Union[str, Path]
        Path to noise product
    config : NESZConfig, optional
        NESZ analysis configuration, by default None

    Returns
    -------
    list[NESZOutput]
        list of NESZOutput dataclasses, one for each channel
    """
    pf_path = Path(pf_path)

    product = ProductFolderManager(pf_path)

    return nesz_analysis(product=product, config=config)


def nesz_analysis(product: QualityInputProduct, config: NESZConfig = None) -> list[NESZOutput]:
    """Noise Equivalent Sigma Zero (NESZ) analysis on input product and noise data.

    Parameters
    ----------
    product : QualityInputProduct
        input product compliant with QualityInputProduct protocol
    config : NESZConfig, optional
        NESZ configuration, if None the default configuration is used, by default None

    Returns
    -------
    list[NESZOutput]
        list of NESZOutput dataclasses, one for each channel
    """
    if config is None:
        log.info("Configuration file not provided, using default")
        config = NESZConfig()

    log.info(f"Starting NESZ Analysis on {product.name}")
    log.info(f"Selected Product has {len(product.channels_list)} channels")

    nesz_channel_output = []
    for channel in product.channels_list:
        channel_data = product.get_channel_data(channel_id=channel)

        log.info(f"Processing channel {channel}, polarization {channel_data.polarization.name}")
        # get sensor position and velocity at reference time
        samples_padding = int(config.pixel_margin + (config.rng_multilook_length - 1) / 2)
        samples_axis_redux = channel_data.slant_range_axis[samples_padding:-samples_padding]
        ref_time = channel_data.mid_azimuth_time
        sensor_pos = channel_data.trajectory.evaluate(ref_time)
        sensor_vel = channel_data.trajectory.evaluate_first_derivatives(ref_time)
        ref_nadir_norm = _compute_norm_nadir_from_sensor_pos(sensor_pos)

        ground_points_rng_edges = direct_geocoding_monostatic(
            sensor_positions=sensor_pos,
            sensor_velocities=sensor_vel,
            range_times=[channel_data.slant_range_axis[0], channel_data.slant_range_axis[-1]],
            geocoding_side=channel_data.looking_side.value,
            frequencies_doppler_centroid=0,
            wavelength=1,
            geodetic_altitude=0,
        )
        near_rng_los = ground_points_rng_edges[0] - sensor_pos
        far_rng_los = ground_points_rng_edges[1] - sensor_pos

        near_look = (
            np.rad2deg(np.arccos(np.dot(ref_nadir_norm, near_rng_los / np.linalg.norm(near_rng_los))))
            - config.look_angle_margin
        )
        far_look = (
            np.rad2deg(np.arccos(np.dot(ref_nadir_norm, far_rng_los / np.linalg.norm(far_rng_los))))
            + config.look_angle_margin
        )
        nesz_look_axis = np.arange(near_look, far_look, config.look_angle_step)

        az_block_size = config.az_block_size
        blocks_num = int(np.floor(channel_data.azimuth_axis.size / config.az_block_size))

        # TOPSAR/SCANSAR case
        block_center_flag = False
        if channel_data.lines_per_burst.size > 1:
            az_block_size = channel_data.lines_per_burst[0]
            blocks_num = channel_data.lines_per_burst.size
            block_center_flag = True

        elevation_angles = []
        profiles = []
        for block in range(blocks_num):
            log.info(f"Processing block {block + 1} of {blocks_num}")

            # reading data
            rng_samples_to_be_read = channel_data.slant_range_axis.size - 2 * config.pixel_margin
            data = channel_data.read_data(
                azimuth_index=az_block_size * block + az_block_size // 2,
                range_index=config.pixel_margin + rng_samples_to_be_read // 2,
                cropping_size=(rng_samples_to_be_read, az_block_size),
            )

            if block_center_flag:
                rng_crop = np.round(az_block_size / 2).astype(int)
                pad = config.burst_center_block_size // 2
                data = data[:, rng_crop - pad : rng_crop + pad]

            data = np.abs(data) ** 2

            # azimuth profile as a sum over range
            current_block_az_profile = np.nansum(data, axis=0)

            # if more than half of the block is populated with zeroes, discard the whole block
            if np.count_nonzero(current_block_az_profile) / current_block_az_profile.size < 0.5:
                continue

            # keeping only data where the current_block_az_profile is not 0
            data = data[:, ~(current_block_az_profile == 0)]
            data[data == 0] = np.nan

            # performing multi-looking 2D convolution
            data = convolve2d(
                data,
                np.ones((config.rng_multilook_length, config.az_multilook_length))
                / (config.rng_multilook_length * config.az_multilook_length),
                mode="valid",
            )

            # compute burst geometry
            half_block_az_time = channel_data.azimuth_axis[
                az_block_size * block + np.round(az_block_size / 2).astype(int)
            ]
            half_block_sensor_pos = channel_data.trajectory.evaluate(half_block_az_time)
            half_block_sensor_vel = channel_data.trajectory.evaluate_first_derivatives(half_block_az_time)
            half_block_nadir_norm = _compute_norm_nadir_from_sensor_pos(half_block_sensor_pos)

            # compute elevation and incidence angles
            ground_points = direct_geocoding_monostatic(
                sensor_positions=half_block_sensor_pos,
                sensor_velocities=half_block_sensor_vel,
                range_times=samples_axis_redux,
                geocoding_side=channel_data.looking_side.value,
                frequencies_doppler_centroid=0,
                wavelength=1,
                geodetic_altitude=0,
            )
            ground_points_norm = ground_points / np.linalg.norm(ground_points, axis=1, keepdims=True)
            samples_axis_los = ground_points - half_block_sensor_pos
            samples_axis_los_norm = samples_axis_los / np.linalg.norm(samples_axis_los, axis=1, keepdims=True)
            elevation_angles.append(np.arccos(np.sum(half_block_nadir_norm * samples_axis_los_norm, axis=1)))
            incidence_angle = np.arccos(np.sum(-samples_axis_los_norm * ground_points_norm, axis=1))

            if config.incidence_compensation:
                # compensation not performed during processing (beta instead of sigma)
                data = data * np.sin(incidence_angle).reshape(-1, 1)

            profiles.append(np.nanpercentile(abs(data).T, 1, axis=0))

        # storing results
        nesz_channel_output.append(
            NESZOutput(
                channel=channel,
                swath=channel_data.swath_name,
                polarization=channel_data.polarization,
                azimuth_blocks_num=blocks_num,
                elevation_angles_deg=np.rad2deg(np.stack(elevation_angles, axis=1)),
                nesz_profiles=np.nan_to_num(np.stack(profiles, axis=1)),
                axis_deg=nesz_look_axis,
            )
        )

    log.info("NESZ analysis completed.")
    return nesz_channel_output


def _compute_norm_nadir_from_sensor_pos(sensor_pos: np.ndarray) -> np.ndarray:
    """Computing Nadir value from sensor position.

    Parameters
    ----------
    sensor_pos : np.ndarray
        sensor position, in the form (3,)

    Returns
    -------
    np.ndarray
        normalized sensor nadir
    """
    sensor_pos_llh = xyz2llh(sensor_pos).squeeze()
    ref_nadir = llh2xyz([sensor_pos_llh[0], sensor_pos_llh[1], 0]).squeeze() - sensor_pos
    return ref_nadir / np.linalg.norm(ref_nadir)


def save_to_netcdf(data: list[NESZOutput], out_path: Union[str, Path]) -> None:
    """Saving NESZ output data to a NetCDF4 file.

    Parameters
    ----------
    data : list[NESZOutput]
        list of NESZOutput dataclasses
    out_path : Union[str, Path]
        path where to save the NetCDF file
    """
    out_path = Path(out_path)

    for item in data:
        out_name = "nesz_profiles_" + item.swath + "_" + item.polarization.name
        log.info(f"Saving {out_name} data to NetCDF file.")

        root = Dataset(out_path.joinpath(out_name).with_suffix(".nc"), "w", format="NETCDF4")
        root.swath = item.swath
        root.channel = item.channel
        root.polarization = item.polarization.name
        root.azimuth_blocks_num = item.azimuth_blocks_num

        # creating common dimensions
        root.createDimension("samples", item.elevation_angles_deg.shape[0])
        root.createDimension("azimuth_blocks", item.elevation_angles_deg.shape[1])

        # creating elevation angles variable
        ele_angles = root.createVariable(
            "elevation_angles", item.elevation_angles_deg.dtype, ("samples", "azimuth_blocks")
        )
        ele_angles.unit = "deg"
        ele_angles[:] = item.elevation_angles_deg

        # # creating incidence angles variable
        # inc_angles = root.createVariable(
        #     "incidence_angles", item.incidence_angles_deg.dtype, ("samples", "azimuth_blocks")
        # )
        # inc_angles.unit = "deg"
        # inc_angles[:] = item.incidence_angles_deg

        # creating nesz profile variable
        nesz_profs = root.createVariable("nesz_profiles", item.nesz_profiles.dtype, ("samples", "azimuth_blocks"))
        nesz_profs.unit = "dB"
        nesz_profs[:] = convert_to_db(item.nesz_profiles)

        root.close()
