# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Block-wise Radiometric Analysis module"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
from arepytools.geometry.geometric_functions import (
    compute_incidence_angles,
    compute_look_angles,
    compute_look_angles_from_trajectory,
)
from scipy.signal import convolve2d, medfilt2d

import arepyextras.quality.core.generic_dataclasses as gdt
import arepyextras.quality.radiometric_analysis.custom_dataclasses as rdt
from arepyextras.quality.core.signal_processing import (
    convert_to_db,
    radiometric_correction,
)
from arepyextras.quality.io.quality_input_protocol import QualityInputProduct
from arepyextras.quality.radiometric_analysis.config import (
    ProfileExtractionParameters,
    RadiometricProfilesConfig,
)
from arepyextras.quality.radiometric_analysis.support import (
    angles_computation_setup,
    blocks_definition,
    compute_2d_histogram,
    masking_outliers_by_percentiles,
)

# syncing with logger
log = logging.getLogger("quality_analysis")

# custom profile extractor callable type to be matched
RadiometricProfileExtractorType = Callable[[np.ndarray, ProfileExtractionParameters], np.ndarray]


def nesz_profiles(
    product: QualityInputProduct,
    output_quantity: gdt.SARRadiometricQuantity = gdt.SARRadiometricQuantity.SIGMA_NOUGHT,
    config: RadiometricProfilesConfig | None = None,
) -> list[rdt.RadiometricProfilesOutput]:
    """Noise Equivalent Sigma-Zero (NESZ) radiometric profiles computation. Profiles along RANGE direction.

    Parameters
    ----------
    product : QualityInputProduct
        object containing product information and data satisfying the QualityInputProduct protocol
    output_quantity : gdt.SARRadiometricQuantity, optional
        desired radiometric output quantity, by default gdt.SARRadiometricQuantity.SIGMA_NOUGHT
    config : RadiometricProfilesConfig | None, optional
        RadiometricProfiles configuration, by default None

    Returns
    -------
    list[rdt.RadiometricProfilesOutput]
        a RadiometricProfilesOutput dataclass for each channel
    """
    config = _nesz_config_manager(config=config)

    log.info(f"Performing NESZ Analysis on {product.name}")

    return radiometric_profiles(
        product=product,
        direction=rdt.RadiometricAnalysisDirection.RANGE,
        profile_extractor_func=_nesz_profiles_extractor,
        output_quantity=output_quantity,
        config=config,
    )


def gamma_profiles(
    product: QualityInputProduct,
    output_quantity: gdt.SARRadiometricQuantity = gdt.SARRadiometricQuantity.GAMMA_NOUGHT,
    config: RadiometricProfilesConfig | None = None,
) -> list[rdt.RadiometricProfilesOutput]:
    """Gamma radiometric profiles computation. Profiles along RANGE direction.

    Parameters
    ----------
    product : QualityInputProduct
        object containing product information and data satisfying the QualityInputProduct protocol
    output_quantity : gdt.SARRadiometricQuantity, optional
        desired radiometric output quantity, by default gdt.SARRadiometricQuantity.GAMMA_NOUGHT
    config : RadiometricProfilesConfig | None, optional
        RadiometricProfiles configuration, by default None

    Returns
    -------
    list[rdt.RadiometricProfilesOutput]
        a RadiometricProfilesOutput dataclass for each channel
    """
    config = _gamma_config_manager(config=config)

    log.info(f"Performing Gamma Profiles Analysis on {product.name}")

    return radiometric_profiles(
        product=product,
        direction=rdt.RadiometricAnalysisDirection.RANGE,
        profile_extractor_func=_gamma_profiles_extractor,
        output_quantity=output_quantity,
        config=config,
    )


def scalloping_profiles(
    product: QualityInputProduct,
    output_quantity: gdt.SARRadiometricQuantity = gdt.SARRadiometricQuantity.GAMMA_NOUGHT,
    config: RadiometricProfilesConfig | None = None,
) -> list[rdt.RadiometricProfilesOutput]:
    """Scalloping radiometric profiles computation. Profiles along AZIMUTH direction.

    Parameters
    ----------
    product : QualityInputProduct
        object containing product information and data satisfying the QualityInputProduct protocol
    output_quantity : gdt.SARRadiometricQuantity, optional
        desired radiometric output quantity, by default gdt.SARRadiometricQuantity.GAMMA_NOUGHT
    config : RadiometricProfilesConfig | None, optional
        RadiometricProfiles configuration, by default None

    Returns
    -------
    list[rdt.RadiometricProfilesOutput]
        a RadiometricProfilesOutput dataclass for each channel
    """
    config = _scalloping_config_manager(config=config)

    log.info(f"Performing Scalloping Profiles Analysis on {product.name}")

    return radiometric_profiles(
        product=product,
        direction=rdt.RadiometricAnalysisDirection.AZIMUTH,
        output_quantity=output_quantity,
        profile_extractor_func=_scalloping_profiles_extractor,
        config=config,
    )


def radiometric_profiles(
    product: QualityInputProduct,
    profile_extractor_func: RadiometricProfileExtractorType,
    direction: rdt.RadiometricAnalysisDirection = rdt.RadiometricAnalysisDirection.RANGE,
    output_quantity: gdt.SARRadiometricQuantity = gdt.SARRadiometricQuantity.GAMMA_NOUGHT,
    config: RadiometricProfilesConfig | None = None,
) -> list[rdt.RadiometricProfilesOutput]:
    """Block-wise Radiometric profiles computation.

    Parameters
    ----------
    product : QualityInputProduct
        object containing product information and data satisfying the QualityInputProduct protocol
    profile_extractor_func : RadiometricProfileExtractorType
        function to perform radiometric profile extraction
    direction : rdt.RadiometricAnalysisDirection, optional
        direction along which profiles are extracted, by default rdt.RadiometricAnalysisDirection.RANGE
    output_quantity : gdt.SARRadiometricQuantity, optional
        desired radiometric output quantity, by default gdt.SARRadiometricQuantity.GAMMA_NOUGHT
    config : RadiometricProfilesConfig | None, optional
        RadiometricProfiles configuration dataclass, by default None

    Returns
    -------
    list[rdt.RadiometricProfilesOutput]
        a RadiometricProfilesOutput dataclass for each channel
    """
    # managing inputs
    if config is None:
        config = RadiometricProfilesConfig()
    log.info("Performing radiometric analysis block-wise.")

    output_results = []
    for channel in product.channels_list:
        channel_data = product.get_channel_data(channel_id=channel)
        log.info(
            f"Analyzing channel {channel}, swath {channel_data.swath_name} and"
            + f"polarization {channel_data.polarization.value}..."
        )

        log.info("Defining blocks partitioning of the whole scene.")
        # defining scene partitioning by blocks
        az_block_size, blocks_num, blocks_centers_px = blocks_definition(
            azimuth_axis=channel_data.azimuth_axis,
            range_axis=channel_data.slant_range_axis,
            lines_per_burst=channel_data.lines_per_burst,
            default_block_size=config.azimuth_block_size,
        )

        if direction == rdt.RadiometricAnalysisDirection.RANGE:
            # creating axis for range direction
            look_angles_mid_swath = compute_look_angles_from_trajectory(
                trajectory=channel_data.trajectory,
                azimuth_time=channel_data.mid_azimuth_time,
                range_times=channel_data.slant_range_axis[config.range_pixel_margin : -config.range_pixel_margin],
                look_direction=channel_data.looking_side.value,
            )
            look_angles_mid_swath = np.rad2deg(look_angles_mid_swath)
            hist_axis = np.arange(look_angles_mid_swath[0] - 0.5, look_angles_mid_swath[-1] + 0.5, 0.01)

        elif direction == rdt.RadiometricAnalysisDirection.AZIMUTH:
            # creating axis for azimuth direction
            azimuth_rel_axis = channel_data.azimuth_axis - channel_data.azimuth_axis[0]
            hist_axis = np.arange(azimuth_rel_axis[10], azimuth_rel_axis[-10], 0.01)

        else:
            raise RuntimeError(f"{direction} invalid. It must be Range or Azimuth.")

        cropping_size = (
            channel_data.slant_range_axis.size - 2 * config.range_pixel_margin,  # range
            az_block_size,  # azimuth
        )

        profiles = []
        look_angles_array = []
        az_rel_times = []
        for bc_num, center in enumerate(blocks_centers_px):
            log.info(f"Processing block {bc_num + 1} of {blocks_num}")

            # reading block
            target_area = channel_data.read_data(
                azimuth_index=center[0],
                range_index=center[1],
                cropping_size=cropping_size,
            )
            # converting image to power
            target_area = np.abs(target_area) ** 2

            sensor_pos, ground_points, nadir = angles_computation_setup(
                trajectory=channel_data.trajectory,
                azimuth_time=channel_data.azimuth_axis[center[0]],
                range_values=channel_data.slant_range_axis[config.range_pixel_margin : -config.range_pixel_margin],
                look_direction=channel_data.looking_side.value,
            )

            if direction == rdt.RadiometricAnalysisDirection.RANGE:
                look_angles = compute_look_angles(
                    sensor_positions=sensor_pos, nadir_directions=nadir, points=ground_points
                )
                look_angles_array.append(np.rad2deg(look_angles))
            elif direction == rdt.RadiometricAnalysisDirection.AZIMUTH:
                az_axis_start_idx = center[0] - np.floor(az_block_size / 2).astype(int)
                az_block_axis = channel_data.azimuth_axis[az_axis_start_idx : az_axis_start_idx + az_block_size]
                az_rel_times.append(az_block_axis - channel_data.azimuth_axis[0])

            # performing radiometric correction, if needed
            if config.input_quantity != output_quantity:
                log.info(
                    f"Converting data from {config.input_quantity.name.lower()} to {output_quantity.name.lower()}."
                )
                incidence_angle_mid_block = compute_incidence_angles(sensor_positions=sensor_pos, points=ground_points)
                target_area = radiometric_correction(
                    data=target_area,
                    incidence_angle=incidence_angle_mid_block,
                    input_quantity=config.input_quantity,
                    output_quantity=output_quantity,
                    exp_power=config.radiometric_correction_exponent,
                )

            # applying provided profile extraction function
            log.debug("Extracting profiles.")
            profiles.append(profile_extractor_func(target_area, config.profile_extraction_parameters))

        # 2D histogram
        profiles = np.ma.stack(profiles)
        look_angles_array = np.vstack(look_angles_array) if look_angles_array else None
        az_rel_times = np.vstack(az_rel_times).astype(float) if az_rel_times else None
        hist, x_bins, y_bins = compute_2d_histogram(
            x_data=look_angles_array if look_angles_array is not None else az_rel_times,
            y_data=profiles,
            x_axis=hist_axis,
            config=config.histogram_parameters,
        )

        # storing results
        output_results.append(
            rdt.RadiometricProfilesOutput(
                swath=channel_data.swath_name,
                channel=channel,
                polarization=channel_data.polarization,
                direction=direction,
                output_radiometric_quantity=output_quantity,
                azimuth_start_time=channel_data.azimuth_axis[0],
                azimuth_block_centers=channel_data.azimuth_axis[[t[0] for t in blocks_centers_px]],
                range_block_centers=channel_data.slant_range_axis[[t[1] for t in blocks_centers_px]],
                blocks_num=blocks_num,
                profiles=profiles,
                block_azimuth_times=az_rel_times,
                look_angles=look_angles_array,
                hist_2d=hist,
                hist_x_bins_axis=x_bins,
                hist_y_bins_axis=y_bins,
            )
        )

    return output_results


def _nesz_profiles_extractor(data: np.ndarray, params: ProfileExtractionParameters) -> np.ndarray:
    """Profiles extraction function for NESZ analysis.

    Parameters
    ----------
    data : np.ndarray
        2D target block to be processed
    params : ProfileExtractionParameters
        radiometric profiles configuration

    Returns
    -------
    np.ndarray
        nesz profile
    """
    # azimuth profile as a sum over range
    current_block_az_profile = np.nansum(data, axis=0)

    # if more than half of the block is populated with zeroes, discard the whole block
    if np.count_nonzero(current_block_az_profile) / current_block_az_profile.size < 0.5:
        # TODO check if this exit strategy is ok
        return np.full(data.shape[0], np.nan)

    # keeping only data where the current_block_az_profile is not 0
    data = data[:, ~(current_block_az_profile == 0)]
    #!!! NAN are an issue when using open cv filter2d
    data[data == 0] = np.nan

    kernel = np.ones((params.filtering_kernel_size[0], params.filtering_kernel_size[1])) / (
        params.filtering_kernel_size[0] * params.filtering_kernel_size[1]
    )
    # TODO check if speed up is much needed, if so this opencv snippets halves computation time (check values equality)
    #!!! the issue with that is that there must be no NAN values in the original array (replace them with 0?)
    # data = np.ascontiguousarray(data)
    # data2 = cv2.filter2D(
    #     src=data,
    #     ddepth=-1,
    #     kernel=kernel,
    #     borderType=cv2.BORDER_CONSTANT,
    # )

    # TODO check if mode "valid" is absolutely necessary ("same" avoid reducing the profile length)
    # performing multi-looking 2D convolution (moving average 2D)
    data = convolve2d(
        data,
        kernel,
        mode="same",
    )
    profile_db = convert_to_db(np.nanpercentile(abs(data), 1, axis=1))
    return np.ma.masked_invalid(profile_db)


def _gamma_profiles_extractor(data: np.ndarray, params: ProfileExtractionParameters) -> np.ndarray:
    """Profiles extraction function for Gamma analysis.

    Parameters
    ----------
    data : np.ndarray
        2D target block to be processed
    params : ProfileExtractionParameters
        radiometric profiles configuration

    Returns
    -------
    np.ndarray
        gamma profile
    """
    if params.smoothening_filter:
        log.info("Applying smoothening median filter...")
        data = medfilt2d(data, params.filtering_kernel_size)

    if params.outlier_removal:
        log.info("Masking outliers...")
        # masking data by percentiles
        data = masking_outliers_by_percentiles(
            data=data,
            kernel=params.outliers_kernel_size,
            percentile_boundaries=params.outliers_percentile_boundaries,
        )

    profile_db = convert_to_db(np.nanmean(data, 1))
    return np.ma.masked_invalid(profile_db)


def _scalloping_profiles_extractor(data: np.ndarray, params: ProfileExtractionParameters) -> np.ndarray:
    """Profiles extraction function for Scalloping analysis.

    Parameters
    ----------
    data : np.ndarray
        2D target block to be processed
    params : ProfileExtractionParameters
        radiometric profiles configuration

    Returns
    -------
    np.ndarray
        scalloping profile
    """
    if params.outlier_removal:
        log.info("Masking outliers...")
        data = masking_outliers_by_percentiles(
            data=data, kernel=params.outliers_kernel_size, percentile_boundaries=params.outliers_percentile_boundaries
        )

    azimuth_profile = np.nanmean(data, axis=0)
    azimuth_profile = azimuth_profile / np.nanmean(azimuth_profile)

    profile_db = convert_to_db(azimuth_profile)
    return np.ma.masked_invalid(profile_db)


def _nesz_config_manager(config: RadiometricProfilesConfig | None) -> RadiometricProfilesConfig:
    """Initializing default NESZ Profiles config if None is provided and check that histogram parameters are not None.

    Parameters
    ----------
    config : RadiometricProfilesConfig | None
        input NESZ configuration

    Returns
    -------
    RadiometricProfilesConfig
        updated/default/checked NESZ configuration
    """
    if config is None:
        log.info("Configuration not provided. Using default NESZ Profiles configuration...")
        config = RadiometricProfilesConfig()

    if config.profile_extraction_parameters.filtering_kernel_size is None:
        config.profile_extraction_parameters.filtering_kernel_size = (7, 7)

    if config.histogram_parameters.x_bins_step is None:
        config.histogram_parameters.x_bins_step = 5

    if config.histogram_parameters.y_bins_num is None:
        config.histogram_parameters.y_bins_num = 301

    if config.histogram_parameters.y_bins_center_margin is None:
        config.histogram_parameters.y_bins_center_margin = 20

    return config


def _gamma_config_manager(config: RadiometricProfilesConfig | None) -> RadiometricProfilesConfig:
    """Initializing default Gamma Profiles config if None is provided and check that histogram parameters are not None.

    Parameters
    ----------
    config : RadiometricProfilesConfig | None
        input Gamma configuration

    Returns
    -------
    RadiometricProfilesConfig
        updated/default/checked Gamma configuration
    """
    if config is None:
        log.info("Configuration not provided. Using default Gamma Profiles configuration...")
        config = RadiometricProfilesConfig()

    if config.profile_extraction_parameters.filtering_kernel_size is None:
        config.profile_extraction_parameters.filtering_kernel_size = (11, 11)

    if config.histogram_parameters.x_bins_step is None:
        config.histogram_parameters.x_bins_step = 5

    if config.histogram_parameters.y_bins_num is None:
        config.histogram_parameters.y_bins_num = 101

    if config.histogram_parameters.y_bins_center_margin is None:
        config.histogram_parameters.y_bins_center_margin = 3

    return config


def _scalloping_config_manager(config: RadiometricProfilesConfig | None) -> RadiometricProfilesConfig:
    """Initializing default Scalloping Profiles config if None is provided and check that histogram parameters are not None.

    Parameters
    ----------
    config : RadiometricProfilesConfig | None
        input Scalloping configuration

    Returns
    -------
    RadiometricProfilesConfig
        updated/default/checked Scalloping configuration
    """
    if config is None:
        log.info("Configuration not provided. Using default Scalloping Profiles configuration...")
        config = RadiometricProfilesConfig()

    if config.profile_extraction_parameters.filtering_kernel_size is None:
        config.profile_extraction_parameters.filtering_kernel_size = (11, 11)

    if config.histogram_parameters.x_bins_step is None:
        config.histogram_parameters.x_bins_step = 3

    if config.histogram_parameters.y_bins_num is None:
        config.histogram_parameters.y_bins_num = 51

    if config.histogram_parameters.y_bins_center_margin is None:
        config.histogram_parameters.y_bins_center_margin = 1.5

    return config
