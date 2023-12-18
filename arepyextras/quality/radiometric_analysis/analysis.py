# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Perform Radiometric Analysis on input Product Folder at requested times"""

import logging
from pathlib import Path
from typing import Union

import numpy as np
from arepytools.geometry.geometric_functions import (
    compute_incidence_angles_from_trajectory,
    compute_look_angles_from_trajectory,
)
from arepytools.timing.precisedatetime import PreciseDateTime
from scipy.signal import savgol_filter

import arepyextras.quality.core.generic_dataclasses as gdt
import arepyextras.quality.radiometric_analysis.custom_dataclasses as rdt
import arepyextras.quality.radiometric_analysis.custom_errors as ra_err
from arepyextras.quality.core.custom_errors import CoordinatesOutOfBounds
from arepyextras.quality.core.generic_dataclasses import SARPolarization, SARProjection
from arepyextras.quality.core.masking_operations import masking_outliers
from arepyextras.quality.core.signal_processing import (
    convert_to_db,
    radiometric_correction,
)
from arepyextras.quality.io.quality_input_from_product_folder import (
    ProductFolderManager,
)
from arepyextras.quality.io.quality_input_protocol import (
    ChannelData,
    QualityInputProduct,
)

# syncing with logger
log = logging.getLogger("quality_analysis")


def radiometric_productfolder_wrapper(
    pf_path: Union[str, Path],
    azimuth_times: list[Union[PreciseDateTime, int]] = None,
    range_times: list[Union[float, int]] = None,
    swath_name: str = None,
    selected_polarization: SARPolarization = None,
    is_pixel: bool = False,
    analysis_config: rdt.RadiometricAnalysisConfig = None,
) -> tuple[list[rdt.RadiometricAnalysisOutput], gdt.SARProjection]:
    """Radiometric Analysis wrapper designed for aresys product folder format.

    Parameters
    ----------
    pf_path : Union[str, Path]
        path to the product folder of choice
    azimuth_times : list[Union[PreciseDateTime, int]], optional
        azimuth times where to perform the radiometric analysis [performed along range direction], can be expressed as
        a list of times (PreciseDateTime) or pixel indexes (int), by default None
    range_times : list[Union[float, int]], optional
        range times where to perform the radiometric analysis [performed along azimuth direction], can be expresses as
        a list of times (float) or pixel indexes (int), by default None
    swath_name : str, optional
        if product has multiple swaths, the one to be analyzed should be specified, by default None
    selected_polarization : SARPolarization, optional
        if input product has more than one polarization, the one to be analyzed should be specified, by default None
    is_pixel : bool, optional
        boolean flag to inform that the input azimuth times and range times values are expressed as pixels and not as
        real times in the swath, by default False
    analysis_config : rdt.RadiometricAnalysisConfig, optional
        configuration dataclass to manage all the different parameters and enabling flags, by default None

    Returns
    -------
    tuple[list[rdt.RadiometricAnalysisOutput], gdt.SARProjection]
        list of RadiometricAnalysisOutput dataclasses for graph plotting
        data projection (slant range or ground range)
    """
    # loading product folder
    product = ProductFolderManager(path=pf_path)
    log.info(f"Selected Product Folder has {len(product.channels_list)} channels.")

    return radiometric_analysis(
        product=product,
        azimuth_times=azimuth_times,
        range_times=range_times,
        swath_name=swath_name,
        selected_polarization=selected_polarization,
        is_pixel=is_pixel,
        analysis_config=analysis_config,
    )


def radiometric_analysis(
    product: QualityInputProduct,
    azimuth_times: list[Union[PreciseDateTime, int]] = None,
    range_times: list[Union[float, int]] = None,
    swath_name: str = None,
    selected_polarization: Union[str, SARPolarization] = None,
    is_pixel: bool = False,
    analysis_config: rdt.RadiometricAnalysisConfig = None,
) -> tuple[list[rdt.RadiometricAnalysisOutput], gdt.SARProjection]:
    """Radiometric Analysis wrapper designed for aresys product folder format.

    Parameters
    ----------
    product : QualityInputProduct
        object containing product information and data satisfying the QualityInputProduct protocol
    azimuth_times : list[Union[PreciseDateTime, int]], optional
        azimuth times where to perform the radiometric analysis [performed along range direction], can be expressed as
        a list of times (PreciseDateTime) or pixel indexes (int), by default None
    range_times : list[Union[float, int]], optional
        range times where to perform the radiometric analysis [performed along azimuth direction], can be expresses as
        a list of times (float) or pixel indexes (int), by default None
    swath_name : str, optional
        if product has multiple swaths, the one to be analyzed should be specified, by default None
    selected_polarization : SARPolarization, optional
        if input product has more than one polarization, the one to be analyzed should be specified, by default None
    is_pixel : bool, optional
        boolean flag to inform that the input azimuth times and range times values are expressed as pixels and not as
        real times in the swath, by default False
    analysis_config : rdt.RadiometricAnalysisConfig, optional
        configuration dataclass to manage all the different parameters and enabling flags, by default None

    Returns
    -------
    tuple[list[rdt.RadiometricAnalysisOutput], gdt.SARProjection]
        list of RadiometricAnalysisOutput dataclasses for graph plotting
        data projection (slant range or ground range)
    """
    # managing inputs
    if analysis_config is None:
        analysis_config = rdt.RadiometricAnalysisConfig()
    log.info(f"Performing radiometric analysis along {analysis_config.direction.name} direction(s).")

    selected_polarization = SARPolarization(selected_polarization) if selected_polarization is not None else None

    if azimuth_times is None and range_times is None:
        raise ra_err.InputMissingError("No times/pixel coordinates provided")

    if azimuth_times is not None:
        azimuth_times = np.asarray(azimuth_times)
    if range_times is not None:
        range_times = np.asarray(range_times)

    # initializing range and/or azimuth pixel arrays
    azimuth_times_px, range_times_px = _init_pixel_arrays(
        azimuth_times=azimuth_times, range_times=range_times, is_pixel=is_pixel
    )

    log.info(f"Input data is {analysis_config.input_type.name.capitalize()}")
    log.info(f"Requested output is {analysis_config.output_type.name.capitalize()}")

    # check number of swaths
    azimuth_avg_times = None
    azimuth_axes, range_axes = [], []
    polarizations, swaths = set(), set()
    for channel in product.channels_list:
        # recovering metadata for the current channel
        channel_data = product.get_channel_data(channel_id=channel)
        azimuth_axes.append(channel_data.azimuth_axis)
        range_axes.append(channel_data.range_axis)
        swaths.add(channel_data.swath_name)
        polarizations.add(channel_data.polarization)

    if swath_name is not None and swath_name not in swaths:
        raise ra_err.SwathNotFoundError("Input Swath name has not been found")

    if len(swaths) > 1:
        # like Topsar
        # different channels are different sub-swaths
        if swath_name is None:
            raise ra_err.MultipleSwathError("Swath name must be provided if product has more than 1 swath")

    elif len(swaths) == 1:
        # like stripmap
        swath_name = swaths.pop()

        # generating times from pixel coordinates if input is in pixel
        # uniforming the time value for each channel (different channels aka poles means different raster infos)
        if azimuth_times_px is not None and is_pixel:
            azimuth_avg_times = _get_azimuth_channel_averaged_time(axes=azimuth_axes, pixels=azimuth_times_px)

    # check polarization
    if selected_polarization is not None:
        if selected_polarization not in polarizations:
            raise ra_err.PolarizationNotFoundError(f"Selected polarization {selected_polarization.value} not found")

    out_storage = []
    for channel in product.channels_list:
        # recovering metadata for the current channel
        channel_data = product.get_channel_data(channel_id=channel)

        if selected_polarization is not None:
            if channel_data.polarization != selected_polarization:
                continue

        # analyzing only selected swath
        if channel_data.swath_name != swath_name:
            continue

        log.info(
            f"Analyzing Channel {channel}, Swath {channel_data.swath_name}, "
            + f"Polarization {channel_data.polarization.value}..."
        )

        # taking into account bursts information
        if azimuth_times is not None:
            if is_pixel:
                bursts = channel_data.pixel_to_burst_association(azimuth_px_indexes=azimuth_times_px)
            else:
                bursts = channel_data.times_to_burst_association(azimuth_times=azimuth_times)
        else:
            bursts = [0] * len(range_times)

        # setup generic product dependent variables
        relative_az_axis = np.array(channel_data.azimuth_axis - channel_data.azimuth_axis[0]).astype("float64")
        az_mid_swath_pixel, rng_mid_swath_pixel = channel_data.times_to_pixel_conversion(
            azimuth_time=channel_data.mid_azimuth_time, range_time=channel_data.mid_range_time
        )
        az_mid_swath_pixel = int(np.round(az_mid_swath_pixel))
        rng_mid_swath_pixel = int(np.round(rng_mid_swath_pixel)) + 1

        if is_pixel:
            # compute times from pixel values
            azimuth_times, range_times = _get_times_from_pixels(
                channel=channel_data,
                azimuth_pixels=azimuth_times_px,
                range_pixels=range_times_px,
                bursts=bursts,
                az_avg_times=azimuth_avg_times,
            )
        else:
            # if times are provided as inputs, compute corresponding pixels
            azimuth_times_px, range_times_px = _get_pixels_from_times(
                channel=channel_data, azimuth_times=azimuth_times, range_times=range_times, bursts=bursts
            )
            if channel_data.projection == SARProjection.GROUND_RANGE and range_times is not None:
                range_times = np.array([channel_data.slant_range_axis[px] for px in range_times_px])

        if analysis_config.direction in (
            rdt.RadiometricAnalysisDirection.ALL,
            rdt.RadiometricAnalysisDirection.RANGE,
        ):
            if azimuth_times_px is None:
                raise ra_err.TimesDirectionMismatchError(
                    f"No azimuth times provided for {analysis_config.direction.name} radiometric analysis"
                )

            # extracting incidence angles and look angles, if needed
            incidence_cond = _is_incidence_angles_computation_required(
                config=analysis_config, direction=rdt.RadiometricAnalysisDirection.RANGE
            )
            look_cond = _is_look_angles_computation_required(
                config=analysis_config, direction=rdt.RadiometricAnalysisDirection.RANGE
            )

            incidence_angles, look_angles = None, None
            if incidence_cond:
                incidence_angles = compute_incidence_angles_from_trajectory(
                    trajectory=channel_data.trajectory,
                    azimuth_time=channel_data.mid_azimuth_time,
                    range_times=channel_data.slant_range_axis,
                    look_direction=channel_data.looking_side.value,
                )

            if look_cond:
                # extracting a look angle value for each range pixel
                look_angles = compute_look_angles_from_trajectory(
                    trajectory=channel_data.trajectory,
                    azimuth_time=channel_data.mid_azimuth_time,
                    range_times=channel_data.slant_range_axis,
                    look_direction=channel_data.looking_side.value,
                )

            out_axis = channel_data.range_axis.copy()
            if analysis_config.axis == rdt.RadiometricAnalysisAxes.INCIDENCE_ANGLE:
                out_axis = np.rad2deg(incidence_angles)
            elif analysis_config.axis == rdt.RadiometricAnalysisAxes.LOOK_ANGLE:
                out_axis = np.rad2deg(look_angles)

            for az_idx, az_pixel_index in enumerate(azimuth_times_px):
                log.info(f"Processing data for azimuth azimuth pixel {az_pixel_index}...")

                az_time = azimuth_times[az_idx]

                # processing swath portion and extracting smoothed and original profiles
                target_area = channel_data.read_data(
                    azimuth_index=az_pixel_index - 1,
                    range_index=rng_mid_swath_pixel,
                    cropping_size=(rng_mid_swath_pixel * 2, analysis_config.parameters.az_average_band),
                )
                smoothed_profile, original_profile = _radiometric_swath_processing(
                    target_area=target_area,
                    direction=rdt.RadiometricAnalysisDirection.RANGE,
                    config=analysis_config,
                    incidence_angles=incidence_angles,
                )

                # store variables in output dataclass
                out_storage.append(
                    rdt.RadiometricAnalysisOutput(
                        swath=channel_data.swath_name,
                        burst=bursts[az_idx] + 1,
                        channel=channel,
                        polarization=channel_data.polarization.value,
                        direction=rdt.RadiometricAnalysisDirection.RANGE,
                        value_type=analysis_config.value,
                        axis_format=analysis_config.axis,
                        radiometric_type=analysis_config.output_type,
                        profile=original_profile,
                        smoothed_profile=smoothed_profile,
                        axis=out_axis,
                        time=str(az_time),
                    )
                )

        if analysis_config.direction in (
            rdt.RadiometricAnalysisDirection.ALL,
            rdt.RadiometricAnalysisDirection.AZIMUTH,
        ):
            if range_times_px is None:
                raise ra_err.TimesDirectionMismatchError(
                    f"No range times provided for {analysis_config.direction.name} radiometric analysis"
                )

            for rng_idx, rng_pixel_index in enumerate(range_times_px):
                # recovering proper range data if Ground Range
                rng_time = range_times[rng_idx]

                log.info(f"Processing data for range pixel {rng_pixel_index}...")

                # extracting incidence angles and look angles, if needed
                incidence_cond = _is_incidence_angles_computation_required(
                    config=analysis_config, direction=rdt.RadiometricAnalysisDirection.AZIMUTH
                )

                incidence_angles = None
                if incidence_cond:
                    incidence_angles = compute_incidence_angles_from_trajectory(
                        trajectory=channel_data.trajectory,
                        azimuth_time=channel_data.mid_azimuth_time,
                        range_times=rng_time,
                        look_direction=channel_data.looking_side.value,
                    )

                # processing swath portion and extracting smoothed and original profiles
                target_area = channel_data.read_data(
                    azimuth_index=az_mid_swath_pixel,
                    range_index=rng_pixel_index - 1,
                    cropping_size=(analysis_config.parameters.rng_average_band, az_mid_swath_pixel * 2),
                )
                smoothed_profile, original_profile = _radiometric_swath_processing(
                    target_area=target_area,
                    direction=rdt.RadiometricAnalysisDirection.AZIMUTH,
                    config=analysis_config,
                    incidence_angles=incidence_angles,
                )

                # store variables in output dataclass
                out_storage.append(
                    rdt.RadiometricAnalysisOutput(
                        swath=channel_data.swath_name,
                        channel=channel,
                        polarization=channel_data.polarization.value,
                        direction=rdt.RadiometricAnalysisDirection.AZIMUTH,
                        value_type=analysis_config.value,
                        axis_format=rdt.RadiometricAnalysisAxes.NATURAL,
                        radiometric_type=analysis_config.output_type,
                        profile=original_profile,
                        smoothed_profile=smoothed_profile,
                        axis=relative_az_axis,
                        time=np.round(rng_time, 10),
                    )
                )

    return out_storage, channel_data.projection


def _get_pixels_from_times(
    channel: ChannelData,
    azimuth_times: Union[list[PreciseDateTime], None],
    range_times: Union[list[float], None],
    bursts: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Get pixel indexes from input times.

    Parameters
    ----------
    channel : ChannelData
        channel data
    azimuth_times : Union[list[PreciseDateTime], None]
        azimuth times from which get the corresponding pixel
    range_times : Union[list[float], None]
        range times from which get the corresponding pixel
    bursts : list[int]
        burst associated to each input time

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        azimuth pixels corresponding to input times, if not None
        range pixels corresponding to input times, if not None
    """
    # overwriting pixel initialized arrays with real pixel values
    azimuth_times_px, range_times_px = None, None
    if azimuth_times is not None:
        azimuth_times_px = [
            channel.times_to_pixel_conversion(azimuth_time=azimuth_times[idx], range_time=0, burst=bursts[idx])[0]
            for idx in range(len(bursts))
        ]
        azimuth_times_px = np.round(azimuth_times_px).astype("int64")
    if range_times is not None:
        range_times_px = [
            channel.times_to_pixel_conversion(
                azimuth_time=channel.mid_azimuth_time, range_time=range_times[idx], burst=bursts[idx]
            )[1]
            for idx in range(len(bursts))
        ]
        range_times_px = np.round(range_times_px).astype("int64")

    return azimuth_times_px, range_times_px


def _get_times_from_pixels(
    channel: ChannelData,
    azimuth_pixels: Union[list[int], None],
    range_pixels: Union[list[int], None],
    bursts: list[int],
    az_avg_times: Union[np.ndarray, None],
) -> tuple[np.ndarray, np.ndarray]:
    """Get time value from corresponding pixel.

    Parameters
    ----------
    channel : ChannelData
        channel data
    azimuth_pixels : Union[list[int], None]
        azimuth pixels from which to get the corresponding times
    range_pixels : Union[list[int], None]
        range pixels from which to get the corresponding times
    bursts : list[int]
        burst associated to each input time
    az_avg_times : Union[np.ndarray, None]
        average azimuth time in case of multiple channels

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        azimuth times corresponding to input times, if not None
        range times corresponding to input times, if not None
    """
    if azimuth_pixels is None:
        az_times = None

    else:
        az_times = []
        for idx, pixel in enumerate(azimuth_pixels):
            if az_avg_times is not None:
                az_times.append(az_avg_times[pixel])
            else:
                az_times.append(
                    channel.pixel_to_times_conversion(azimuth_index=pixel, range_index=0, burst=bursts[idx])[0]
                )
        az_times = np.array(az_times)

    if range_pixels is None:
        rng_times = None

    else:
        rng_times = []
        rng_times = np.array([channel.slant_range_axis[px] for px in range_pixels])

    return az_times, rng_times


def _init_pixel_arrays(
    azimuth_times: Union[np.ndarray, None], range_times: Union[np.ndarray, None], is_pixel: bool
) -> tuple[Union[np.ndarray, None], Union[np.ndarray, None]]:
    """This function initializes the azimuth pixel values and range pixel values from input vectors.
    It just creates copies of input arrays if there are no issues detected or values are not None.
    Main logic explained:
        - if input times are None, output pixels are None too
        - if input time is not None, it checks for possible mismatches between declared input (is_pixel flag) and actual
        values provided (i.e. input azimuth times are PreciseDateTime but pixel flag is True and vice versa)
        - raises errors if mismatches or issues are detected

    Parameters
    ----------
    azimuth_times : Union[np.ndarray, None]
        azimuth times, it could be None or an array of PreciseDateTime
    range_times : Union[np.ndarray, None]
        azimuth times, it could be None or an array of numbers
    is_pixel : bool
        pixel boolean flag, if True it means that input values are expected to be numbers corresponding to pixel indexes

    Returns
    -------
    tuple[Union[np.ndarray, None], Union[np.ndarray, None]]
        azimuth pixel array,
        range pixel array

    Raises
    ------
    ra_err.PixelTimesMismatchError
        if azimuth input values are PreciseDateTime but pixel flag is True, or input with mixed types
    ra_err.PixelTimesMismatchError
        if azimuth input values are floats but pixel flag is False, or input with mixed types
    ra_err.PixelTimesMismatchError
        if range input values are floats but pixel flag is True, or input with mixed types
    ra_err.PixelTimesMismatchError
        if range input values are integers but pixel flag is False, or input with mixed types
    """

    if azimuth_times is not None:
        if azimuth_times.ndim == 2:
            azimuth_times = azimuth_times.squeeze()

        main_type = {type(t) for t in azimuth_times}
        if is_pixel:
            if len(main_type) > 1 or not np.issubdtype(main_type.pop(), np.number):
                raise ra_err.PixelTimesMismatchError("Mismatch between pixel flag and input azimuth values")
        else:
            if len(main_type) > 1 or not main_type.pop() is PreciseDateTime:
                raise ra_err.PixelTimesMismatchError("Mismatch between pixel flag and input azimuth values")

        azimuth_times_px = azimuth_times.copy()
    else:
        azimuth_times_px = None

    if range_times is not None:
        range_times = np.asarray(range_times)
        if range_times.ndim == 2:
            range_times = range_times.squeeze()

        main_type = {type(t) for t in range_times}
        if is_pixel:
            if len(main_type) > 1 or not np.issubdtype(main_type.pop(), np.integer):
                raise ra_err.PixelTimesMismatchError("Mismatch between pixel flag and input range values")
        else:
            if len(main_type) > 1 or not np.issubdtype(main_type.pop(), np.floating):
                raise ra_err.PixelTimesMismatchError("Mismatch between pixel flag and input range values")

        range_times_px = range_times.copy()
    else:
        range_times_px = None

    return azimuth_times_px, range_times_px


def _get_azimuth_channel_averaged_time(axes: list[np.ndarray], pixels: np.ndarray) -> dict[int, PreciseDateTime]:
    """Generating uniform azimuth time for a given pixel value by averaging times across channels.

    Parameters
    ----------
    axes : list[np.ndarray]
        azimuth absolute axes, PreciseDateTime format, one for each channel in the product
    pixels : np.ndarray
        azimuth pixels selected for the radiometric analysis

    Returns
    -------
    dict[int, PreciseDateTime]
        dictionary with keys being the azimuth pixel and values being corresponding averaged times

    Raises
    ------
    ra_err.CoordinatesOutOfBounds
        if a pixel exceeds axes boundaries
    """

    # instantiating an empty dictionary with keys equal to the pixel values and values being empty lists
    azimuth_times_per_pixel = dict.fromkeys(pixels)

    try:
        for pixel in pixels:
            # for each pixel, selecting the corresponding time in each channel axis
            azimuth_times_per_pixel[pixel] = [ax[pixel] for ax in axes]
    except IndexError as err:
        raise CoordinatesOutOfBounds from err

    # for each pixel in the dictionary
    for key, value in azimuth_times_per_pixel.items():
        # taking the first value as a reference
        val_ref = value[0]
        # subtracting to each channel time the ref time
        rescaled_value = [v - val_ref for v in value]
        # calculating the mean time discrepancy between channels
        time_delta_avg = np.mean(rescaled_value)
        # taking the reference + mean delta time as the unique time for all these channels
        azimuth_times_per_pixel[key] = val_ref + time_delta_avg

    return azimuth_times_per_pixel


def _is_incidence_angles_computation_required(
    config: rdt.RadiometricAnalysisConfig, direction: rdt.RadiometricAnalysisDirection
) -> bool:
    """Compose incidence angles logical condition to check if incidence angles must be computed.

    Parameters
    ----------
    config : rdt.RadiometricAnalysisConfig
        radiometric analysis configuration dataclass
    direction : rdt.RadiometricAnalysisDirection
        radiometric analysis computation direction

    Returns
    -------
    bool
        boolean flag for computing incidence angles
    """

    # evaluating incidence angles logical condition
    # amplitude output + radiometric conversion required
    conversion_cond = np.logical_and(
        config.input_type != config.output_type, config.value == rdt.RadiometricAnalysisValue.AMPLITUDE
    )
    # output axis in incidence angles
    out_angle_cond = np.logical_or(
        direction == rdt.RadiometricAnalysisDirection.RANGE, config.axis == rdt.RadiometricAnalysisAxes.INCIDENCE_ANGLE
    )

    return np.logical_or(conversion_cond, out_angle_cond)


def _is_look_angles_computation_required(
    config: rdt.RadiometricAnalysisConfig, direction: rdt.RadiometricAnalysisDirection
) -> bool:
    """Compose look angles logical condition to check if look angles must be computed.

    Parameters
    ----------
    config : rdt.RadiometricAnalysisConfig
        radiometric analysis configuration dataclass
    direction : rdt.RadiometricAnalysisDirection
        radiometric analysis computation direction

    Returns
    -------
    bool
        boolean flag for computing look angles
    """

    # evaluating look angles logical condition
    # output axis in look angles
    return np.logical_and(
        direction == rdt.RadiometricAnalysisDirection.RANGE, config.axis == rdt.RadiometricAnalysisAxes.LOOK_ANGLE
    )


def _radiometric_swath_processing(
    target_area: np.ndarray,
    direction: rdt.RadiometricAnalysisDirection,
    config: rdt.RadiometricAnalysisConfig,
    incidence_angles: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray]:
    """This function orchestrates the whole process of processing input 2D swath portion.

    Parameters
    ----------
    target_area : np.ndarray
        target area 2D array to be processed to extract profiles
    direction : rdt.RadiometricAnalysisDirection
        direction of radiometric analysis
    config : rdt.RadiometricAnalysisConfig
        radiometric analysis config dataclass
    incidence_angles : np.ndarray, optional
        incidence angles array, by default None

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        smoothed average profile
        original average profile
    """

    if config.value == rdt.RadiometricAnalysisValue.AMPLITUDE:
        if config.input_type != config.output_type:
            if direction == rdt.RadiometricAnalysisDirection.RANGE:
                incidence_angle_array = np.tile(incidence_angles, (config.parameters.az_average_band, 1)).T
            elif direction == rdt.RadiometricAnalysisDirection.AZIMUTH:
                incidence_angle_array = np.ones_like(target_area) * incidence_angles

            target_area = radiometric_correction(
                data=target_area,
                incidence_angle=incidence_angle_array,
                input_type=config.input_type,
                output_type=config.output_type,
                exp_power=config.parameters.radiometric_correction_exponent,
            )

        smoothed_profile, original_profile = _extract_radiometric_profiles(
            data=target_area,
            direction=direction,
            outlier=config.outlier_removal,
            smoothening=config.smoothening_filter,
            config_params=config.parameters,
        )

        # conversion to decibel
        smoothed_profile = convert_to_db(smoothed_profile)
        original_profile = convert_to_db(original_profile)

    elif config.value == rdt.RadiometricAnalysisValue.PHASE:
        if direction == rdt.RadiometricAnalysisDirection.RANGE:
            smoothed_profile = np.nanmean(np.angle(target_area), 1)
        elif direction == rdt.RadiometricAnalysisDirection.AZIMUTH:
            smoothed_profile = np.nanmean(np.angle(target_area), 0)

        original_profile = smoothed_profile.copy()

    return smoothed_profile, original_profile


def _extract_radiometric_profiles(
    data: np.ndarray,
    direction: rdt.RadiometricAnalysisDirection,
    config_params: rdt.RadiometricAnalysisParameters,
    smoothening: bool = True,
    outlier: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract an average and a smoothed radiometric profile from input swath portion in the selected direction.
    Outliers can be removed if the input flag is True.

    Parameters
    ----------
    data : np.ndarray
        portion of the swath
    direction : rdt.RadiometricAnalysisDirection
        direction of profile extraction (range or azimuth)
    config_params : rdt.RadiometricAnalysisParameters
        configuration parameters for filtering and outliers removal
    smoothening : bool, optional
        smoothening flag, if True a Savitzky-Golay is applied with parameters specified as other inputs, by default True
    outlier : bool, optional
        boolean flag to enable the outlier removal from input data, by default False

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        average profile smoothed
        average profile
    """

    # switching to power domain
    data = data.copy()
    data = np.abs(data) ** 2

    # compute data profiles from original data as incoherent mean
    if direction == rdt.RadiometricAnalysisDirection.RANGE:
        original_profile = np.nanmean(data, 1)
    elif direction == rdt.RadiometricAnalysisDirection.AZIMUTH:
        original_profile = np.nanmean(data, 0)

    if outlier:
        filtered_data = masking_outliers(
            data=data,
            kernel_size=config_params.outliers_kernel_size,
            filter_size=config_params.outliers_filter_kernel_size,
            percentile_boundaries=config_params.outliers_percentile_boundaries,
        )
        if direction == rdt.RadiometricAnalysisDirection.RANGE:
            profile = np.nanmean(filtered_data, 1)
        elif direction == rdt.RadiometricAnalysisDirection.AZIMUTH:
            profile = np.nanmean(filtered_data, 0)
    else:
        profile = original_profile.copy()

    if smoothening:
        smth_window_len = config_params.smoothening_window_length
        smth_order = config_params.smoothening_order

        # applying Savitzky-Golay filter
        profile = savgol_filter(profile, window_length=smth_window_len, polyorder=smth_order)
        # forcing negative values to 0
        profile = np.clip(profile, a_min=0, a_max=None)

    return profile, original_profile
