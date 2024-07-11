# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Point Target Full Analysis function: IRF, RCS and Localization Errors"""
from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
from arepytools.constants import LIGHT_SPEED
from arepytools.geometry.inverse_geocoding_core import inverse_geocoding_monostatic_core
from arepytools.io.io_support import NominalPointTarget

import arepyextras.quality.core.custom_errors as c_err
import arepyextras.quality.core.generic_dataclasses as gdt
import arepyextras.quality.core.signal_processing as sp
import arepyextras.quality.point_targets_analysis.custom_dataclasses as ptdt
from arepyextras.quality.io.quality_input_protocol import (
    ChannelData,
    QualityInputProduct,
)
from arepyextras.quality.point_targets_analysis.config import PointTargetAnalysisConfig
from arepyextras.quality.point_targets_analysis.point_target_irf import (
    PointTargetIRFAnalysis,
)
from arepyextras.quality.point_targets_analysis.support import (
    check_targets_visibility,
    compute_side_lobes_directions,
)

# syncing with logger
log = logging.getLogger("quality_analysis")


def point_target_analysis(
    product: QualityInputProduct,
    point_targets: dict[str, NominalPointTarget],
    config: PointTargetAnalysisConfig | None = None,
) -> tuple[pd.DataFrame, list[ptdt.PointTargetGraphicalData]]:
    """Function to compute the full point target analysis: IRF, RCS and localization errors.

    Parameters
    ----------
    product : QualityInputProduct
        object satisfying the QualityInputProduct protocol
    point_targets : dict[str, NominalPointTarget]
        dictionary of point targets locations, with keys being the target id label and value a NominalPointTarget
        dataclass instance with point target location data
    config : PointTargetAnalysisConfig, optional
        config file PointTargetAnalysisConfig dataclass to enable and manage different features, if provided,
        by default None

    Returns
    -------
    tuple[pd.DataFrame, list[ptdt.PointTargetGraphicalData]]
        pandas dataframe containing all the computed features for each point target,
        list of PointTargetGraphicalData data output for plotting graphs

    Raises
    ------
    c_err.SideLobesDirectionsEstimationError
        could not compute side lobes directions
    """

    # defining default config if None is given
    if config is None:
        log.info("Configuration file not provided, using default")
        config = PointTargetAnalysisConfig()

    log.info(f"Starting Point Target Analysis on {product.name}")
    log.info(f"Selected Product has {len(product.channels_list)} channels")

    # check which target are inside the scene
    if config.check_targets_in_scene:
        log.info("Checking which targets are visible in the scene")
        visible_targets = check_targets_visibility(product, point_targets)
        not_visible_targets = visible_targets.query("burst == None")["id"]
        visible_targets = visible_targets.query("burst.notnull()", engine="python")
        if not not_visible_targets.empty:
            log.info(f"Point Targets {not_visible_targets} are not visible in the scene.")

    # initializing lists for storing data
    graphs = []
    res = []
    for channel in visible_targets["channel"].unique():
        swath = visible_targets[visible_targets["channel"] == channel]["swath"].iloc[0]
        polar = visible_targets[visible_targets["channel"] == channel]["polarization"].iloc[0].upper()
        log.info(f"Analyzing Channel {channel}, Swath {swath}, Polarization {polar}...")

        # recovering metadata for the current channel
        channel_data = product.get_channel_data(channel_id=channel)

        # recovering only targets visible by this channel
        targets_visible_by_channel = visible_targets[visible_targets["channel"] == channel]["id"]

        for trgt_idx, trgt in enumerate(targets_visible_by_channel):
            bursts_selection = visible_targets.query("channel == @channel & id == @trgt")
            bursts_selection = bursts_selection.loc[:, "burst"].to_list()[0]

            for burst in bursts_selection:
                # initializing generic output info
                info = ptdt.GenericInfoOutput(
                    swath=channel_data.swath_name,
                    burst=burst,
                    product_type=channel_data.image_type.name.lower(),
                    polarization=channel_data.polarization.value,
                )

                log.info(
                    f"Processing Target Point {trgt} ({trgt_idx+1}/{len(targets_visible_by_channel)}), Burst #{burst}"
                )

                # extracting azimuth and range coordinates
                try:
                    trgt_az_time, trgt_rng_time = inverse_geocoding_monostatic_core(
                        trajectory=channel_data.trajectory,
                        ground_points=point_targets[trgt].xyz_coordinates,
                        initial_guesses=channel_data.mid_azimuth_time,
                        frequencies_doppler_centroid=0,
                        wavelength=1,
                    )
                    if point_targets[trgt].delay is not None:
                        trgt_rng_time += point_targets[trgt].delay

                    trgt_azmth_idx, trgt_rng_idx = channel_data.times_to_pixel_conversion(
                        azimuth_time=trgt_az_time, range_time=trgt_rng_time, burst=burst
                    )

                    az_rng_coords = gdt.SARCoordinates(
                        azimuth=trgt_az_time,
                        range=trgt_rng_time,
                        azimuth_index_subpx=trgt_azmth_idx,
                        range_index_subpx=trgt_rng_idx,
                    )
                except Exception:
                    az_rng_coords = gdt.SARCoordinates()

                if None in (
                    az_rng_coords.azimuth,
                    az_rng_coords.range,
                    az_rng_coords.azimuth_index_subpx,
                    az_rng_coords.range_index_subpx,
                ):
                    res.append(ptdt.PointTargetAnalysisOutput(target=trgt, channel=channel, info=info))
                    graphs.append(ptdt.PointTargetGraphicalData(target=trgt, channel=channel))
                    continue

                # evaluating additional metadata in product based on actual target
                location_data = channel_data.get_location_data(
                    azimuth_time=az_rng_coords.azimuth, range_time=az_rng_coords.range
                )

                # saving other useful info
                additional_info = ptdt.PTAdditionalInfo(
                    orbit_direction=channel_data.orbit_direction.name,
                    look_angle=np.rad2deg(location_data.look_angle),
                    ground_velocity=location_data.ground_velocity,
                )

                # ale limits conversion from meters to pixels
                ale_limits_rescaled = config.ale_limits
                if config.ale_limits is not None:
                    ale_limits_rescaled = (
                        np.max([np.ceil(config.ale_limits[0] / channel_data.range_step_m * 2), 3]),
                        np.max([np.ceil(config.ale_limits[1] / location_data.azimuth_step_m * 2), 3]),
                    )

                # extracting cropped target area centered on peak coordinates
                target_area, peak_coords, nominal_coords, peak_coords_swath = _extract_target_area(
                    channel_data=channel_data,
                    azimuth_range_coordinates=az_rng_coords,
                    ale_limits=ale_limits_rescaled,
                    initial_crop=config.irf_parameters.peak_finding_roi_size,
                    final_crop=config.irf_parameters.analysis_roi_size,
                )

                check_condition = (t is None for t in (target_area, peak_coords, nominal_coords))
                if any(check_condition):
                    res.append(
                        ptdt.PointTargetAnalysisOutput(
                            target=trgt, channel=channel, info=info, additional_info=additional_info
                        )
                    )
                    graphs.append(ptdt.PointTargetGraphicalData(target=trgt, channel=channel))
                    continue

                # assembling coordinates in the whole swath
                az_time_peak, rng_time_peak = channel_data.pixel_to_times_conversion(
                    azimuth_index=peak_coords_swath[1], range_index=peak_coords_swath[0], burst=burst
                )

                # calculating the side lobes directions and squint angle
                try:
                    side_lobes_directions, squint_angle, doppler_centroid = compute_side_lobes_directions(
                        channel_data=channel_data,
                        peak_azimuth_time=az_time_peak,
                        peak_range_time=rng_time_peak,
                        azimuth_step_m=location_data.azimuth_step_m,
                    )
                except Exception as err:
                    raise c_err.SideLobesDirectionsEstimationError("Could not evaluate side lobes directions") from err

                log.info(f"Measured squint angle: {np.round(np.rad2deg(squint_angle), 4)} \u00b0")

                if abs(np.rad2deg(squint_angle)) <= config.irf_parameters.zero_doppler_abs_squint_threshold_deg:
                    log.info(
                        "Measured squint angle is below threshold "
                        + f"{config.irf_parameters.zero_doppler_abs_squint_threshold_deg}"
                    )
                    log.info("Assuming zero doppler")
                    side_lobes_directions = np.array([np.inf, 0])

                # updating the azimuth and range steps for localization purposes due to side lobes directions
                original_az_step = location_data.azimuth_step_m
                original_rng_step = location_data.range_step_m
                if not np.isinf(side_lobes_directions[0]):
                    data_aspect_ratio = target_area.shape[1] / target_area.shape[0]
                    # managing range direction
                    if np.abs(side_lobes_directions[0] * data_aspect_ratio) > 1:
                        location_data.range_step_m = np.sqrt(
                            original_rng_step**2 + (original_az_step / side_lobes_directions[0]) ** 2
                        )
                    else:
                        location_data.range_step_m = np.sqrt(
                            (side_lobes_directions[0] * original_rng_step) ** 2 + original_az_step**2
                        )

                    # managing azimuth direction
                    if np.abs(side_lobes_directions[1] * data_aspect_ratio) > 1:
                        location_data.azimuth_step_m = np.sqrt(
                            original_rng_step**2 + (original_az_step / side_lobes_directions[1]) ** 2
                        )
                    else:
                        location_data.azimuth_step_m = np.sqrt(
                            (side_lobes_directions[1] * original_rng_step) ** 2 + original_az_step**2
                        )

                # estimating doppler rate and doppler rate theoretical at peak position
                doppler_rate = (
                    channel_data.doppler_rate.evaluate(azimuth_time=az_time_peak, range_time=rng_time_peak)
                    if channel_data.doppler_rate is not None
                    else None
                )
                doppler_rate_th = sp.compute_doppler_rate_theoretical(
                    trajectory=channel_data.trajectory,
                    azimuth_time=az_time_peak,
                    coords=point_targets[trgt].xyz_coordinates,
                    fc_hz=channel_data.carrier_frequency,
                )
                additional_info.doppler_rate_real = doppler_rate
                additional_info.doppler_rate_theoretical = doppler_rate_th

                # computing steering doppler frequency
                mid_burst_az, _ = channel_data.get_mid_burst_times(burst=burst)
                az_steering_rate = channel_data.get_steering_rate(azimuth_time=az_time_peak, burst=burst)
                steering_doppler_freq = 0
                if mid_burst_az is not None and az_steering_rate is not None and doppler_rate is not None:
                    steering_doppler_freq = sp.compute_steering_doppler_frequency(
                        trajectory=channel_data.trajectory,
                        azimuth_time=az_time_peak,
                        az_mid_burst_time=mid_burst_az,
                        az_steering_rate_rad_s=az_steering_rate,
                        doppler_rate=doppler_rate,
                        fc_hz=channel_data.carrier_frequency,
                    )

                # storing generic info for output
                additional_info.doppler_frequency = doppler_centroid
                additional_info.peak_azimuth_time = az_time_peak
                additional_info.peak_range_time = rng_time_peak
                additional_info.peak_azimuth_from_burst_start = peak_coords_swath[1] - sum(
                    channel_data.lines_per_burst[:burst]
                )
                additional_info.steering_doppler_frequency = steering_doppler_freq
                info.squint_angle = np.round(squint_angle, 6)
                info.range_position = peak_coords_swath[0]
                info.azimuth_position = peak_coords_swath[1]
                info.incidence_angle = np.rad2deg(location_data.incidence_angle)

                # evaluating IRF and RCS analysis on target area
                irf = PointTargetIRFAnalysis(
                    target_area=target_area,
                    target_pos_ref=nominal_coords,
                    target_pos_real=peak_coords,
                    mask_method=config.irf_parameters.masking_method,
                    oversampling_factor=config.irf_parameters.oversampling_factor,
                    rcs_interp_factor=config.rcs_parameters.interpolation_factor,
                    side_lobes_directions=side_lobes_directions,
                )
                if config.perform_irf:
                    log.debug("Performing IRF analysis...")
                    irf_res, graph_out_irf = irf.compute_irf(
                        pslr_flag=config.evaluate_pslr,
                        islr_flag=config.evaluate_islr,
                        sslr_flag=config.evaluate_sslr,
                        loc_errs_flag=config.evaluate_localization,
                    )

                    # storing resolutions and step distances for graphical output
                    graph_out_irf.rng_resolution = irf_res.range_resolution
                    graph_out_irf.az_resolution = irf_res.azimuth_resolution
                    graph_out_irf.rng_step_distance = location_data.range_step_m
                    graph_out_irf.az_step_distance = location_data.azimuth_step_m
                    graph_out_irf.side_lobes_directions = side_lobes_directions

                    # resolution and localization errors conversion to meters
                    irf_res.range_resolution *= location_data.range_step_m
                    irf_res.azimuth_resolution *= location_data.azimuth_step_m

                    irf_res.slant_range_localization_error *= -original_rng_step
                    irf_res.ground_range_localization_error *= -location_data.ground_range_step_m
                    irf_res.azimuth_localization_error *= -original_az_step

                    if config.perform_rcs:
                        log.debug("Performing RCS analysis...")
                        # correcting resolution taking into account the change in steps due to side lobes directions
                        if not np.isinf(side_lobes_directions[0]):
                            irf.irf_resolution_px = (
                                irf.irf_resolution_px[0] * location_data.range_step_m / original_rng_step,
                                irf.irf_resolution_px[1] * location_data.azimuth_step_m / original_az_step,
                            )

                        rcs_res, graph_out_rcs = irf.compute_rcs(
                            roi_size_factor=config.rcs_parameters.roi_dimension,
                            k_lin=config.rcs_parameters.calibration_factor,
                            s_f=config.rcs_parameters.resampling_factor,
                        )

                        # selecting proper spatial distances along range direction
                        step_distances = [location_data.range_step_m, location_data.azimuth_step_m]
                        if channel_data.projection == gdt.SARProjection.GROUND_RANGE:
                            original_rng_step *= np.sin(location_data.incidence_angle)
                            step_distances[0] *= np.sin(location_data.incidence_angle)

                        graph_out_rcs.rng_step_distance = step_distances[0]
                        graph_out_rcs.az_step_distance = step_distances[1]

                        # determining satellite position at a given azimuth time
                        sat_pos = channel_data.trajectory.evaluate(az_rng_coords.azimuth)

                        # rescaling rcs values, computing peak phase error
                        rcs_values = _compute_additional_rcs_values(
                            rcs_input=rcs_res,
                            step_distances=(original_rng_step, original_az_step),
                            interp_factor=irf.rcs_interp_factor,
                            polarization=channel_data.polarization,
                            target_info=point_targets[trgt],
                            sat_position=sat_pos,
                            fc_hz=channel_data.carrier_frequency,
                        )
                        graph_out_rcs.rcs_lin = rcs_values[0]
                        graph_out_rcs.rcs_db = rcs_values[1]
                        rcs_res.rcs = rcs_values[1]
                        rcs_res.rcs_error = rcs_values[2]
                        rcs_res.peak_phase_error = rcs_values[3]
                    else:
                        log.info("RCS analysis has been disabled in configuration file.")
                        rcs_res = ptdt.RCSDataOutput()
                        graph_out_rcs = ptdt.RCSGraphDataOutput()
                else:
                    log.info("IRF analysis has been disabled in configuration file.")
                    irf_res = ptdt.IRFDataOutput()
                    graph_out_irf = ptdt.IRFGraphDataOutput()
                    rcs_res = ptdt.RCSDataOutput()
                    graph_out_rcs = ptdt.RCSGraphDataOutput()

                # appending and storing IRF results for each channel
                res.append(
                    ptdt.PointTargetAnalysisOutput(
                        target=trgt,
                        channel=channel,
                        info=info,
                        additional_info=additional_info,
                        irf=irf_res,
                        rcs=rcs_res,
                    )
                )
                graphs.append(
                    ptdt.PointTargetGraphicalData(
                        target=trgt,
                        channel=channel,
                        swath=channel_data.swath_name,
                        burst=burst,
                        polarization=channel_data.polarization,
                        irf=graph_out_irf,
                        rcs=graph_out_rcs,
                    )
                )

    # storing values in a pandas dataframe
    if res:
        results_df = _results_to_dataframe(res)
        results_df.sort_values(by=["target", "polarization"], inplace=True)
    else:
        log.error("Provided Point Targets are not visible in the scene. Analysis could not be performed.")
        results_df = pd.DataFrame(columns=["error_point_target_not_in_scene"])
        graphs = [ptdt.PointTargetGraphicalData()]

    return results_df, graphs


def _extract_target_area(
    channel_data: ChannelData,
    azimuth_range_coordinates: gdt.SARCoordinates,
    ale_limits: tuple[float, float] | None = None,
    initial_crop: tuple[int, int] = (33, 33),
    final_crop: tuple[int, int] = (128, 128),
    ovrs_factor: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract a portion of the swath around target area from input product.

    Parameters
    ----------
    product : ProductManager
        aresys product manager instance
    azimuth_range_coordinates : dtc.SARCoordinates
        azimuth and range coordinates SARCoordinates dataclass
    ale_limits : tuple[float, float] | None, optional
        absolute localization error limits, by default None
    initial_crop : tuple[int, int], optional
        first step roi boundaries (range, azimuth), by default (33, 33)
    final_crop : tuple[int, int], optional
        final step roi boundaries (range, azimuth), by default (128, 128)
    ovrs_factor : int, optional
        oversampling factor (arbitrarily chosen)

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        cropped target area centered on interpolated peak coordinates,
        peak coordinates,
        nominal target reference coordinates in this frame [row, col],
        main lobe peak coordinates (rng[0], az[1]) in pixels referred to the whole swath
    """

    # managing ale limits
    if ale_limits is not None:
        initial_crop = tuple(int(ale) for ale in ale_limits)
        log.info(f"External Maximum ALE limits provided: using {initial_crop} ROI for peak searching")

    # first cropping around target nominal position
    try:
        log.debug(f"Cropping target area around nominal target position: size {initial_crop}")
        target_area = channel_data.read_data(
            azimuth_index=np.round(azimuth_range_coordinates.azimuth_index_subpx).astype("int64"),
            range_index=np.round(azimuth_range_coordinates.range_index_subpx).astype("int64"),
            cropping_size=initial_crop,
        )
    except (c_err.AzimuthExceedsBoundariesError, c_err.RangeExceedsBoundariesError) as err:
        log.critical(err)
        return None, None, None, None

    # locating real peak position in cropped image
    log.debug("Locating signal peak position")
    if np.isrealobj(target_area):
        _, peak_range_im, peak_azimuth_im = sp.locate_max_2d_interp(data=np.abs(target_area) ** 2)
    else:
        _, peak_range_im, peak_azimuth_im = sp.locate_max_2d_interp(data=target_area)

    if np.isnan(peak_range_im) or np.isnan(peak_azimuth_im):
        log.error("Could not find peak of the target area")
        return None, None, None, None

    # evaluating distance between nominal target and peak
    delta_rng_trgt_pk = peak_range_im - target_area.shape[0] // 2
    delta_az_trgt_pk = peak_azimuth_im - target_area.shape[1] // 2

    # if peak outside of ALE distance, break
    if ale_limits is not None:
        ale_break_cond = np.logical_or(
            np.abs(delta_rng_trgt_pk) > ale_limits[0], np.abs(delta_az_trgt_pk) > ale_limits[1]
        )
        if ale_break_cond:
            log.error("Target not within ALE limits")
            return None, None, None, None

    # second cropping, centered on peak coordinates
    peak_coords_swath = np.array(
        (
            np.round(azimuth_range_coordinates.range_index_subpx) - np.round(initial_crop[0] / 2) + peak_range_im,
            np.round(azimuth_range_coordinates.azimuth_index_subpx) - np.round(initial_crop[1] / 2) + peak_azimuth_im,
        )
    )
    peak_az_index = np.round(
        np.round(azimuth_range_coordinates.azimuth_index_subpx)
        - np.floor(initial_crop[1] / 2)
        + np.floor(peak_azimuth_im)
    )
    peak_rng_index = np.round(
        np.round(azimuth_range_coordinates.range_index_subpx) - np.floor(initial_crop[0] / 2) + np.floor(peak_range_im)
    )

    # final cropping around peak position
    try:
        log.debug(f"Cropping target area around signal peak position: size {final_crop}")
        target_area = channel_data.read_data(
            azimuth_index=int(peak_az_index),
            range_index=int(peak_rng_index),
            cropping_size=final_crop,
        )
    except (c_err.AzimuthExceedsBoundariesError, c_err.RangeExceedsBoundariesError) as err:
        log.critical(err)
        return None, None, None, None

    # checking for other conditions
    rng_ovrs = 1
    az_ovrs = 1

    if channel_data.sampling_constants is not None:
        try:
            rng_ovrs = np.max(
                [
                    np.round(
                        channel_data.sampling_constants.range_freq_hz
                        / channel_data.sampling_constants.range_bandwidth_freq_hz
                        / ovrs_factor
                    ),
                    1,
                ]
            )
            az_ovrs = np.max(
                [
                    np.round(
                        channel_data.sampling_constants.azimuth_freq_hz
                        / channel_data.sampling_constants.azimuth_bandwidth_freq_hz
                        / ovrs_factor
                    ),
                    1,
                ]
            )
        except ZeroDivisionError:
            rng_ovrs = 1
            az_ovrs = 1

    if rng_ovrs > 1 or az_ovrs > 1:
        try:
            target_area = channel_data.read_data(
                azimuth_index=int(peak_az_index - final_crop[1] * (az_ovrs - 1) / 2),
                range_index=int(peak_rng_index - final_crop[0] * (rng_ovrs - 1) / 2),
                cropping_size=(
                    np.round(final_crop[0] * rng_ovrs).astype("int64"),
                    np.round(final_crop[1] * az_ovrs).astype("int64"),
                ),
            )
        except (c_err.AzimuthExceedsBoundariesError, c_err.RangeExceedsBoundariesError) as err:
            log.critical(err)
            return None, None, None, None

    peak_coordinates = np.array(
        [
            target_area.shape[0] // 2 + peak_range_im - np.floor(peak_range_im),
            target_area.shape[1] // 2 + peak_azimuth_im - np.floor(peak_azimuth_im),
        ]
    )

    nom_rng = azimuth_range_coordinates.range_index_subpx - (
        np.round(peak_rng_index).astype("int64") - target_area.shape[0] // 2
    )
    nom_az = azimuth_range_coordinates.azimuth_index_subpx - (
        np.round(peak_az_index).astype("int64") - target_area.shape[1] // 2
    )
    nominal_coordinates = np.array([nom_rng, nom_az])

    return target_area, peak_coordinates, nominal_coordinates, peak_coords_swath


def _compute_additional_rcs_values(
    rcs_input: ptdt.RCSDataOutput,
    step_distances: list,
    interp_factor: int,
    polarization: gdt.SARPolarization,
    target_info: NominalPointTarget,
    sat_position: np.ndarray,
    fc_hz: float,
) -> tuple[float, float, float, float]:
    """Adjust rcs output values and calculate peak phase error.

    Parameters
    ----------
    rcs_input : ptdt.RCSDataOutput
        rcs output values from PointTargetIRF object
    step_distances : list
        step distances [range, azimuth]
    interp_factor : int
        rcs interpolation factor
    polarization : EPolarization
        polarization value [V/V, H/H, H/V, V/H]
    target_info : NominalPointTarget
        target info as NominalPointTarget
    sat_position : np.ndarray
        satellite position at given azimuth time
    fc_hz : float
        carrier frequency

    Returns
    -------
    tuple[float, float, float, float]
        rcs linear,
        rcs [db],
        rcs error [db],
        peak phase error [deg]
    """
    # convert rcs from intensity per unit pixel area to decibel
    rcs_pixel_area = np.prod(step_distances) / interp_factor**2
    rcs = rcs_pixel_area * rcs_input.rcs
    rcs_db = sp.convert_to_db(rcs)

    # selecting the right point target rcs reference value based on polarization
    if polarization == gdt.SARPolarization.HH:
        ptrcs = target_info.rcs_hh
    elif polarization == gdt.SARPolarization.HV:
        ptrcs = target_info.rcs_hv
    elif polarization == gdt.SARPolarization.VV:
        ptrcs = target_info.rcs_vv
    elif polarization == gdt.SARPolarization.VH:
        ptrcs = target_info.rcs_vh

    # evaluating RCS Error and Peak Phase Error
    arg = math.dist(sat_position, target_info.xyz_coordinates) / (LIGHT_SPEED / fc_hz)
    peak_phase_error = np.angle(rcs_input.peak_value_complex * np.exp(1j * 4 * np.pi * arg), deg=True)
    ptrcs_db = sp.convert_to_db(abs(ptrcs)) if np.iscomplexobj(ptrcs) else ptrcs
    rcs_error = rcs_db - ptrcs_db

    return rcs, rcs_db, rcs_error, peak_phase_error


def _results_to_dataframe(results: list[ptdt.PointTargetAnalysisOutput]) -> pd.DataFrame:
    """Organizing results dataclass into a single pandas dataframe for easy exporting.

    Parameters
    ----------
    results : list[dtc.PointTargetAnalysisOutput]
        list of PointTargetAnalysisOutput dataclass with stored results

    Returns
    -------
    pd.DataFrame
        pandas dataframe containing all the results organized
    """

    # extracting dataframes
    info_df = pd.DataFrame([r.info for r in results])
    additional_info_df = pd.DataFrame([r.additional_info for r in results])
    irf_df = pd.DataFrame([r.irf for r in results])
    rcs_df = pd.DataFrame([r.rcs for r in results])
    ch_trgt_df = pd.DataFrame([(r.target, r.channel) for r in results], columns=["target", "channel"])

    # merging dataframe horizontally
    df_res = pd.concat([ch_trgt_df, info_df, additional_info_df, irf_df, rcs_df], axis=1)
    df_res.drop(["peak_value_complex"], axis=1, inplace=True)

    # adding unit of measure to column names
    new_col = _add_unit_of_measure(columns=df_res.columns)
    df_res.columns = new_col
    try:
        df_res["target"] = pd.to_numeric(df_res["target"])
    except ValueError:
        pass

    return df_res


def _add_unit_of_measure(columns: pd.Index) -> list:
    """Attributing unit of measure to dataframe column names.

    Parameters
    ----------
    columns : pd.Index
        results dataframe column names

    Returns
    -------
    list
        new names with unit of measure added
    """

    ref_dict = {
        "incidence_angle": "[deg]",
        "look_angle": "[deg]",
        "ground_velocity": "[ms]",
        "doppler_frequency": "[Hz]",
        "steering_doppler_frequency": "[Hz]",
        "doppler_rate_real": "[Hzs]",
        "doppler_rate_theoretical": "[Hzs]",
        "peak_azimuth_time": "[UTC]",
        "peak_range_time": "[s]",
        "squint_angle": "[rad]",
        "azimuth_resolution": "[m]",
        "azimuth_pslr": "[dB]",
        "azimuth_islr": "[dB]",
        "azimuth_sslr": "[dB]",
        "azimuth_localization_error": "[m]",
        "range_resolution": "[m]",
        "range_pslr": "[dB]",
        "range_islr": "[dB]",
        "range_sslr": "[dB]",
        "pslr_2d": "[dB]",
        "islr_2d": "[dB]",
        "sslr_2d": "[dB]",
        "ground_range_localization_error": "[m]",
        "slant_range_localization_error": "[m]",
        "rcs": "[dB]",
        "rcs_error": "[dB]",
        "peak_phase_error": "[deg]",
        "clutter": "[dB]",
        "scr": "[dB]",
    }

    new_col = [(c + "_" + ref_dict[c] if c in ref_dict else c) for c in columns]

    return new_col
