# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Point Target Analysis support functions"""

import numpy as np
import pandas as pd
from arepytools.constants import LIGHT_SPEED
from arepytools.geometry.direct_geocoding import direct_geocoding_monostatic
from arepytools.geometry.geometric_functions import get_geometric_squint
from arepytools.geometry.inverse_geocoding_core import (
    inverse_geocoding_attitude_core,
    inverse_geocoding_monostatic_core,
)
from arepytools.io.io_support import NominalPointTarget
from arepytools.timing.precisedatetime import PreciseDateTime

from arepyextras.quality.core.generic_dataclasses import PointTargetVisibility
from arepyextras.quality.io.quality_input_protocol import (
    ChannelData,
    QualityInputProduct,
)


def check_targets_visibility(product: QualityInputProduct, points: dict[str, NominalPointTarget]) -> pd.DataFrame:
    """Checking if a set of targets is seen by the sensor in the recorded swath.

    Parameters
    ----------
    product : ProductManager
        product folder ProductManager instance
    points : dict[NominalPointTarget]
        dict of NominalPointTarget target dataclass

    Returns
    -------
    pd.DataFrame
        pandas dataframe collecting all visible points
    """

    coordinates = np.vstack([coord.xyz_coordinates for coord in points.values()])
    target_ids = list(points.keys())

    valid_points = []
    # iterating over channels
    for channel in product.channels_list:
        channel_data = product.get_channel_data(channel_id=channel)

        bursts_associations = channel_data.ground_points_to_burst_association(coordinates=coordinates)

        for index, item in enumerate(bursts_associations):
            valid_points.append(
                PointTargetVisibility(
                    id=target_ids[index],
                    channel=channel,
                    burst=item,
                    swath=channel_data.swath_name,
                    polarization=channel_data.polarization.name,
                )
            )

    return pd.DataFrame(valid_points)


def get_squint_angle(channel_data: ChannelData, azimuth_time: PreciseDateTime, ground_point: np.ndarray) -> float:
    """Compute squint angle (radians) for a given azimuth time and ground point.

    Parameters
    ----------
    channel_data : ChannelManager
        ChannelManager instance
    azimuth_time : PreciseDateTime
        azimuth time at which compute the squint angle
    ground_point : np.ndarray
        ground point seen by the sensor at the provided azimuth time

    Returns
    -------
    float
        squint angle (rad)
    """
    sensor_position = channel_data.trajectory.evaluate(azimuth_time).squeeze()
    sensor_velocity = channel_data.trajectory.evaluate_first_derivatives(azimuth_time).squeeze()

    return get_geometric_squint(
        sensor_positions=sensor_position, sensor_velocities=sensor_velocity, ground_points=ground_point
    )


def get_doppler_centroid(channel_data: ChannelData, azimuth_time: PreciseDateTime, ground_point: np.ndarray) -> float:
    """Computing doppler centroid frequency from azimuth time and its corresponding squint angle.

    Parameters
    ----------
    channel_data : ChannelManager
        ChannelManager instance
    azimuth_time : PreciseDateTime
        azimuth time at which compute doppler centroid frequency
    squint_angle : float
        sensor squint angle at that time

    Returns
    -------
    float
        doppler centroid frequency (Hertz)
    """

    squint_angle = get_squint_angle(channel_data=channel_data, azimuth_time=azimuth_time, ground_point=ground_point)
    sensor_velocity = channel_data.trajectory.evaluate_first_derivatives(azimuth_time).squeeze()
    sensor_velocity_norm = np.linalg.norm(sensor_velocity)
    carrier_freq = channel_data.carrier_frequency / LIGHT_SPEED

    return 2.0 * carrier_freq * sensor_velocity_norm * np.sin(squint_angle)


def compute_side_lobes_directions(
    channel_data: ChannelData,
    peak_azimuth_time: PreciseDateTime,
    peak_range_time: float,
    azimuth_step_m: float,
) -> tuple[tuple[float, float], float, float]:
    """Computing side lobe directions for squinted data and squint angle.

    Parameters
    ----------
    channel_data : ChannelManager
        ChannelManager instance
    peak_azimuth_time : PreciseDateTime
        azimuth time corresponding to the point target signal peak
    peak_range_time : float
        range time corresponding to the point target signal peak

    Returns
    -------
    tuple[float, float]
        side lobes directions,
        squint angle (radians),
        doppler centroid (Hz)
    """

    sensor_pos = channel_data.trajectory.evaluate(peak_azimuth_time)
    sensor_vel = channel_data.trajectory.evaluate_first_derivatives(peak_azimuth_time)

    earth_point_zero_doppler = direct_geocoding_monostatic(
        sensor_positions=sensor_pos,
        sensor_velocities=sensor_vel,
        range_times=peak_range_time,
        geocoding_side=channel_data.looking_side.value,
        geodetic_altitude=0,
        frequencies_doppler_centroid=0,
        wavelength=1,
    )

    if channel_data.boresight_normal_curve is None and channel_data.doppler_centroid is None:
        # no attitude or doppler centroid provided, returning zero doppler condition
        return np.array([np.inf, 0]), 0, 0

    if channel_data.boresight_normal_curve is not None:
        # computing side lobes with attitude
        sensor_time_with_doppler = inverse_geocoding_attitude_core(
            trajectory=channel_data.trajectory,
            boresight_normal=channel_data.boresight_normal_curve,
            ground_points=earth_point_zero_doppler,
            initial_guesses=peak_azimuth_time,
        )[0]

        # computing squint angle and doppler centroid
        squint_angle = get_squint_angle(
            channel_data=channel_data, azimuth_time=sensor_time_with_doppler, ground_point=earth_point_zero_doppler
        )
        doppler_centroid = get_doppler_centroid(
            channel_data=channel_data, azimuth_time=sensor_time_with_doppler, ground_point=earth_point_zero_doppler
        )

    elif channel_data.doppler_centroid is not None:
        # computing side lobes with doppler
        doppler_centroid = channel_data.doppler_centroid.evaluate(
            azimuth_time=peak_azimuth_time, range_time=peak_range_time
        )
        sensor_time_with_doppler = inverse_geocoding_monostatic_core(
            trajectory=channel_data.trajectory,
            ground_points=earth_point_zero_doppler,
            frequencies_doppler_centroid=doppler_centroid,
            wavelength=LIGHT_SPEED / channel_data.carrier_frequency,
            initial_guesses=peak_azimuth_time,
        )[0]
        sat_velocity = np.linalg.norm(channel_data.trajectory.evaluate_first_derivatives(peak_azimuth_time))
        squint_angle = doppler_centroid / (2 * sat_velocity / (LIGHT_SPEED / channel_data.carrier_frequency))

    sensor_position_zero_doppler = channel_data.trajectory.evaluate(peak_azimuth_time).T
    sensor_position_with_doppler = channel_data.trajectory.evaluate(sensor_time_with_doppler).T

    los_zd = np.squeeze(sensor_position_zero_doppler - earth_point_zero_doppler)
    los_hd = np.squeeze(sensor_position_with_doppler - earth_point_zero_doppler)
    slope = np.sign(doppler_centroid) * np.arctan2(np.linalg.norm(np.cross(los_zd, los_hd)), np.dot(los_zd, los_hd))

    # evaluating range and azimuth angular coefficients in samples (IRF Rng and Az cuts)
    step_ratio = azimuth_step_m / channel_data.range_step_m
    rng_cut = step_ratio / np.tan(slope)
    az_cut = -np.tan(slope) * step_ratio

    return (rng_cut, az_cut), squint_angle, doppler_centroid
