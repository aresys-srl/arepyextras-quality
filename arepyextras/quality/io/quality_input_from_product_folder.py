# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Aresys Product Folder wrapper compliant with Quality Input Protocol"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import numpy.typing as npt
from arepytools.constants import LIGHT_SPEED
from arepytools.geometry.curve import Generic3DCurve
from arepytools.geometry.generalsarattitude import (
    create_attitude_boresight_normal_curve_wrapper,
    create_general_sar_attitude,
)
from arepytools.geometry.generalsarorbit import (
    GSO3DCurveWrapper,
    compute_ground_velocity,
    compute_incidence_angles_from_orbit,
    compute_look_angles_from_orbit,
    create_general_sar_orbit,
)
from arepytools.geometry.inverse_geocoding import inverse_geocoding_monostatic
from arepytools.io import (
    open_product_folder,
    read_metadata,
    read_raster_with_raster_info,
)
from arepytools.io.metadata import DopplerCentroidVector, DopplerRateVector
from arepytools.math.genericpoly import create_sorted_poly_list
from arepytools.timing.precisedatetime import PreciseDateTime

import arepyextras.quality.core.custom_errors as c_err
from arepyextras.quality.core.generic_dataclasses import (
    SARImageType,
    SAROrbitDirection,
    SARPolarization,
    SARProjection,
    SARRadiometricQuantity,
    SARSamplingFrequencies,
    SARSideLooking,
)
from arepyextras.quality.io.quality_input_protocol import LocationData


class DopplerPolynomialWrapper:
    """Generic Polynomial wrapper used to interpolate Doppler data (Doppler Centroid or Rate)"""

    def __init__(self, poly_list: Union[DopplerCentroidVector, DopplerRateVector]) -> None:
        self._sorted_poly_list = create_sorted_poly_list(poly_list)

    def evaluate(self, azimuth_time: PreciseDateTime, range_time: float) -> float:
        """Evaluate the Doppler Polynomial at given azimuth and range times.

        Parameters
        ----------
        azimuth_time : PreciseDateTime
            azimuth time at which evaluate the polynomial
        range_time : float
            range time at which evaluate the polynomial

        Returns
        -------
        float
            doppler at that time
        """
        return self._sorted_poly_list.evaluate((azimuth_time, range_time))


class ProductFolderManager:
    """Product Manager class satisfying the QualityInputProduct protocol"""

    def __init__(self, path: Union[str, Path]) -> None:
        self._path = Path(path)
        self._product_name = self._path.name
        self._product = open_product_folder(self._path)
        self._channel_list = self._product.get_channels_list()

    @property
    def path(self) -> Path:
        """Get product path"""
        return self._path

    @property
    def name(self) -> str:
        """Get product name"""
        return self._product_name

    @property
    def channels_list(self) -> list[int]:
        """Get list of available channels for this product"""
        return self._channel_list

    def get_channel_data(self, channel_id: int) -> ChannelManager:
        """Get data and info referring to the selected channel.

        Parameters
        ----------
        channel_id : int
            selected channel number

        Returns
        -------
        ChannelManager
            ChannelManager containing data corresponding to the selected channel
        """
        metadata = self._product.get_channel_metadata(channel_id)
        raster = self._product.get_channel_data(channel_id)
        return ChannelManager(channel_metadata_path=metadata, channel_raster_path=raster, channel_num=channel_id)


class ChannelManager:
    """Channel Manager class satisfying the ChannelData protocol"""

    def __init__(self, channel_metadata_path: Path, channel_raster_path: Path, channel_num: int) -> None:
        """Creating a ChannelManager object compliant with the ChannelData protocol.

        Parameters
        ----------
        channel_metadata_path : Path
            Path to the channel metadata xml file
        channel_raster_path : int
            Path to the channel raster file
        channel_num : int
            number of current channel
        """
        self._channel_num = channel_num
        self._channel_raster = channel_raster_path
        self._channel_metadata = read_metadata(channel_metadata_path)
        self._state_vectors = self._channel_metadata.get_state_vectors()
        self._raster_info = self._channel_metadata.get_raster_info()
        self._swath_info = self._channel_metadata.get_swath_info()
        self._dataset_info = self._channel_metadata.get_dataset_info()
        self._attitude_info = self._channel_metadata.get_attitude_info()
        self._burst_info = self._channel_metadata.get_burst_info()
        self._pulse = self._channel_metadata.get_pulse()
        self._acquisition_time_line = self._channel_metadata.get_acquisition_time_line()
        # TODO: read this from product metadata
        self._radiometric_quantity = SARRadiometricQuantity.BETA_NOUGHT

        # re-arranging signal sampling frequencies
        self._sampling_constants = self._channel_metadata.get_sampling_constants()
        self._signal_constants = SARSamplingFrequencies(
            range_freq_hz=self._sampling_constants.frg_hz,
            azimuth_freq_hz=self._sampling_constants.faz_hz,
            range_bandwidth_freq_hz=self._sampling_constants.brg_hz,
            azimuth_bandwidth_freq_hz=self._sampling_constants.baz_hz,
        )
        self._signal_pulse_rate = np.nan
        if self._pulse.bandwidth is not None:
            self._signal_pulse_rate = self._pulse.bandwidth / self._pulse.pulse_length

        # creating doppler centroid and rate polynomial wrappers
        centroid_poly = self._channel_metadata.get_doppler_centroid()
        rate_poly = self._channel_metadata.get_doppler_rate()
        self._doppler_centroid_poly = (
            DopplerPolynomialWrapper(poly_list=centroid_poly) if centroid_poly.get_number_of_poly() > 0 else None
        )
        self._doppler_rate_poly = (
            DopplerPolynomialWrapper(poly_list=rate_poly) if rate_poly.get_number_of_poly() > 0 else None
        )
        # retrieving azimuth steering rate polynomial coefficients
        self._steering_rate_poly_coeff = self._swath_info.azimuth_steering_rate_pol

        # swath and main parameters
        self._swath = self._swath_info.swath
        self._product_folder_image_type = SARImageType.from_str(self._dataset_info.image_type)
        self._channel_projection = SARProjection(self._dataset_info.projection)
        self._polarization = SARPolarization(self._swath_info.polarization.value)
        self._looking_side = SARSideLooking(self._dataset_info.side_looking.value.upper())
        self._carrier_freq = self._dataset_info.fc_hz
        rng_time_half_swath = (
            self._raster_info.samples_start + (self._raster_info.samples - 1) * self._raster_info.samples_step / 2
        )
        self._azimuth_axis = self._compute_azimuth_axis()
        self._az_time_half_swath = self._azimuth_axis[self.azimuth_axis.size // 2]
        self._range_axis = (
            np.arange(0, self._raster_info.samples, 1) * self._raster_info.samples_step
            + self._raster_info.samples_start
        )
        self._slant_range_axis = self._compute_slant_range_axis()
        if self._channel_projection == SARProjection.GROUND_RANGE:
            poly = self._channel_metadata.get_ground_to_slant()
            rng_time_half_swath = create_sorted_poly_list(poly).evaluate(
                (self._az_time_half_swath, np.floor(rng_time_half_swath))
            )
        self._rng_time_half_swath = rng_time_half_swath

        # computing range_step_m
        self._az_step_s = self._raster_info.lines_step
        self._range_step_m = self._compute_range_step_m()

        # generating trajectory
        self._orbit = create_general_sar_orbit(self._state_vectors, ignore_anx_after_orbit_start=True)
        self._trajectory = GSO3DCurveWrapper(orbit=self._orbit)
        self._orbit_direction = SAROrbitDirection[self._state_vectors.orbit_direction.value]

        # generating attitude boresight normal curve
        self._boresight_normal = None
        if self._attitude_info is not None:
            self._attitude = create_general_sar_attitude(
                self._state_vectors, attitude_info=self._attitude_info, ignore_anx_after_orbit_start=True
            )
            self._boresight_normal = create_attitude_boresight_normal_curve_wrapper(attitude=self._attitude)

        # assemble swst changes
        self._swst_changes = None
        if self._acquisition_time_line is not None and self._acquisition_time_line._swst_changes_values:
            self._swst_changes = list(
                zip(
                    [
                        self._raster_info.lines_start + t
                        for t in self._acquisition_time_line._swst_changes_azimuth_times
                    ],
                    self._acquisition_time_line._swst_changes_values,
                )
            )

        if self._burst_info is not None:
            self._lines_per_burst_array = np.repeat(
                self._burst_info.lines_per_burst, self._burst_info.get_number_of_bursts()
            )
        else:
            # should be a 1D array
            self._lines_per_burst_array = np.repeat(self._raster_info.lines, 1)

    def _compute_slant_range_axis(self) -> np.ndarray:
        """Computing slant range full axis.

        Returns
        -------
        np.ndarray
            slant range axis
        """
        slant_rng_axis = self._range_axis
        if self._channel_projection == SARProjection.GROUND_RANGE:
            poly = self._channel_metadata.get_ground_to_slant()
            slant_rng_axis = create_sorted_poly_list(poly).evaluate((self._az_time_half_swath, self._range_axis))

        return slant_rng_axis

    def _compute_azimuth_axis(self) -> np.ndarray:
        """Compute azimuth full axis.

        Returns
        -------
        np.ndarray
            azimuth axis
        """
        az_axis = (
            np.arange(0, self._raster_info.lines, 1) * self._raster_info.lines_step + self._raster_info.lines_start
        )
        if self._burst_info is not None:
            az_axis = []
            for brst in range(self._burst_info.get_number_of_bursts()):
                az_axis.append(
                    self._burst_info.get_azimuth_start_time(brst)
                    + np.arange(0, self._burst_info.get_lines(brst), 1) * self._raster_info.lines_step
                )
            az_axis = np.concatenate(az_axis)
        return az_axis

    def _compute_range_step_m(self) -> float:
        """Computing step along range direction, in meters"""
        if self._channel_projection == SARProjection.GROUND_RANGE:
            return self._raster_info.samples_step

        return self._raster_info.samples_step * LIGHT_SPEED / 2

    def _get_raster_layout(self) -> tuple[list[PreciseDateTime], list[float]]:
        """Evaluating raster boundaries taking into account the bursts, if needed.

        Returns
        -------
        tuple[list[list[PreciseDateTime, PreciseDateTime]], list[list[float, float]]]
            azimuth raster boundaries (azimuth start, azimuth stop),
            range raster boundaries (range start, range stop)
        """

        if self._burst_info is not None:
            az_times = self._burst_info.get_azimuth_start_time()
            rng_times = self._burst_info.get_range_start_time()
            burst_az_boundaries = []
            for az_time in az_times:
                burst_az_boundaries.append(
                    [az_time, az_time + self._burst_info.lines_per_burst * self._raster_info.lines_step]
                )
            burst_rng_boundaries = []
            for rng_time in rng_times:
                burst_rng_boundaries.append(
                    [rng_time, rng_time + self._raster_info.samples * self._raster_info.samples_step]
                )
        else:
            burst_az_boundaries = [
                [
                    self._raster_info.lines_start,
                    self._raster_info.lines_start + self._raster_info.lines * self._raster_info.lines_step,
                ]
            ]
            burst_rng_boundaries = [
                [
                    self._raster_info.samples_start,
                    self._raster_info.samples_start + self._raster_info.samples * self._raster_info.samples_step,
                ]
            ]

        return burst_az_boundaries, burst_rng_boundaries

    def _times_to_burst_association(self, azimuth_times: npt.ArrayLike) -> list[int]:
        """Associate the right burst to a given input time point. This function returns 1 association for each
        input time.
        Associating time only to the first burst containing it.

        Parameters
        ----------
        azimuth_time : npt.ArrayLike
            azimuth time array in PreciseDateTime format

        Returns
        -------
        list[int]
            burst associated with a given time

        Raises
        ------
        c_err.CoordinatesOutOfBounds
            if input time exceeds tme boundaries of the swath
        """
        if self._burst_info is None:
            return [0] * len(azimuth_times)

        bursts_start_times = self._burst_info.get_azimuth_start_time()
        last_time = bursts_start_times[0] + self._burst_info.get_lines().sum() * self._raster_info.lines_step

        bursts = []
        for time in azimuth_times:
            if time < bursts_start_times[0] or time > last_time:
                raise c_err.CoordinatesOutOfBounds(f"{time} is out of the recorded timeline")

            time_diff = time - bursts_start_times
            time_mask = np.ma.masked_less(time_diff.astype("float64"), 0)
            # associating time only to the first burst containing it
            bursts.append(time_mask.argmin())

        return bursts

    @property
    def swath_name(self) -> str:
        """Name of the swath being analyzed"""
        return self._swath

    @property
    def channel_id(self) -> int:
        """Number corresponding to the current channel data"""
        return self._channel_num

    @property
    def range_step_m(self) -> float:
        """Step along range direction, in meters"""
        return self._range_step_m

    @property
    def azimuth_step_s(self) -> float:
        """Step along azimuth direction, in seconds"""
        return self._az_step_s

    @property
    def projection(self) -> SARProjection:
        """Channel data projection"""
        return self._channel_projection

    @property
    def polarization(self) -> SARPolarization:
        """Channel data polarization"""
        return self._polarization

    @property
    def orbit_direction(self) -> SAROrbitDirection:
        """Channel data orbit direction"""
        return self._orbit_direction

    @property
    def image_type(self) -> SARImageType:
        """Channel raster image type"""
        return self._product_folder_image_type

    @property
    def sampling_constants(self) -> SARSamplingFrequencies:
        """Channel data signal sampling frequencies"""
        return self._signal_constants

    @property
    def pulse_rate(self) -> float:
        """Signal pulse rate"""
        return self._signal_pulse_rate

    @property
    def looking_side(self) -> SARSideLooking:
        """Sensor look direction for this channel"""
        return self._looking_side

    @property
    def carrier_frequency(self) -> float:
        """Signal carrier frequency"""
        return self._carrier_freq

    @property
    def mid_azimuth_time(self) -> PreciseDateTime:
        """Azimuth time at half swath"""
        return self._az_time_half_swath

    @property
    def trajectory(self) -> GSO3DCurveWrapper:
        """Channel trajectory 3D curve"""
        return self._trajectory

    @property
    def boresight_normal_curve(self) -> Union[None, Generic3DCurve]:
        """Channel attitude boresight normal 3D curve"""
        return self._boresight_normal

    @property
    def doppler_centroid(self) -> Union[None, DopplerPolynomialWrapper]:
        """Channel doppler centroid polynomial wrapper"""
        return self._doppler_centroid_poly

    @property
    def doppler_rate(self) -> Union[None, DopplerPolynomialWrapper]:
        """Channel doppler rate polynomial wrapper"""
        return self._doppler_rate_poly

    @property
    def mid_range_time(self) -> float:
        """Range time at half swath"""
        return self._rng_time_half_swath

    @property
    def range_axis(self) -> np.ndarray:
        """Range axis"""
        return self._range_axis

    @property
    def slant_range_axis(self) -> np.ndarray:
        """Slant Range axis"""
        return self._slant_range_axis

    @property
    def azimuth_axis(self) -> np.ndarray:
        """Azimuth axis, PreciseDateTime format"""
        return self._azimuth_axis

    @property
    def lines_per_burst(self) -> np.ndarray:
        """Lines per burst, for each burst in the swath"""
        return self._lines_per_burst_array

    @property
    def radiometric_quantity(self) -> np.ndarray:
        """Product radiometric quantity"""
        return self._radiometric_quantity

    @property
    def pulse_latch_time(self) -> float:
        """Signal pulse latch time"""
        return np.nan

    @property
    def swst_changes(self) -> Union[list[tuple[PreciseDateTime, float]], None]:
        """SWST changes list as tuple of time of change and new SWST value"""
        return self._swst_changes

    def get_mid_burst_times(self, burst: int) -> tuple[PreciseDateTime, float]:
        """Compute mid azimuth and range times for a given burst.

        Returns
        -------
        tuple(PreciseDateTime, float)
            azimuth and range mid burst times
        """
        az_mid_burst = self.mid_azimuth_time
        rng_mid_burst = self.mid_range_time
        if self._burst_info is not None:
            az_time_boundaries, rng_time_boundaries = self._get_raster_layout()
            az_mid_burst = (az_time_boundaries[burst][1] - az_time_boundaries[burst][0]) / 2 + az_time_boundaries[
                burst
            ][0]
            rng_mid_burst = (rng_time_boundaries[burst][1] - rng_time_boundaries[burst][0]) / 2 + rng_time_boundaries[
                burst
            ][0]

        return az_mid_burst, rng_mid_burst

    def get_steering_rate(self, azimuth_time: PreciseDateTime, burst: int) -> float:
        """Compute steering rate at a given azimuth time and for a given burst.

        Parameters
        ----------
        azimuth_time : PreciseDateTime
            azimuth time
        burst : int
            burst corresponding to the input time

        Returns
        -------
        float
            azimuth steering rate
        """
        if self._burst_info is not None:
            time_rel = azimuth_time - self._burst_info.get_azimuth_start_time(burst)
        else:
            time_rel = azimuth_time - self._raster_info.lines_start
        return (
            self._steering_rate_poly_coeff[0]
            + self._steering_rate_poly_coeff[1] * time_rel
            + self._steering_rate_poly_coeff[2] * time_rel**2
        )

    def pixel_to_times_conversion(
        self, azimuth_index: float, range_index: float, burst: int = None
    ) -> tuple[PreciseDateTime, float]:
        """Converting input raster pixel coordinates (azimuth_index and range index) to corresponding absolute times,
        azimuth and range.

        Parameters
        ----------
        azimuth_index : float
            azimuth pixel index, subpixel precision
        range_index : float
            range pixel index, subpixel precision
        burst : int, optional
            burst index, by default None

        Returns
        -------
        tuple[PreciseDateTime, float]
            azimuth time,
            range time
        """

        if self._burst_info is not None and burst is not None:
            start_time_az = self._burst_info.get_azimuth_start_time(burst)
            start_time_rng = self._burst_info.get_range_start_time(burst)
        else:
            start_time_az = self._raster_info.lines_start
            start_time_rng = self._raster_info.samples_start

        rng_time = range_index * self._raster_info.samples_step + start_time_rng
        if self._burst_info is not None and burst is not None:
            az_time = (
                azimuth_index - self._burst_info.get_lines()[:burst].sum()
            ) * self._raster_info.lines_step + start_time_az
        else:
            az_time = azimuth_index * self._raster_info.lines_step + start_time_az

        if self.projection == SARProjection.GROUND_RANGE:
            poly = self._channel_metadata.get_ground_to_slant()
            rng_time = create_sorted_poly_list(poly).evaluate((self.mid_azimuth_time, rng_time))

        return az_time, rng_time

    def times_to_pixel_conversion(
        self, azimuth_time: PreciseDateTime, range_time: float, burst: int = None
    ) -> tuple[float, float]:
        """Converting azimuth and range times to raster image pixels indexes with subpixel precision.

        Parameters
        ----------
        azimuth_time : PreciseDateTime
            azimuth time
        range_time : float
            range time
        burst : int
            burst number corresponding to these times

        Returns
        -------
        tuple[float, float]
            pixel corresponding to azimuth time,
            pixel corresponding to range time
        """

        rng_value = range_time
        if self.projection == SARProjection.GROUND_RANGE:
            # if projection is GROUND RANGE, range info are expressed in meters, so it must be converted to meters
            poly = self._channel_metadata.get_slant_to_ground()
            rng_value = create_sorted_poly_list(poly).evaluate((azimuth_time, range_time))

        if self._burst_info is not None:
            # i.e. for TOPSAR products, burst information must be taken into account
            if burst is None:
                burst = self._times_to_burst_association([azimuth_time])[0]

            rng_idx = (rng_value - self._burst_info.get_range_start_time(burst)) / self._raster_info.samples_step
            azmth_idx = (
                azimuth_time - self._burst_info.get_azimuth_start_time(burst)
            ) / self._raster_info.lines_step + self._burst_info.get_lines()[:burst].sum()

        else:
            # i.e. for STRIPMAP products, azimuth and range indexes are referred to a single swath
            rng_idx = (rng_value - self._raster_info.samples_start) / self._raster_info.samples_step
            azmth_idx = (azimuth_time - self._raster_info.lines_start) / self._raster_info.lines_step

        return azmth_idx, rng_idx

    def ground_points_to_burst_association(self, coordinates: npt.ArrayLike) -> list[Union[list[int], None]]:
        """Determining the burst (or bursts) where the input coordinates lie. If no association can be found (i.e. the
        point is not visible in the scene), None is returned.

        Parameters
        ----------
        coordinates : npt.ArrayLike
            array of coordinates, in the form (N, 3)

        Returns
        -------
        list[Union[list[int], None]]
            list containing the burst association for each input point, None if no association was found
        """

        coordinates = np.atleast_2d(coordinates)

        burst_az_boundaries, burst_rng_boundaries = self._get_raster_layout()

        bursts = []
        for point in coordinates:
            try:
                t_azmth, t_rng = inverse_geocoding_monostatic(
                    orbit=self._orbit,
                    ground_points=point,
                    wavelength=1,
                    frequencies_doppler_centroid=0,
                )

                az_check = [(t_azmth < az[1] and t_azmth > az[0]) for az in burst_az_boundaries]
                rng_check = [(t_rng < rng[1] and t_rng > rng[0]) for rng in burst_rng_boundaries]
                check = np.logical_and(az_check, rng_check)
                if check.any():
                    bursts.append(list(np.where(check)[0]))
                else:
                    bursts.append(None)
            except Exception:
                bursts.append(None)

        return bursts

    def read_data(
        self,
        azimuth_index: int,
        range_index: int,
        cropping_size: tuple[int, int] = (150, 150),
        output_radiometric_quantity: SARRadiometricQuantity = SARRadiometricQuantity.BETA_NOUGHT,
    ) -> np.ndarray:
        """Extracting the swath portion centered to the provided target position and of size cropping_size by
        cropping_size. Target position is provided via its azimuth and range indexes in the swath array.

        Parameters
        ----------
        azimuth_index : int
            index of azimuth time in swath array
        range_index : int
            index of range time in swath array
        cropping_size : tuple[int, int], optional
            size in pixel of the swath portion to be read (number of samples, number of lines), by default (150, 150)
        output_radiometric_quantity : SARRadiometricQuantity, optional
            selected output radiometric quantity to convert the read data to, if needed,
            by default SARRadiometricQuantity.BETA_NOUGHT

        Returns
        -------
        np.ndarray
            cropped swath array centered to the input target coordinates, data is provided with shape (samples, lines)
            by default the output radiometric quantity is BETA_NOUGHT, unless specified otherwise
        Raises
        ------
        c_err.AzimuthExceedsBoundariesError
            azimuth index exceeds swath boundaries
        c_err.RangeExceedsBoundariesError
            range index exceeds swath boundaries
        """

        # this is due to the fact that PF products are BETA but do not have an incidence angle poly to convert data
        if self._radiometric_quantity != output_radiometric_quantity:
            raise RuntimeError("Cannot convert radiometric quantity")

        # creating the target block identifier for partial swath reading
        # [start line, start sample, number of lines, number of samples]
        target_block = [
            azimuth_index - np.floor(cropping_size[1] / 2).astype(int),
            range_index - np.floor(cropping_size[0] / 2).astype(int),
            cropping_size[1],
            cropping_size[0],
        ]
        if target_block[0] > self._raster_info.lines or target_block[0] < 0:
            # starting azimuth line to be read is out of swath boundaries
            raise c_err.AzimuthExceedsBoundariesError(
                f"First ROI line {target_block[0]} is out of azimuth swath boundaries"
            )

        if target_block[1] > self._raster_info.samples or target_block[1] < 0:
            # starting range sample to be read is out of swath boundaries
            raise c_err.RangeExceedsBoundariesError(
                f"First ROI sample {target_block[1]} is out of range swath boundaries"
            )

        if target_block[0] + target_block[2] > self._raster_info.lines:
            # last azimuth line to be read is out of swath boundaries
            raise c_err.AzimuthExceedsBoundariesError(
                f"Last ROI line {target_block[0] + target_block[2]} exceeds azimuth swath boundaries"
            )

        if target_block[1] + target_block[3] > self._raster_info.samples:
            # last range sample to be read is out of swath boundaries
            raise c_err.RangeExceedsBoundariesError(
                f"Last ROI sample {target_block[1] + target_block[3]} exceeds range swath boundaries"
            )

        return read_raster_with_raster_info(
            raster_file=self._channel_raster, raster_info=self._raster_info, block_to_read=target_block
        ).T

    def get_location_data(self, azimuth_time: PreciseDateTime, range_time: float) -> LocationData:
        """Generating a LocationData object containing data and info derived from the current ChannelManager and
        declined to the specific azimuth and range times selected.

        Parameters
        ----------
        abs_azimuth_time : PreciseDateTime
            selected absolute azimuth time
        abs_range_time : float
            selected absolute range time

        Returns
        -------
        LocationData
            LocationData instance related to the selected location
        """

        incidence_angle = compute_incidence_angles_from_orbit(
            orbit=self._orbit,
            azimuth_time=azimuth_time,
            range_times=range_time,
            look_direction=self.looking_side.value,
        )
        # TODO compute look angles/ground velocity directly at target position and not a mid range
        look_angle = compute_look_angles_from_orbit(
            orbit=self._orbit,
            azimuth_time=azimuth_time,
            range_times=self.mid_range_time,
            look_direction=self.looking_side.value,
        )
        v_ground = compute_ground_velocity(orbit=self._orbit, time_point=azimuth_time, look_angles=look_angle)
        azimuth_step_m = self.azimuth_step_s * v_ground

        if self.projection == SARProjection.SLANT_RANGE:
            ground_range_step_m: float = self.range_step_m / np.sin(incidence_angle)
        elif self.projection == SARProjection.GROUND_RANGE:
            ground_range_step_m: float = self.range_step_m

        return LocationData(
            abs_azimuth_time=azimuth_time,
            abs_range_time=range_time,
            incidence_angle=incidence_angle,
            look_angle=look_angle,
            ground_velocity=v_ground,
            azimuth_step_m=azimuth_step_m,
            range_step_m=self.range_step_m,
            ground_range_step_m=ground_range_step_m,
        )
