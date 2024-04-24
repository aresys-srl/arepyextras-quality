# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Definition of a protocol for input product folder structure for Arepyextras Quality"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
from arepytools.geometry.curve import Generic3DCurve
from arepytools.geometry.curve_protocols import TwiceDifferentiable3DCurve
from arepytools.io.metadata import PreciseDateTime

from arepyextras.quality.core.generic_dataclasses import (
    LocationData,
    SARImageType,
    SAROrbitDirection,
    SARPolarization,
    SARProjection,
    SARSamplingFrequencies,
    SARSideLooking,
)


@runtime_checkable
class SARCoordinatesFunction(Protocol):
    """Protocol to define a function taking SAR coordinates (Azimuth, Range) as inputs and returns a float
    This can be any generic f: SAR Times -> R.
    """

    def evaluate(self, azimuth_time: PreciseDateTime, range_time: float) -> float:
        """Evaluate the wrapped function at given azimuth and range times.

        Parameters
        ----------
        azimuth_time : PreciseDateTime
            azimuth time at which evaluate the function
        range_time : float
            range time at which evaluate the function

        Returns
        -------
        float
            output of the wrapped function
        """


@runtime_checkable
class QualityInputProduct(Protocol):
    """Protocol to define characteristics of input product for quality tool"""

    @property
    def name(self) -> str:
        """Get product name"""

    @property
    def channels_list(self) -> list[int] | list[str]:
        """Get list of available channels for this product"""

    def get_channel_data(self, channel_id: int | str) -> ChannelData:
        """Gathering all the information that are channel dependent and storing them in a protocol compliant object.

        Parameters
        ----------
        channel_id : int | str
            selected channel identifier

        Returns
        -------
        ChannelData
            ChannelData-compliant object containing data corresponding to the selected channel
        """


@runtime_checkable
class ChannelData(Protocol):
    """Protocol to define an object that contains all the channel dependent info and data"""

    @property
    def swath_name(self) -> str:
        """Name of the swath being analyzed"""

    @property
    def channel_id(self) -> int | str:
        """Identifier corresponding to the current channel data"""

    @property
    def range_step_m(self) -> float:
        """Step along range direction, in meters"""

    @property
    def azimuth_step_s(self) -> float:
        """Step along azimuth direction, in seconds"""

    @property
    def projection(self) -> SARProjection:
        """Channel data projection"""

    @property
    def polarization(self) -> SARPolarization:
        """Channel data polarization"""

    @property
    def orbit_direction(self) -> SAROrbitDirection:
        """Channel data orbit direction"""

    @property
    def image_type(self) -> SARImageType:
        """Channel raster image type"""

    @property
    def sampling_constants(self) -> SARSamplingFrequencies:
        """Channel data sampling constants"""

    @property
    def pulse_rate(self) -> float:
        """Signal pulse rate (bandwidth / pulse length)"""

    @property
    def looking_side(self) -> SARSideLooking:
        """Sensor look direction for this channel"""

    @property
    def carrier_frequency(self) -> float:
        """Signal carrier frequency"""

    @property
    def mid_azimuth_time(self) -> PreciseDateTime:
        """Azimuth time at half swath"""

    @property
    def mid_range_time(self) -> float:
        """Range time at half swath"""

    @property
    def trajectory(self) -> TwiceDifferentiable3DCurve:
        """Channel trajectory/orbit"""

    @property
    def boresight_normal_curve(self) -> Generic3DCurve | None:
        """Channel attitude boresight normal 3D curve"""

    @property
    def doppler_centroid(self) -> SARCoordinatesFunction | None:
        """Channel doppler centroid polynomial wrapper"""

    @property
    def doppler_rate(self) -> SARCoordinatesFunction | None:
        """Channel doppler rate polynomial wrapper"""

    @property
    def range_axis(self) -> np.ndarray:
        """Range axis"""

    @property
    def slant_range_axis(self) -> np.ndarray:
        """Slant range axis"""

    @property
    def azimuth_axis(self) -> np.ndarray:
        """Azimuth axis, PreciseDateTime format"""

    @property
    def lines_per_burst(self) -> np.ndarray:
        """Lines per burst array, a value for each burst in the swath"""

    @property
    def pulse_latch_time(self) -> float:
        """Signal pulse latch time"""

    @property
    def swst_changes(self) -> list[tuple[PreciseDateTime, float]]:
        """SWST changes list as tuple of time of change and new SWST value"""

    def get_mid_burst_times(self, burst: int) -> tuple[PreciseDateTime, float] | tuple[None, None]:
        """Compute mid azimuth and range times for a given burst.

        Returns
        -------
        tuple[PreciseDateTime, float] | tuple[None, None]
            azimuth and range mid burst times, (None, None) if no bursts
        """

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

    def ground_points_to_burst_association(self, coordinates: npt.ArrayLike) -> list[list[int] | None]:
        """Determining the burst (or bursts) where the input coordinates lie. If no association can be found (i.e. the
        point is not visible in the scene), None is returned.

        Parameters
        ----------
        coordinates : npt.ArrayLike
            array of coordinates, in the form (N, 3)

        Returns
        -------
        list[list[int] | None]
            list containing the burst association for each input point, None if no association was found
        """

    def read_data(self, azimuth_index: int, range_index: int, cropping_size: tuple[int, int]) -> np.ndarray:
        """Extracting the swath portion centered to the provided target position and of size cropping_size by
        cropping_size. Target position is provided via its azimuth and range indexes in the swath array.

        Data block to be read will be assembled this way:
            0. first line to be read: azimuth_index - cropping_size[1] // 2
            1. first sample to be read: range_index - cropping_size[0] // 2
            2. total number of lines to be read: cropping_size[1]
            3. total number of samples to be read: cropping_size[0]

        Parameters
        ----------
        azimuth_index : int
            index of azimuth time in swath array
        range_index : int
            index of range time in swath array
        cropping_size : tuple[int, int]
            size in pixel of the swath portion to be read (number of samples, number of lines)

        Returns
        -------
        np.ndarray
            cropped swath array centered to the input target coordinates, data is provided with shape (samples, lines)
        """

    def get_location_data(
        self,
        azimuth_time: PreciseDateTime,
        range_time: float,
    ) -> LocationData:
        """Generating a LocationManager object containing data and info derived from the current ChannelManager and
        declined to the specific azimuth and range times selected.

        Parameters
        ----------
        azimuth_time : PreciseDateTime
            selected absolute azimuth time
        range_time : float
            selected absolute range time

        Returns
        -------
        LocationData
            LocationData instance related to the selected location
        """
