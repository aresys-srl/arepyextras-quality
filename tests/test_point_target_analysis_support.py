# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Unittest for point_target_analysis/support.py core functionalities"""

import unittest

import numpy as np
import pandas as pd
from arepytools.io.io_support import NominalPointTarget
from arepytools.timing.precisedatetime import PreciseDateTime

import arepyextras.quality.point_targets_analysis.support as support
from arepyextras.quality.core.generic_dataclasses import SARPolarization

REF_TIME = PreciseDateTime.from_utc_string("15-JAN-2019 16:37:12.051461300098")
REF_GROUND_POINT = np.array([-4989394.044, 2746844.389, -2862070.09])
REF_POINTS = {
    "0": NominalPointTarget(
        xyz_coordinates=np.array([4921229.04081908, -4051559.15884936, 216078.76707954]),
        rcs_hh=(100000 + 0j),
        rcs_vv=0j,
        rcs_vh=(100000 + 0j),
        rcs_hv=0j,
        delay=None,
    ),
    "1": NominalPointTarget(
        xyz_coordinates=np.array([4832296.19624738, -4155847.75546086, 241004.24360898]),
        rcs_hh=(100000 + 0j),
        rcs_vv=0j,
        rcs_vh=(100000 + 0j),
        rcs_hv=0j,
        delay=None,
    ),
    "2": NominalPointTarget(
        xyz_coordinates=np.array([4891219.45186627, -4087200.87719583, 225939.83847657]),
        rcs_hh=(100000 + 0j),
        rcs_vv=0j,
        rcs_vh=(100000 + 0j),
        rcs_hv=0j,
        delay=None,
    ),
}


class MockTrajectory:
    """Mocking trajectory class"""

    def evaluate(self, time) -> np.ndarray:
        """Mocking position interpolation"""
        out = [-381087.525550857, 932485.770149446, -7007146.93083064]
        if np.size(time) == 1:
            return np.array(out)
        return np.stack([out] * np.size(time))

    def evaluate_first_derivatives(self, time) -> np.ndarray:
        """Mocking velocity interpolation"""
        out = [7057.60934660782, 2768.35602191122, -0.259400938909807]
        if np.size(time) == 1:
            return np.array(out)
        return np.stack([out] * np.size(time))


class MockChannelData:
    """Mocking ChannelData class"""

    def __init__(self, channel_id: int = 1) -> None:
        self.channel_id = channel_id
        self._trajectory = MockTrajectory()

    @property
    def swath_name(self) -> str:
        """Mocking swath name"""
        return f"S{self.channel_id}"

    @property
    def polarization(self) -> str:
        """Mocking polarization"""
        return SARPolarization.HH

    @property
    def trajectory(self) -> MockTrajectory:
        """Exposing mock trajectory"""
        return self._trajectory

    @property
    def carrier_frequency(self) -> float:
        """Exposing mock carrier_frequency"""
        return 5405000000

    def ground_points_to_burst_association(self, coordinates: np.ndarray) -> list:
        """Mocking ground_points_to_burst_association function"""
        if self.channel_id == 1:
            return [[0], None, [1]]
        return [None, [0], None]


class MockProduct:
    """Mocking Product class"""

    @property
    def channels_list(self) -> list[int]:
        """Mocking channels list"""
        return [1, 2]

    def get_channel_data(self, channel_id: int) -> MockChannelData:
        """Mocking get channel data"""
        return MockChannelData(channel_id=channel_id)


class GetSquintAngleTest(unittest.TestCase):
    """Testing point_target_analysis/support.py get_squint_angle function"""

    def setUp(self) -> None:
        # creating test data
        self.tolerance = 1e-9
        self.expected_res = -0.5964450004813256

    def test_get_squint_angle(self):
        """Testing get_squint_angle function"""
        squint = support.get_squint_angle(
            channel_data=MockChannelData(), azimuth_time=REF_TIME, ground_point=REF_GROUND_POINT
        )
        np.testing.assert_allclose(squint, self.expected_res, atol=self.tolerance, rtol=0)

    def test_get_squint_angle_vectorized_0(self):
        """Testing get_squint_angle function, vectorized ground points"""
        squint = support.get_squint_angle(
            channel_data=MockChannelData(),
            azimuth_time=REF_TIME,
            ground_point=np.tile(REF_GROUND_POINT, 4).reshape(-1, 3),
        )
        np.testing.assert_allclose(squint, np.repeat(self.expected_res, 4), atol=self.tolerance, rtol=0)

    def test_get_squint_angle_vectorized_1(self):
        """Testing get_squint_angle function, vectorized times"""
        squint = support.get_squint_angle(
            channel_data=MockChannelData(), azimuth_time=np.array([REF_TIME, REF_TIME]), ground_point=REF_GROUND_POINT
        )
        np.testing.assert_allclose(squint, np.repeat(self.expected_res, 2), atol=self.tolerance, rtol=0)

    def test_get_squint_angle_vectorized_2(self):
        """Testing get_squint_angle function, vectorized all"""
        squint = support.get_squint_angle(
            channel_data=MockChannelData(),
            azimuth_time=np.array([REF_TIME, REF_TIME, REF_TIME, REF_TIME]),
            ground_point=np.tile(REF_GROUND_POINT, 4).reshape(-1, 3),
        )
        np.testing.assert_allclose(squint, np.repeat(self.expected_res, 4), atol=self.tolerance, rtol=0)


class GetDopplerCentroidTest(unittest.TestCase):
    """Testing point_target_analysis/support.py get_doppler_centroid function"""

    def setUp(self) -> None:
        # creating test data
        self.tolerance = 1e-9
        self.expected_res = -153549.19005492592

    def test_get_doppler_centroid(self):
        """Testing get_doppler_centroid function"""
        dc = support.get_doppler_centroid(
            channel_data=MockChannelData(), azimuth_time=REF_TIME, ground_point=REF_GROUND_POINT
        )
        np.testing.assert_allclose(dc, self.expected_res, atol=self.tolerance, rtol=0)

    def test_get_doppler_centroid_vectorized_0(self):
        """Testing get_doppler_centroid function, vectorized ground points"""
        dc = support.get_doppler_centroid(
            channel_data=MockChannelData(),
            azimuth_time=REF_TIME,
            ground_point=np.tile(REF_GROUND_POINT, 4).reshape(-1, 3),
        )
        np.testing.assert_allclose(dc, np.repeat(self.expected_res, 4), atol=self.tolerance, rtol=0)

    def test_get_doppler_centroid_vectorized_1(self):
        """Testing get_doppler_centroid function, vectorized times"""
        dc = support.get_doppler_centroid(
            channel_data=MockChannelData(), azimuth_time=np.array([REF_TIME, REF_TIME]), ground_point=REF_GROUND_POINT
        )
        np.testing.assert_allclose(dc, np.repeat(self.expected_res, 2), atol=self.tolerance, rtol=0)

    def test_get_doppler_centroid_vectorized_2(self):
        """Testing get_doppler_centroid function, vectorized all"""
        dc = support.get_doppler_centroid(
            channel_data=MockChannelData(),
            azimuth_time=np.array([REF_TIME, REF_TIME, REF_TIME, REF_TIME]),
            ground_point=np.tile(REF_GROUND_POINT, 4).reshape(-1, 3),
        )
        np.testing.assert_allclose(dc, np.repeat(self.expected_res, 4), atol=self.tolerance, rtol=0)


class CheckTargetsVisibilityTest(unittest.TestCase):
    """Testing point_target_analysis/support.py check_targets_visibility function"""

    def setUp(self) -> None:
        # creating test data
        data_dict = {
            "id": [0, 1, 2, 0, 1, 2],
            "channel": [1, 1, 1, 2, 2, 2],
            "burst": [[0], None, [1], None, [0], None],
            "swath": ["S1", "S1", "S1", "S2", "S2", "S2"],
            "polarization": ["HH"] * 6,
        }
        self.reference_df = pd.DataFrame(data=data_dict)

    def test_check_targets_visibility(self):
        """Testing check_targets_visibility"""
        targets_visibility = support.check_targets_visibility(product=MockProduct(), points=REF_POINTS)
        targets_visibility["id"] = targets_visibility["id"].astype("int64")
        pd.testing.assert_frame_equal(targets_visibility, self.reference_df)


if __name__ == "__main__":
    unittest.main()
