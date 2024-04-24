# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Unittest for quality_input_from_product_folder.py functionalities"""

import unittest
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from arepytools.geometry.curve import Generic3DCurve
from arepytools.geometry.generalsarorbit import GSO3DCurveWrapper
from arepytools.io import (
    create_new_metadata,
    create_product_folder,
    write_metadata,
    write_raster_with_raster_info,
)
from arepytools.io.metadata import (
    AcquisitionTimeLine,
    AttitudeInfo,
    DataSetInfo,
    Pulse,
    RasterInfo,
    SamplingConstants,
    StateVectors,
    SwathInfo,
)
from arepytools.timing.precisedatetime import PreciseDateTime

import arepyextras.quality.core.generic_dataclasses as gdt
from arepyextras.quality.io import quality_input_from_product_folder as qi
from arepyextras.quality.io.quality_input_protocol import (
    ChannelData,
    LocationData,
    QualityInputProduct,
)

warnings.filterwarnings("ignore")

_state_vectors_pos = np.array(
    [
        -2436991.5828411,
        -5204802.0068121,
        3930826.97358041,
        -2437320.89529173,
        -5205114.6965147,
        3930212.37496998,
        -2437650.18445934,
        -5205427.32102359,
        3929597.72839696,
        -2437979.44876892,
        -5205739.87884475,
        3928983.03679951,
        -2438308.68978573,
        -5206052.37146529,
        3928368.29725382,
    ]
).reshape(-1, 3)

_state_vectors_vel = np.array(
    [
        -3293.24796212427,
        -3127.22969476878,
        -6145.75945327732,
        -3293.00730517032,
        -3126.57032088015,
        -6146.22446105275,
        -3292.76659941114,
        -3125.91091234456,
        -6146.689397119,
        -3292.52584599689,
        -3125.2514723148,
        -6147.15425925362,
        -3292.28504378193,
        -3124.59199765488,
        -6147.61904966779,
    ]
).reshape(-1, 3)

_yaw = np.array(
    [
        6.01558980504364e-06,
        1.72212173549011e-05,
        2.15175381458397e-05,
        1.89119065338883e-05,
        9.41149065433075e-06,
        5.92311556040909e-06,
        1.70738023389626e-05,
    ]
)
_pitch = np.array(
    [
        2.76942236127373e-06,
        7.98298153997112e-06,
        9.98141065677743e-06,
        8.76586934113811e-06,
        4.33952823916987e-06,
        2.72025187601975e-06,
        7.93086688889418e-06,
    ]
)
_roll = np.array(
    [
        25.9269737687551,
        25.9265769603434,
        25.9261802440172,
        25.9257836127292,
        25.9253870594373,
        25.9249872414186,
        25.9245819474607,
    ]
)


def _create_product_folder(path: Path) -> None:
    """Creating a dummy product folder for tests

    Parameters
    ----------
    path : Path
        path where the product folder will be created
    """
    pf = create_product_folder(path)
    metadata = create_new_metadata()
    # pf.append_channel(5, 5, "FLOAT32")
    # pf.write_metadata(0)
    swath_info = SwathInfo(swath_i="SwathName", polarization_i="H/V", acquisition_prf_i=2000)
    swath_info.acquisition_start_time = PreciseDateTime.from_utc_string("1 JAN 2021 00:00:00.00000")
    raster_info = RasterInfo(
        filename=pf.get_channel_data(1).name,
        lines=1000,
        samples=1500,
        celltype="FLOAT32",
    )
    raster_info.set_lines_axis(
        lines_start=swath_info.acquisition_start_time, lines_start_unit="Utc", lines_step=1e-2, lines_step_unit="s"
    )
    raster_info.set_samples_axis(
        samples_start=0.0039239928270080006, samples_start_unit="s", samples_step=1e-6, samples_step_unit="s"
    )
    sampling_constants = SamplingConstants(100, 50, 1000, 500)
    state_vect = StateVectors(
        _state_vectors_pos, _state_vectors_vel, PreciseDateTime.from_utc_string("1 JAN 2021 00:00:00.00000"), 5
    )
    dataset_info = DataSetInfo(acquisition_mode_i="TOPSAR", fc_hz_i=9650000000)
    dataset_info.image_type = "AZIMUTH FOCUSED RANGE COMPENSATED"
    dataset_info.description = ""
    dataset_info.sensor_name = ""
    dataset_info.projection = "SLANT RANGE"
    dataset_info.acquisition_station = ""
    dataset_info.processing_center = ""
    dataset_info.processing_software = ""
    dataset_info.side_looking = "RIGHT"
    attitude_info = AttitudeInfo(
        yaw=_yaw,
        pitch=_pitch,
        roll=_roll,
        t0=PreciseDateTime.from_utc_string("1 JAN 2021 00:00:00.00000"),
        delta_t=5,
        ref_frame="GEOCENTRIC",
        rot_order="YPR",
    )
    pulse_info = Pulse(
        i_bandwidth=1,
        i_pulse_energy=1,
        i_pulse_length=1,
        i_pulse_sampling_rate=1,
        i_pulse_start_frequency=1,
        i_pulse_direction=None,
        i_pulse_start_phase=1,
    )
    acquisition_timeline = AcquisitionTimeLine(
        swst_changes_azimuth_times_i=[0, 6], swst_changes_number_i=2, swst_changes_values_i=[1, 1.1]
    )
    metadata.insert_element(swath_info)
    metadata.insert_element(raster_info)
    metadata.insert_element(sampling_constants)
    metadata.insert_element(dataset_info)
    metadata.insert_element(state_vect)
    metadata.insert_element(attitude_info)
    metadata.insert_element(pulse_info)
    metadata.insert_element(acquisition_timeline)

    # Write it
    write_metadata(metadata, pf.get_channel_metadata(1))
    data = np.arange(4).reshape((2, 2))
    write_raster_with_raster_info(pf.get_channel_data(1), data, raster_info, (1, 1))


class ProductFolderManagerTest(unittest.TestCase):
    """Testing ProductFolderManager class"""

    def setUp(self) -> None:
        # creating test data
        self.t_test = PreciseDateTime.from_utc_string("1 JAN 2021 00:00:00.00000") + 6
        self.r_time_test = 0.004673992827008
        self.pos_check = np.array([-2437386.755083514, -5205177.226722712, 3930089.44931395])
        self.vel_check = np.array([-65.85931313052917, -62.52889180621289, -122.92659103337307])
        self.acc_check = np.array([0.0009644373056515517, 0.0026391772443180297, -0.0018567001340109445])
        self.boresight_check_0 = np.array([0.7584196934921817, -0.5798490032009762, -0.2976150231558497])
        self.boresight_check_1 = np.array([2.3520895006515986e-06, 3.160122624121073e-06, -1.630368509755916e-07])
        self.boresight_check_2 = np.array([6.450487976226806e-08, 3.913163091097625e-08, 8.818871801130608e-08])
        self.tol = 1e-9

    def test_init(self) -> None:
        """Testing class initialization"""
        with TemporaryDirectory() as temp_dir:
            pf_path = Path(temp_dir).joinpath("test")
            _create_product_folder(path=pf_path)
            pfm = qi.ProductFolderManager(pf_path)

            assert isinstance(pfm, QualityInputProduct)
            self.assertIsInstance(pfm, qi.ProductFolderManager)
            self.assertEqual(pfm.path, pf_path)
            self.assertEqual(pfm.name, pf_path.name)
            self.assertEqual(pfm.channels_list, [1])

    def test_get_channel(self) -> None:
        """Testing class get_channel_data method"""
        with TemporaryDirectory() as temp_dir:
            pf_path = Path(temp_dir).joinpath("test")
            _create_product_folder(path=pf_path)
            pfm = qi.ProductFolderManager(pf_path)
            channel = pfm.get_channel_data(1)
            location_data = channel.get_location_data(azimuth_time=self.t_test, range_time=self.r_time_test)

            # checking ProductFolderManager
            assert isinstance(pfm, QualityInputProduct)
            self.assertIsInstance(pfm, qi.ProductFolderManager)
            self.assertEqual(pfm.path, pf_path)
            self.assertEqual(pfm.name, pf_path.name)
            self.assertEqual(pfm.channels_list, [1])

            # checking ChannelManager
            assert isinstance(channel, ChannelData)
            self.assertIsInstance(channel, qi.ChannelManager)
            self.assertEqual(channel.swath_name, "SwathName")
            self.assertEqual(channel.channel_id, 1)
            self.assertEqual(channel.range_step_m, 149.896229)
            self.assertEqual(channel.azimuth_step_s, 0.01)
            self.assertIsInstance(channel.projection, gdt.SARProjection)
            self.assertEqual(channel.projection, gdt.SARProjection.SLANT_RANGE)
            self.assertIsInstance(channel.polarization, gdt.SARPolarization)
            self.assertEqual(channel.polarization, gdt.SARPolarization.HV)
            self.assertIsInstance(channel.image_type, gdt.SARImageType)
            self.assertEqual(channel.image_type, gdt.SARImageType.SLC)
            self.assertIsInstance(channel.sampling_constants, gdt.SARSamplingFrequencies)
            self.assertEqual(channel.sampling_constants.range_freq_hz, 100)
            self.assertEqual(channel.sampling_constants.azimuth_freq_hz, 1000)
            self.assertEqual(channel.sampling_constants.range_bandwidth_freq_hz, 50)
            self.assertEqual(channel.sampling_constants.azimuth_bandwidth_freq_hz, 500)
            self.assertIsInstance(channel.looking_side, gdt.SARSideLooking)
            self.assertEqual(channel.looking_side, gdt.SARSideLooking.RIGHT_LOOKING)
            self.assertEqual(channel.carrier_frequency, 9650000000)
            self.assertTrue(np.isnan(channel.pulse_latch_time))
            self.assertIsInstance(channel.swst_changes, list)
            self.assertIsInstance(channel.swst_changes[0], tuple)
            self.assertIsInstance(channel.swst_changes[0][0], PreciseDateTime)
            self.assertIsInstance(channel.swst_changes[0][1], float)
            self.assertEqual(len(channel.swst_changes), 2)
            self.assertAlmostEqual(channel.swst_changes[0][0] - channel._raster_info.lines_start, 0)
            self.assertAlmostEqual(channel.swst_changes[1][0] - self.t_test, 0)
            self.assertEqual(channel.swst_changes[0][1], 1)
            self.assertEqual(channel.swst_changes[1][1], 1.1)
            self.assertAlmostEqual(
                channel.mid_azimuth_time - PreciseDateTime.from_utc_string("01-JAN-2021 00:00:05.000000000000"), 0
            )
            self.assertAlmostEqual(channel.mid_range_time, 0.004673492827008001)
            self.assertIsInstance(channel.trajectory, GSO3DCurveWrapper)
            self.assertIsInstance(channel.boresight_normal_curve, Generic3DCurve)

            np.testing.assert_allclose(channel.trajectory.evaluate(self.t_test), self.pos_check, atol=self.tol, rtol=0)
            np.testing.assert_allclose(
                channel.trajectory.evaluate_first_derivatives(self.t_test), self.vel_check, atol=self.tol, rtol=0
            )
            np.testing.assert_allclose(
                channel.trajectory.evaluate_second_derivatives(self.t_test), self.acc_check, atol=self.tol, rtol=0
            )

            np.testing.assert_allclose(
                channel.boresight_normal_curve.evaluate(self.t_test), self.boresight_check_0, atol=self.tol, rtol=0
            )
            np.testing.assert_allclose(
                channel.boresight_normal_curve.evaluate_first_derivatives(self.t_test),
                self.boresight_check_1,
                atol=self.tol,
                rtol=0,
            )
            np.testing.assert_allclose(
                channel.boresight_normal_curve.evaluate_second_derivatives(self.t_test),
                self.boresight_check_2,
                atol=self.tol,
                rtol=0,
            )

            # test location data
            self.assertIsInstance(location_data, LocationData)

    def test_channel_methods(self) -> None:
        """Testing channel methods"""

        with TemporaryDirectory() as temp_dir:
            pf_path = Path(temp_dir).joinpath("test")
            _create_product_folder(path=pf_path)
            pfm = qi.ProductFolderManager(pf_path)
            channel = pfm.get_channel_data(1)
            channel._raster_info.set_lines_axis(self.t_test, "s", 1, "s")
            channel._raster_info.set_samples_axis(0, "s", self.r_time_test / 2, "s")

            pixel = channel.times_to_pixel_conversion(azimuth_time=self.t_test + 2, range_time=self.r_time_test)
            self.assertTupleEqual(pixel, (2, 2))

            times = channel.pixel_to_times_conversion(azimuth_index=5.36, range_index=5.36)
            np.testing.assert_allclose(
                abs(times[0] - PreciseDateTime.from_utc_string("01-JAN-2021 00:00:11.360000000000")),
                0,
                atol=self.tol,
                rtol=0,
            )
            np.testing.assert_allclose(times[1], 0.012526300776381442, atol=self.tol, rtol=0)

    def test_get_location_data(self) -> None:
        """Testing channel get_location_data method"""

        with TemporaryDirectory() as temp_dir:
            pf_path = Path(temp_dir).joinpath("test")
            _create_product_folder(path=pf_path)
            pfm = qi.ProductFolderManager(pf_path)
            channel = pfm.get_channel_data(1)
            loc_data = channel.get_location_data(azimuth_time=self.t_test, range_time=self.r_time_test)

            assert isinstance(pfm, QualityInputProduct)
            self.assertIsInstance(pfm, qi.ProductFolderManager)
            assert isinstance(channel, ChannelData)
            self.assertIsInstance(channel, qi.ChannelManager)
            self.assertIsInstance(loc_data, LocationData)

            self.assertIsInstance(loc_data.abs_azimuth_time, PreciseDateTime)
            self.assertEqual(loc_data.abs_azimuth_time, self.t_test)
            self.assertIsInstance(loc_data.abs_range_time, float)
            self.assertEqual(loc_data.abs_range_time, self.r_time_test)
            np.testing.assert_allclose(loc_data.incidence_angle, 0.5938752890629725, atol=self.tol, rtol=0)
            np.testing.assert_allclose(loc_data.azimuth_step_m, 1.3902222530090496, atol=self.tol, rtol=0)
            np.testing.assert_allclose(loc_data.range_step_m, 149.896229, atol=self.tol, rtol=0)
            np.testing.assert_allclose(loc_data.ground_range_step_m, 267.87419133531984, atol=self.tol, rtol=0)


if __name__ == "__main__":
    unittest.main()
