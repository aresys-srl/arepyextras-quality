# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Unittest for radiometric_analysis/support.py core functionalities"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from arepytools.timing.precisedatetime import PreciseDateTime
from netCDF4 import Dataset

import arepyextras.quality.radiometric_analysis.support as support
from arepyextras.quality.core.generic_dataclasses import (
    SARPolarization,
    SARRadiometricQuantity,
)
from arepyextras.quality.radiometric_analysis.config import (
    Radiometric2DHistogramParameters,
)
from arepyextras.quality.radiometric_analysis.custom_dataclasses import (
    RadiometricAnalysisDirection,
    RadiometricProfilesOutput,
)


class MockTrajectory:
    """Mocking trajectory class"""

    def evaluate(self, time) -> np.ndarray:
        """Mocking position interpolation"""
        return [5634298.570491991, -4277813.834855013, 183850.74790036504]

    def evaluate_first_derivatives(self, time) -> np.ndarray:
        """Mocking velocity interpolation"""
        return [-797.011102366091, -1383.8309567802658, -7427.764230040876]


class AnglesComputationSetupTest(unittest.TestCase):
    """Testing radiometric_analysis/support.py angles_computation_setup function"""

    def setUp(self) -> None:
        # reference results
        self.tolerance = 1e-9
        self._trajectory = MockTrajectory()
        self._az_time = PreciseDateTime.from_utc_string("05-JAN-2017 08:29:41.068885794433")
        self._rng_values = np.array(
            [
                0.004975715993854,
                0.004975729022631,
                0.004975742051408,
                0.004975755080185,
                0.004975768108962,
                0.004975781137739,
                0.004975794166516,
                0.004975807195293,
                0.00497582022407,
                0.004975833252847,
                0.004975846281624,
                0.004975859310401,
                0.004975872339178,
                0.004975885367955,
                0.004975898396732,
                0.004975911425509,
                0.004975924454286,
                0.004975937483063001,
                0.004975950511840001,
                0.004975963540617001,
                0.004975976569394001,
            ]
        )
        # expected results
        self._ref_sensor_pos = [5634298.570491991, -4277813.834855013, 183850.74790036504]
        self._ref_ground_points = np.array(
            [
                4.926391951508208178e06,
                -4.045258062462532893e06,
                2.164839266398744367e05,
                4.926388638358334079e06,
                -4.045262038216231391e06,
                2.164850228497198550e05,
                4.926385325266964734e06,
                -4.045266013892872725e06,
                2.164861190389312687e05,
                4.926382012234100141e06,
                -4.045269989492459223e06,
                2.164872152075095219e05,
                4.926378699259732850e06,
                -4.045273965014998335e06,
                2.164883113554565352e05,
                4.926375386343861930e06,
                -4.045277940460492857e06,
                2.164894074827730074e05,
                4.926372073486481793e06,
                -4.045281915828950703e06,
                2.164905035894609464e05,
                4.926368760687589645e06,
                -4.045285891120372806e06,
                2.164915996755208762e05,
                4.926365447947181761e06,
                -4.045289866334765218e06,
                2.164926957409543102e05,
                4.926362135265253484e06,
                -4.045293841472133063e06,
                2.164937917857626453e05,
                4.926358822641801089e06,
                -4.045297816532485187e06,
                2.164948878099481226e05,
                4.926355510076822713e06,
                -4.045301791515820194e06,
                2.164959838135105674e05,
                4.926352197570312768e06,
                -4.045305766422146000e06,
                2.164970797964519879e05,
                4.926348885122268461e06,
                -4.045309741251465864e06,
                2.164981757587734028e05,
                4.926345572732684202e06,
                -4.045313716003788635e06,
                2.164992717004768783e05,
                4.926342260401559062e06,
                -4.045317690679115243e06,
                2.165003676215629093e05,
                4.926338948128886521e06,
                -4.045321665277452208e06,
                2.165014635220333003e05,
                4.926335635914664716e06,
                -4.045325639798806049e06,
                2.165025594018894772e05,
                4.926332323758888990e06,
                -4.045329614243176300e06,
                2.165036552611317602e05,
                4.926329011661556549e06,
                -4.045333588610572275e06,
                2.165047510997624195e05,
                4.926325699622661807e06,
                -4.045337562901000027e06,
                2.165058469177829102e05,
            ]
        ).reshape(-1, 3)
        self._ref_nadir = np.array([-556144.218795416, 422249.79801344173, -18257.49818938377])

    def test_angles_computation_setup(self):
        """Testing angles_computation_setup"""
        sensor_pos, ground_points, nadir = support.angles_computation_setup(
            trajectory=self._trajectory,
            azimuth_time=self._az_time,
            look_direction="RIGHT",
            range_values=self._rng_values,
        )
        np.testing.assert_allclose(sensor_pos, self._ref_sensor_pos, atol=self.tolerance, rtol=0)
        np.testing.assert_allclose(ground_points, self._ref_ground_points, atol=self.tolerance, rtol=0)
        np.testing.assert_allclose(nadir, self._ref_nadir, atol=self.tolerance, rtol=0)


class BlocksDefinitionTest(unittest.TestCase):
    """Testing radiometric_analysis/support.py blocks_definition function"""

    def setUp(self) -> None:
        # creating test data
        self._az_axis = np.zeros(2500)
        self._rng_axis = np.zeros(4500)
        self._lines_per_burst = np.array([300] * 5)
        self._default_block_size = 2000
        self.expected_res_0 = [300, 5, [(150, 2250), (450, 2250), (750, 2250), (1050, 2250), (1350, 2250)]]
        self.expected_res_1 = [2000, 1, [(1000, 2250)]]

    def test_blocks_definition_0(self):
        """Testing blocks_definition function"""
        blocks_data = support.blocks_definition(
            azimuth_axis=self._az_axis,
            default_block_size=self._default_block_size,
            lines_per_burst=self._lines_per_burst,
            range_axis=self._rng_axis,
        )
        self.assertListEqual(list(blocks_data), self.expected_res_0)

    def test_blocks_definition_1(self):
        """Testing blocks_definition function"""
        blocks_data = support.blocks_definition(
            azimuth_axis=self._az_axis,
            default_block_size=self._default_block_size,
            lines_per_burst=np.array([self._lines_per_burst[0]]),
            range_axis=self._rng_axis,
        )
        self.assertListEqual(list(blocks_data), self.expected_res_1)


class Compute2DHistogramTest(unittest.TestCase):
    """Testing radiometric_analysis/support.py compute_2d_histogram function"""

    def setUp(self) -> None:
        # reference results
        self.tolerance = 1e-9
        self._ref_hist = np.stack(
            [
                [0, 0, 0, 0],
                [0, 0, 189, 17],
                [0, 0, 0, 0],
            ]
        )
        self._ref_x_bins = np.array([0.0, 0.975609756097561, 1.951219512195122, 2.926829268292683, 3.902439024390244])
        self._ref_y_bins = np.array([1.9841646002547808, 8.650831266921447, 15.317497933588115, 21.98416460025478])

    def test_compute_2d_histogram(self):
        """Testing compute_2d_histogram function"""
        random_rng1 = np.random.default_rng(12345)
        hist, x_bins, y_bins = support.compute_2d_histogram(
            x_data=random_rng1.random((206)) + random_rng1.integers(2, 6),
            y_data=random_rng1.random((206)) * 4 + 10,
            x_axis=np.linspace(0, 4, 206),
            config=Radiometric2DHistogramParameters(y_bins_center_margin=10, y_bins_num=4, x_bins_step=50),
        )
        np.testing.assert_allclose(hist, self._ref_hist, atol=self.tolerance, rtol=0)
        np.testing.assert_allclose(x_bins, self._ref_x_bins, atol=self.tolerance, rtol=0)
        np.testing.assert_allclose(y_bins, self._ref_y_bins, atol=self.tolerance, rtol=0)


class MaskingOutliersByPercentileTest(unittest.TestCase):
    """Testing radiometric_analysis/support.py masking_outliers_by_percentiles function"""

    def setUp(self) -> None:
        # reference results
        self._ref_masked_num = 17

    def test_masking_outliers_by_percentiles(self):
        """Testing masking_outliers_by_percentiles"""
        random_rng1 = np.random.default_rng(123)
        raster = random_rng1.random((15, 15))
        masked = support.masking_outliers_by_percentiles(data=raster, kernel=(5, 5), percentile_boundaries=[20, 80])
        self.assertEqual(np.sum(np.isnan(masked)), self._ref_masked_num)


class RadiometricProfilesToNetCDF(unittest.TestCase):
    """Testing radiometric_analysis/support.py radiometric_profiles_to_netcdf function"""

    def test_radiometric_profiles_to_netcdf(self):
        """Testing radiometric_profiles_to_netcdf function"""
        with TemporaryDirectory() as temp_dir:
            out_fldr = Path(temp_dir).joinpath("out")
            out_fldr.mkdir()
            tag = "test"
            data = RadiometricProfilesOutput(
                swath="S1",
                channel=1,
                blocks_num=3,
                direction=RadiometricAnalysisDirection.RANGE,
                azimuth_block_centers=np.array(
                    [
                        PreciseDateTime.from_numeric_datetime(2020, 1, 1),
                        PreciseDateTime.from_numeric_datetime(2020, 1, 2),
                        PreciseDateTime.from_numeric_datetime(2020, 1, 3),
                    ]
                ),
                range_block_centers=np.array([250, 250, 250]),
                azimuth_start_time=PreciseDateTime.from_numeric_datetime(2020, 1, 1),
                hist_2d=np.array([[0, 0, 0], [0, 10, 0], [1, 2, 0]]).reshape(3, 3),
                block_azimuth_times=np.tile(np.linspace(9, 13, 10), 3).reshape((3, 10)),
                hist_x_bins_axis=np.ones(10),
                hist_y_bins_axis=np.ones(10),
                look_angles=np.ones((3, 10)),
                output_radiometric_quantity=SARRadiometricQuantity.BETA_NOUGHT,
                polarization=SARPolarization.HH,
                profiles=np.ones((3, 10)),
            )
            out_file = out_fldr.joinpath(tag + "_profiles_" + data.swath + "_" + data.polarization.name + ".nc")
            support.radiometric_profiles_to_netcdf(data=data, out_path=out_fldr, tag=tag)

            out_file = out_fldr.joinpath(tag + "_profiles_" + data.swath + "_" + data.polarization.name + ".nc")
            # checking results
            self.assertTrue(out_file.exists())
            self.assertTrue(out_file.is_file())
            root = Dataset(out_file, "r", format="NETCDF4")
            self.assertEqual(root.swath, data.swath)
            self.assertEqual(root.channel, data.channel)
            self.assertEqual(root.direction, data.direction.name.lower())
            self.assertEqual(root.output_radiometric_quantity, data.output_radiometric_quantity.name)
            self.assertEqual(root.azimuth_blocks_num, data.blocks_num)
            self.assertListEqual(root.azimuth_block_centers, [str(d) for d in data.azimuth_block_centers])
            np.testing.assert_array_equal(root.range_block_centers, data.range_block_centers)
            np.testing.assert_array_equal(root.variables["look_angles"][:].data, data.look_angles)
            np.testing.assert_array_equal(root.variables["radiometric_profiles"][:].data, data.profiles)
            np.testing.assert_array_equal(root.variables["azimuth_times"][:].data, data.block_azimuth_times)
            root.close()


if __name__ == "__main__":
    unittest.main()
