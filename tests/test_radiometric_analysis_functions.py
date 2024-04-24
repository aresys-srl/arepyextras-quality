# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Unittest for radiometric_analysis core functionalities"""

import unittest

import numpy as np

import arepyextras.quality.radiometric_analysis.support as support
from arepyextras.quality.radiometric_analysis.config import (
    Radiometric2DHistogramParameters,
)


class TestRadiometricAnalysisFunctions(unittest.TestCase):
    """Testing Radiometric Analysis support functionalities"""

    def setUp(self) -> None:
        """Testing setup"""
        self._test_array = np.array(
            [
                [56, 79, 1, 25, 19, 13, 16, 67, 67, 24, 88, 16, 46, 31, 19, 50, 46, 9, 12, 64, 9, 93, 40, 25, 37, 58],
                [44, 6, 46, 14, 25, 91, 72, 79, 73, 25, 89, 77, 12, 74, 75, 28, 25, 24, 60, 17, 84, 43, 21, 63, 89, 89],
                [42, 25, 3, 86, 86, 83, 28, 19, 37, 87, 59, 49, 85, 10, 73, 58, 68, 66, 93, 98, 22, 59, 9, 53, 97, 65],
                [58, 37, 34, 24, 54, 49, 54, 34, 40, 23, 83, 32, 20, 57, 41, 25, 94, 90, 96, 64, 47, 58, 62, 5, 11, 19],
                [96, 42, 63, 22, 82, 97, 27, 77, 27, 12, 56, 18, 36, 76, 43, 30, 96, 61, 34, 57, 95, 95, 4, 29, 44, 15],
                [34, 78, 79, 18, 71, 32, 55, 64, 95, 49, 11, 2, 83, 43, 49, 29, 39, 93, 63, 2, 23, 13, 82, 30, 12, 33],
                [90, 36, 76, 70, 68, 0, 82, 52, 19, 15, 47, 83, 75, 48, 65, 66, 80, 82, 0, 69, 49, 82, 79, 95, 19, 25],
                [27, 90, 16, 61, 93, 95, 87, 19, 6, 10, 50, 76, 12, 72, 88, 97, 26, 9, 24, 36, 28, 14, 12, 67, 94, 46],
                [63, 16, 63, 97, 36, 11, 87, 8, 3, 42, 96, 1, 62, 35, 93, 19, 31, 70, 52, 98, 8, 72, 43, 34, 21, 26],
                [11, 33, 57, 54, 94, 51, 64, 42, 16, 82, 92, 6, 84, 93, 86, 60, 13, 24, 98, 40, 99, 76, 7, 3, 73, 0],
                [64, 55, 57, 22, 16, 46, 71, 78, 2, 12, 75, 32, 7, 31, 47, 32, 98, 3, 15, 26, 45, 18, 19, 0, 47, 71],
                [8, 48, 58, 52, 11, 65, 71, 3, 74, 32, 47, 52, 50, 30, 71, 98, 79, 84, 92, 53, 48, 93, 58, 23, 87, 32],
                [39, 84, 4, 9, 12, 18, 87, 89, 57, 49, 14, 99, 69, 30, 22, 85, 22, 97, 61, 26, 28, 81, 89, 83, 92, 36],
                [44, 81, 0, 82, 93, 55, 12, 98, 95, 32, 81, 44, 47, 39, 75, 34, 5, 43, 49, 62, 83, 81, 70, 53, 20, 65],
                [37, 72, 56, 56, 33, 15, 66, 49, 49, 37, 43, 30, 7, 70, 35, 72, 58, 50, 90, 29, 1, 22, 53, 28, 97, 29],
                [98, 76, 92, 44, 58, 50, 97, 67, 28, 22, 21, 75, 98, 73, 74, 78, 97, 42, 88, 25, 35, 39, 22, 89, 9, 3],
                [93, 5, 28, 85, 85, 15, 67, 54, 21, 67, 46, 46, 65, 76, 73, 10, 64, 49, 14, 38, 99, 24, 87, 55, 67, 80],
                [11, 44, 19, 42, 88, 57, 38, 10, 80, 45, 49, 48, 97, 5, 36, 98, 3, 85, 42, 25, 12, 80, 77, 71, 45, 24],
                [35, 24, 10, 70, 44, 66, 89, 99, 67, 89, 96, 99, 72, 57, 84, 14, 15, 95, 0, 30, 62, 24, 23, 39, 92, 71],
                [20, 2, 10, 41, 62, 81, 29, 2, 86, 83, 65, 76, 72, 82, 36, 63, 20, 99, 33, 32, 93, 2, 51, 15, 55, 20],
                [19, 55, 72, 38, 70, 32, 9, 45, 76, 83, 44, 38, 0, 38, 63, 3, 99, 86, 36, 50, 38, 57, 66, 11, 98, 80],
                [55, 85, 1, 87, 32, 4, 8, 25, 8, 87, 58, 39, 73, 63, 23, 0, 94, 50, 61, 32, 72, 87, 32, 78, 95, 84],
                [33, 46, 64, 69, 54, 16, 84, 97, 18, 2, 61, 43, 40, 53, 12, 80, 32, 66, 56, 68, 16, 10, 25, 12, 31, 49],
                [73, 52, 41, 99, 73, 10, 36, 51, 17, 58, 99, 82, 56, 46, 20, 6, 56, 89, 60, 4, 63, 18, 64, 16, 44, 94],
                [26, 5, 74, 65, 18, 30, 99, 5, 17, 80, 86, 67, 90, 72, 68, 95, 26, 23, 97, 33, 63, 37, 26, 75, 21, 61],
                [3, 93, 80, 85, 14, 27, 89, 7, 69, 76, 95, 18, 13, 61, 84, 27, 37, 27, 62, 52, 29, 87, 39, 55, 66, 62],
                [72, 0, 91, 48, 72, 50, 45, 12, 19, 92, 30, 27, 48, 11, 12, 13, 81, 5, 26, 70, 50, 78, 13, 90, 42, 61],
            ],
            dtype=float,
        )
        # expected results
        self._masking_results = 66
        self._hist = np.array(
            [
                [
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    1.0,
                    0.0,
                    3.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    2.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                    0.0,
                    2.0,
                    0.0,
                ],
                [
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    2.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    2.0,
                    0.0,
                    0.0,
                    2.0,
                    1.0,
                    5.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    1.0,
                    0.0,
                    1.0,
                    2.0,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    1.0,
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                    3.0,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    2.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    2.0,
                    0.0,
                ],
                [
                    1.0,
                    1.0,
                    0.0,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    2.0,
                    2.0,
                    0.0,
                    3.0,
                    0.0,
                    0.0,
                    2.0,
                    2.0,
                    0.0,
                    0.0,
                    2.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    2.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    1.0,
                    1.0,
                    2.0,
                    0.0,
                    2.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    2.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    2.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    2.0,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    2.0,
                    0.0,
                    0.0,
                ],
                [
                    1.0,
                    1.0,
                    0.0,
                    2.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    2.0,
                    1.0,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                ],
            ]
        )
        self._x_bins = np.array(
            [
                0.0,
                4.0,
                9.0,
                12.0,
                14.0,
                18.0,
                19.0,
                23.0,
                25.0,
                27.0,
                30.0,
                34.0,
                37.0,
                42.0,
                46.0,
                49.0,
                53.0,
                58.0,
                61.0,
                64.0,
                67.0,
                72.0,
                76.0,
                79.0,
                83.0,
                87.0,
                90.0,
                93.0,
                96.0,
                98.0,
            ]
        )
        self._y_bins = np.array(
            [
                20.935897435897438,
                27.602564102564102,
                34.269230769230774,
                40.93589743589744,
                47.6025641025641,
                54.26923076923077,
                60.93589743589744,
                67.6025641025641,
                74.26923076923077,
                80.93589743589743,
            ]
        )

    def test_mask_outlier_by_percentile(self) -> None:
        """Testing masking outliers"""
        kernel = [5, 5]
        masked = support.masking_outliers_by_percentiles(
            data=self._test_array, kernel=kernel, percentile_boundaries=[15, 75]
        )
        self.assertEqual(np.sum(np.isnan(masked)), self._masking_results)

    def test_histogram_2d(self) -> None:
        """Testing histogram 2D generation"""
        config = Radiometric2DHistogramParameters()
        config.x_bins_step = 8
        config.y_bins_center_margin = 30
        config.y_bins_num = 10
        data_split = np.array_split(self._test_array, 3)
        hist, x_bins, y_bins = support.compute_2d_histogram(
            x_axis=np.sort(data_split[0].ravel()),
            y_data=data_split[1].ravel(),
            x_data=data_split[2].ravel(),
            config=config,
        )
        np.testing.assert_array_equal(x_bins, self._x_bins)
        np.testing.assert_array_equal(y_bins, self._y_bins)
        np.testing.assert_array_equal(hist, self._hist)

    def test_block_definition_0(self) -> None:
        """Testing block definition, case 0"""
        az_axis = np.zeros(800)
        rng_axis = np.zeros(1200)
        block_size = 200
        lines_per_burst = np.array([600])
        blk_sz, blk_num, blk_centers = support.blocks_definition(
            azimuth_axis=az_axis, range_axis=rng_axis, default_block_size=block_size, lines_per_burst=lines_per_burst
        )
        self.assertEqual(block_size, blk_sz)
        self.assertEqual(blk_num, az_axis.size // block_size)
        self.assertListEqual(
            blk_centers, [(block_size // 2 + block_size * n, rng_axis.size // 2) for n in range(blk_num)]
        )

    def test_block_definition_1(self) -> None:
        """Testing block definition, case 1"""
        az_axis = np.zeros(800)
        rng_axis = np.zeros(1200)
        block_size = 200
        lines_per_burst = np.array([300, 300])
        blk_sz, blk_num, blk_centers = support.blocks_definition(
            azimuth_axis=az_axis, range_axis=rng_axis, default_block_size=block_size, lines_per_burst=lines_per_burst
        )
        self.assertFalse(block_size == blk_sz)
        self.assertEqual(blk_sz, lines_per_burst[0])
        self.assertEqual(blk_num, lines_per_burst.size)
        self.assertListEqual(blk_centers, [(b * n + b // 2, rng_axis.size // 2) for n, b in enumerate(lines_per_burst)])


if __name__ == "__main__":
    unittest.main()
