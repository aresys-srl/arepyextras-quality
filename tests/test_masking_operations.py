# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Unittest for masking_operations.py functions"""

import unittest

import numpy as np

import arepyextras.quality.core.masking_operations as masking
from arepyextras.quality.core.generic_dataclasses import MaskingMethod

from . import _support as test_support

# import _support as test_support


class MaskingOperationsTest(unittest.TestCase):
    """Testing masking_operations.py functions"""

    def setUp(self) -> None:
        # creating test data
        # benchmarking values
        self.reference_values = {
            "lines": 256,
            "samples": 256,
            "rng_resolution": np.array(0.3),
            "az_resolution": np.array(0.4),
            "main_lobe_mask_filling_size_peak": 144,
            "main_lobe_mask_filling_size_peak_lobes": 50,
            "main_lobe_mask_filling_size_res": 120,
            "main_lobe_mask_filling_size_res_lobes": 96,
            "main_lobe_mask_lobes_filling_size": 104,
            "indexes_lobes": np.ones(13),
            "rect_mask_shape": (80, 160),
            "rect_mask_filling_size": 1189,
            "resolution_mask_lobes_size": 925,
            "resolution_mask_lobes_rng_size": 89,
            "resolution_mask_lobes_az_size": 13,
            "pslr_filling_size_peak": 2928,
            "pslr_filling_size_peak_lobes": 2374,
            "pslr_filling_size_res": 2952,
            "pslr_filling_size_res_lobes": 2328,
            "islr_filling_size_peak": 2223,
            "islr_filling_size_peak_lobes": 1314,
            "islr_filling_size_res": 2192,
            "islr_filling_size_res_lobes": 1580,
            "sslr_filling_size": 9216,
            "sslr_filling_size_lobes": 7260,
            "pslr_cut_filling": 51,
            "islr_main_cut_filling": 24,
            "islr_side_cut_filling": 208,
            "islr_side_cut_filling_rng_lobes": 46,
            "islr_side_cut_filling_az_lobes": 19,
            "sslr_cut_filling": 50,
            "sslr_cut_rng_max_lobes": 0.0004309653387735526,
            "sslr_cut_az_max_lobes": 0.00038432402929485405,
        }
        self.data, self.peak_pos, _ = test_support.generate_data_for_test(
            lines=self.reference_values["lines"],
            samples=self.reference_values["samples"],
            samples_start=test_support.default_input_data_generation["samples_start"],
            lines_step=test_support.default_input_data_generation["lines_step"],
            samples_step=test_support.default_input_data_generation["samples_step"],
            fc_hz=test_support.default_input_data_generation["fc_hz"],
            perc=0.9,
        )
        self.x_axis = np.arange(-20, 20, 0.25)
        self.y_axis = np.arange(-30, 30, 0.75)
        self.mask_res = (2, 2.2)
        self.mask_multiplier = (5, 10)
        self.side_lobes = (0.36, 1.7)

    def test_generate_peak_mask(self):
        """Testing generate_peak_mask function on testing data"""
        data = np.abs(self.data) ** 2
        # generating main lobe mask of 1s
        main_lobe_mask = masking.generate_peak_mask(data)
        main_lobe = main_lobe_mask[main_lobe_mask != 0]

        self.assertEqual(main_lobe.size, self.reference_values["main_lobe_mask_filling_size_peak"])
        self.assertFalse(any(main_lobe != 1))

    def test_generate_peak_mask_lobes_0(self):
        """Testing generate_peak_mask_lobes function on testing data, case 0"""
        data = np.abs(self.data) ** 2
        # generating main lobe mask of 1s
        rng_idx, az_idx, main_lobe_mask = masking.generate_peak_mask_lobes(data, side_lobes_directions=self.side_lobes)
        main_lobe = main_lobe_mask[main_lobe_mask != 0]

        np.testing.assert_array_equal(rng_idx[rng_idx != 0], self.reference_values["indexes_lobes"][:-1])
        np.testing.assert_array_equal(az_idx[az_idx != 0], self.reference_values["indexes_lobes"])
        self.assertEqual(main_lobe.size, self.reference_values["main_lobe_mask_lobes_filling_size"])
        self.assertFalse(any(main_lobe != 1))

    def test_generate_peak_mask_lobes_1(self):
        """Testing generate_peak_mask_lobes function on testing data, case 1"""
        data = np.abs(self.data) ** 2
        # generating main lobe mask of 1s
        rng_idx, az_idx, main_lobe_mask = masking.generate_peak_mask_lobes(
            data, side_lobes_directions=(self.side_lobes[1], self.side_lobes[0])
        )
        main_lobe = main_lobe_mask[main_lobe_mask != 0]

        np.testing.assert_array_equal(rng_idx[rng_idx != 0], self.reference_values["indexes_lobes"])
        np.testing.assert_array_equal(az_idx[az_idx != 0], self.reference_values["indexes_lobes"][:-1])
        self.assertEqual(main_lobe.size, self.reference_values["main_lobe_mask_lobes_filling_size"])
        self.assertFalse(any(main_lobe != 1))

    def test_generate_rectangular_mask(self):
        """Testing generate_rectangular_mask function on testing data"""

        # generating rectangular mask from axes
        rect_mask = masking.generate_rectangular_mask(
            x_axis=self.x_axis,
            y_axis=self.y_axis,
            size_x=self.mask_res[0] * self.mask_multiplier[0],
            size_y=self.mask_res[1] * self.mask_multiplier[1],
        )
        filled_mask = rect_mask[rect_mask != 0]

        self.assertEqual(rect_mask.shape, self.reference_values["rect_mask_shape"])
        self.assertEqual(filled_mask.size, self.reference_values["rect_mask_filling_size"])
        self.assertFalse(any(filled_mask != 1))

    def test_generate_resolution_mask(self):
        """Testing generate_resolution_mask function on testing data"""
        res_mask = masking.generate_resolution_mask(
            x_axis=self.x_axis,
            y_axis=self.y_axis,
            res_x=self.mask_res[0],
            res_y=self.mask_res[1],
            multiplier_x=self.mask_multiplier[0],
            multiplier_y=self.mask_multiplier[1],
        )
        filled_mask = res_mask[res_mask != 0]

        self.assertEqual(res_mask.shape, self.reference_values["rect_mask_shape"])
        self.assertEqual(filled_mask.size, self.reference_values["rect_mask_filling_size"])
        self.assertFalse(any(filled_mask != 1))

    def test_generate_resolution_mask_lobes(self):
        """Testing generate_resolution_mask_lobes function on testing data"""
        rng_idx, az_idx, main_lobe_mask = masking.generate_resolution_mask_lobes(
            x_axis=self.x_axis,
            y_axis=self.y_axis,
            res_x=self.mask_res[0],
            res_y=self.mask_res[1],
            multiplier_x=self.mask_multiplier[0],
            multiplier_y=self.mask_multiplier[1],
            side_lobes_directions=self.side_lobes,
        )
        filled_mask = main_lobe_mask[main_lobe_mask != 0]
        filled_rng_idx = rng_idx[rng_idx != 0]
        filled_az_idx = az_idx[az_idx != 0]

        self.assertEqual(rng_idx.size, main_lobe_mask.shape[1])
        self.assertEqual(az_idx.size, main_lobe_mask.shape[0])
        self.assertEqual(filled_rng_idx.size, self.reference_values["resolution_mask_lobes_rng_size"])
        self.assertEqual(filled_az_idx.size, self.reference_values["resolution_mask_lobes_az_size"])
        self.assertEqual(main_lobe_mask.shape, self.reference_values["rect_mask_shape"])
        self.assertEqual(filled_mask.size, self.reference_values["resolution_mask_lobes_size"])
        self.assertFalse(any(filled_mask != 1))

    def test_pslr_masking_peak(self):
        """Testing pslr_masking functions on testing data, PEAK method"""
        pslr_mask = masking.pslr_masking(
            data=self.data,
            mask_flag=MaskingMethod.PEAK,
            peak_pos=self.peak_pos,
            resolution=(self.reference_values["rng_resolution"], self.reference_values["az_resolution"]),
            side_lobes_directions=(np.inf, 0),
        )
        filled_mask = pslr_mask[pslr_mask != 0]

        # testing mask
        self.assertEqual(pslr_mask.shape, (self.reference_values["samples"], self.reference_values["lines"]))
        self.assertEqual(filled_mask.size, self.reference_values["pslr_filling_size_peak"])
        self.assertFalse(any(filled_mask != 1))

    def test_pslr_masking_peak_lobes(self):
        """Testing pslr_masking functions on testing data, PEAK method with lobes"""
        pslr_mask = masking.pslr_masking(
            data=self.data,
            mask_flag=MaskingMethod.PEAK,
            peak_pos=self.peak_pos,
            resolution=(self.reference_values["rng_resolution"], self.reference_values["az_resolution"]),
            side_lobes_directions=self.side_lobes,
        )
        filled_mask = pslr_mask[pslr_mask != 0]

        # testing mask
        self.assertEqual(pslr_mask.shape, (self.reference_values["samples"], self.reference_values["lines"]))
        self.assertEqual(filled_mask.size, self.reference_values["pslr_filling_size_peak_lobes"])
        self.assertFalse(any(filled_mask != 1))

    def test_pslr_masking_resolution(self):
        """Testing pslr_masking functions on testing data, RESOLUTION method"""
        pslr_mask = masking.pslr_masking(
            data=self.data,
            mask_flag=MaskingMethod.RESOLUTION,
            peak_pos=self.peak_pos,
            resolution=(self.reference_values["rng_resolution"], self.reference_values["az_resolution"]),
            side_lobes_directions=(np.inf, 0),
        )
        filled_mask = pslr_mask[pslr_mask != 0]

        # testing mask
        self.assertEqual(pslr_mask.shape, (self.reference_values["samples"], self.reference_values["lines"]))
        self.assertEqual(filled_mask.size, self.reference_values["pslr_filling_size_res"])
        self.assertFalse(any(filled_mask != 1))

    def test_pslr_masking_resolution_lobes(self):
        """Testing pslr_masking functions on testing data, RESOLUTION method with side lobes"""
        pslr_mask = masking.pslr_masking(
            data=self.data,
            mask_flag=MaskingMethod.RESOLUTION,
            peak_pos=self.peak_pos,
            resolution=(self.reference_values["rng_resolution"], self.reference_values["az_resolution"]),
            side_lobes_directions=self.side_lobes,
        )
        filled_mask = pslr_mask[pslr_mask != 0]

        # testing mask
        self.assertEqual(pslr_mask.shape, (self.reference_values["samples"], self.reference_values["lines"]))
        self.assertEqual(filled_mask.size, self.reference_values["pslr_filling_size_res_lobes"])
        self.assertFalse(any(filled_mask != 1))

    def test_islr_masking_peak(self):
        """Testing islr_masking + islr_profile_cutting functions on testing data, PEAK method"""
        main_lobe_mask, islr_mask = masking.islr_masking(
            data=self.data,
            mask_flag=MaskingMethod.PEAK,
            peak_pos=self.peak_pos,
            resolution=(self.reference_values["rng_resolution"], self.reference_values["az_resolution"]),
            side_lobes_directions=(np.inf, 0),
        )
        filled_main_mask = main_lobe_mask[main_lobe_mask != 0]
        filled_islr_mask = islr_mask[islr_mask != 0]

        # testing masks
        self.assertEqual(main_lobe_mask.shape, (self.reference_values["samples"], self.reference_values["lines"]))
        self.assertEqual(main_lobe_mask.shape, islr_mask.shape)
        self.assertEqual(filled_main_mask.size, self.reference_values["main_lobe_mask_filling_size_peak"])
        self.assertFalse(any(filled_main_mask != 1))
        self.assertEqual(filled_islr_mask.size, self.reference_values["islr_filling_size_peak"])
        self.assertFalse(any(filled_islr_mask != 1))

    def test_islr_masking_peak_lobes(self):
        """Testing islr_masking + islr_profile_cutting functions on testing data, PEAK method, with lobes"""
        main_lobe_mask, islr_mask = masking.islr_masking(
            data=self.data,
            mask_flag=MaskingMethod.PEAK,
            peak_pos=self.peak_pos,
            resolution=(self.reference_values["rng_resolution"], self.reference_values["az_resolution"]),
            side_lobes_directions=self.side_lobes,
        )
        filled_main_mask = main_lobe_mask[main_lobe_mask != 0]
        filled_islr_mask = islr_mask[islr_mask != 0]

        # testing masks
        self.assertEqual(main_lobe_mask.shape, (self.reference_values["samples"], self.reference_values["lines"]))
        self.assertEqual(main_lobe_mask.shape, islr_mask.shape)
        self.assertEqual(filled_main_mask.size, self.reference_values["main_lobe_mask_filling_size_peak_lobes"])
        self.assertFalse(any(filled_main_mask != 1))
        self.assertEqual(filled_islr_mask.size, self.reference_values["islr_filling_size_peak_lobes"])
        self.assertFalse(any(filled_islr_mask != 1))

    def test_islr_masking_resolution(self):
        """Testing islr_masking + islr_profile_cutting functions on testing data, RESOLUTION method"""
        main_lobe_mask, islr_mask = masking.islr_masking(
            data=self.data,
            mask_flag=MaskingMethod.RESOLUTION,
            peak_pos=self.peak_pos,
            resolution=(self.reference_values["rng_resolution"], self.reference_values["az_resolution"]),
            side_lobes_directions=(np.inf, 0),
        )
        filled_main_mask = main_lobe_mask[main_lobe_mask != 0]
        filled_islr_mask = islr_mask[islr_mask != 0]

        # testing masks
        self.assertEqual(main_lobe_mask.shape, (self.reference_values["samples"], self.reference_values["lines"]))
        self.assertEqual(main_lobe_mask.shape, islr_mask.shape)
        self.assertEqual(filled_main_mask.size, self.reference_values["main_lobe_mask_filling_size_res"])
        self.assertFalse(any(filled_main_mask != 1))
        self.assertEqual(filled_islr_mask.size, self.reference_values["islr_filling_size_res"])
        self.assertFalse(any(filled_islr_mask != 1))

    def test_islr_masking_resolution_lobes(self):
        """Testing islr_masking + islr_profile_cutting functions on testing data, RESOLUTION method with lobes"""
        main_lobe_mask, islr_mask = masking.islr_masking(
            data=self.data,
            mask_flag=MaskingMethod.RESOLUTION,
            peak_pos=self.peak_pos,
            resolution=(self.reference_values["rng_resolution"], self.reference_values["az_resolution"]),
            side_lobes_directions=self.side_lobes,
        )
        filled_main_mask = main_lobe_mask[main_lobe_mask != 0]
        filled_islr_mask = islr_mask[islr_mask != 0]

        # testing masks
        self.assertEqual(main_lobe_mask.shape, (self.reference_values["samples"], self.reference_values["lines"]))
        self.assertEqual(main_lobe_mask.shape, islr_mask.shape)
        self.assertEqual(filled_main_mask.size, self.reference_values["main_lobe_mask_filling_size_res_lobes"])
        self.assertFalse(any(filled_main_mask != 1))
        self.assertEqual(filled_islr_mask.size, self.reference_values["islr_filling_size_res_lobes"])
        self.assertFalse(any(filled_islr_mask != 1))

    def test_sslr_masking(self):
        """Testing sslr_masking functions on testing data"""
        sslr_mask = masking.sslr_masking(
            data=self.data,
            peak_pos=self.peak_pos,
            resolution=(self.reference_values["rng_resolution"], self.reference_values["az_resolution"]),
            side_lobes_directions=(np.inf, 0),
        )
        filled_sslr_mask = sslr_mask[sslr_mask != 0]

        # testing mask
        self.assertEqual(sslr_mask.shape, (self.reference_values["samples"], self.reference_values["lines"]))
        self.assertEqual(filled_sslr_mask.size, self.reference_values["sslr_filling_size"])
        self.assertFalse(any(filled_sslr_mask != 1))

    def test_sslr_masking_lobes(self):
        """Testing sslr_masking functions on testing data"""
        sslr_mask = masking.sslr_masking(
            data=self.data,
            peak_pos=self.peak_pos,
            resolution=(self.reference_values["rng_resolution"], self.reference_values["az_resolution"]),
            side_lobes_directions=self.side_lobes,
        )
        filled_sslr_mask = sslr_mask[sslr_mask != 0]

        # testing mask
        self.assertEqual(sslr_mask.shape, (self.reference_values["samples"], self.reference_values["lines"]))
        self.assertEqual(filled_sslr_mask.size, self.reference_values["sslr_filling_size_lobes"])
        self.assertFalse(any(filled_sslr_mask != 1))

    def test_pslr_profile_cutting(self):
        """Testing pslr_profile_cutting functions on testing data"""
        pslr_mask = np.zeros_like(self.data)
        idxs = np.array([-1, 1]) * pslr_mask.shape[0] // 10 + pslr_mask.shape[0] // 2
        pslr_mask[idxs[0] : idxs[1], idxs[0] : idxs[1]] = 1
        masked_data = self.data * pslr_mask
        rng_cut, az_cut = masking.pslr_profile_cutting(
            masked_data=masked_data, peak_pos=[int(p) for p in self.peak_pos], side_lobes_directions=(np.inf, 0)
        )
        rng_cut_data = rng_cut[rng_cut != 0]
        az_cut_data = az_cut[az_cut != 0]

        # testing profile cutting
        self.assertEqual(rng_cut_data.size, self.reference_values["pslr_cut_filling"])
        self.assertEqual(az_cut_data.size, self.reference_values["pslr_cut_filling"])
        self.assertEqual(rng_cut.size, self.reference_values["samples"])
        self.assertEqual(az_cut.size, self.reference_values["lines"])

    def test_pslr_profile_cutting_lobes(self):
        """Testing pslr_profile_cutting functions on testing data, with side lobes"""
        pslr_mask = np.zeros_like(self.data)
        idxs = np.array([-1, 1]) * pslr_mask.shape[0] // 10 + pslr_mask.shape[0] // 2
        pslr_mask[idxs[0] : idxs[1], idxs[0] : idxs[1]] = 1
        masked_data = self.data * pslr_mask
        rng_cut, az_cut = masking.pslr_profile_cutting(
            masked_data=masked_data, peak_pos=[int(p) for p in self.peak_pos], side_lobes_directions=self.side_lobes
        )
        rng_cut_data = rng_cut[rng_cut != 0]
        az_cut_data = az_cut[az_cut != 0]

        # testing profile cutting
        self.assertEqual(rng_cut_data.size, self.reference_values["pslr_cut_filling"])
        self.assertEqual(az_cut_data.size, self.reference_values["pslr_cut_filling"])
        self.assertEqual(rng_cut.size, self.reference_values["samples"])
        self.assertEqual(az_cut.size, self.reference_values["lines"])

    def test_islr_profile_cutting(self):
        """Testing islr_profile_cutting functions on testing data"""
        mask = np.zeros_like(self.data)
        pad = mask.shape[0] // 20
        horizontal = mask.copy()
        vertical = mask.copy()
        horizontal[mask.shape[0] // 2 - pad : mask.shape[0] // 2 + pad, pad:-pad] = 1
        vertical[pad:-pad, mask.shape[1] // 2 - pad : mask.shape[1] // 2 + pad] = 1
        islr_mask = np.abs(vertical).astype(bool) ^ np.abs(horizontal).astype(bool)
        main_lobe_mask = np.abs(vertical).astype(bool) & np.abs(horizontal).astype(bool)

        main_lobe_cuts, side_lobes_cuts = masking.islr_profile_cutting(
            data=self.data,
            main_lobe_mask=main_lobe_mask.astype("int64"),
            islr_mask=islr_mask.astype("int64"),
            peak_pos=[int(p) for p in self.peak_pos],
            side_lobes_directions=(np.inf, 0),
        )
        main_cut_rng, main_cut_az = main_lobe_cuts
        side_cut_rng, side_cut_az = side_lobes_cuts
        main_rng_filling = main_cut_rng[main_cut_rng != 0]
        main_az_filling = main_cut_az[main_cut_az != 0]
        side_rng_filling = side_cut_rng[side_cut_rng != 0]
        side_az_filling = side_cut_az[side_cut_az != 0]

        # testing profile cutting
        self.assertEqual(main_rng_filling.size, self.reference_values["islr_main_cut_filling"])
        self.assertEqual(main_az_filling.size, self.reference_values["islr_main_cut_filling"])
        self.assertEqual(side_rng_filling.size, self.reference_values["islr_side_cut_filling"])
        self.assertEqual(side_az_filling.size, self.reference_values["islr_side_cut_filling"])
        self.assertEqual(main_cut_rng.size, self.reference_values["samples"])
        self.assertEqual(main_cut_az.size, self.reference_values["lines"])
        self.assertEqual(side_cut_rng.size, self.reference_values["samples"])
        self.assertEqual(side_cut_az.size, self.reference_values["lines"])

    def test_islr_profile_cutting_lobes(self):
        """Testing islr_profile_cutting functions on testing data, with side lobes"""
        mask = np.zeros_like(self.data)
        pad = mask.shape[0] // 20
        horizontal = mask.copy()
        vertical = mask.copy()
        horizontal[mask.shape[0] // 2 - pad : mask.shape[0] // 2 + pad, pad:-pad] = 1
        vertical[pad:-pad, mask.shape[1] // 2 - pad : mask.shape[1] // 2 + pad] = 1
        islr_mask = np.abs(vertical).astype(bool) ^ np.abs(horizontal).astype(bool)
        main_lobe_mask = np.abs(vertical).astype(bool) & np.abs(horizontal).astype(bool)

        main_lobe_cuts, side_lobes_cuts = masking.islr_profile_cutting(
            data=self.data,
            main_lobe_mask=main_lobe_mask.astype("int64"),
            islr_mask=islr_mask.astype("int64"),
            peak_pos=[int(p) for p in self.peak_pos],
            side_lobes_directions=self.side_lobes,
        )
        main_cut_rng, main_cut_az = main_lobe_cuts
        side_cut_rng, side_cut_az = side_lobes_cuts
        main_rng_filling = main_cut_rng[main_cut_rng != 0]
        main_az_filling = main_cut_az[main_cut_az != 0]
        side_rng_filling = side_cut_rng[side_cut_rng != 0]
        side_az_filling = side_cut_az[side_cut_az != 0]

        # testing profile cutting
        self.assertEqual(main_rng_filling.size, self.reference_values["islr_main_cut_filling"])
        self.assertEqual(main_az_filling.size, self.reference_values["islr_main_cut_filling"])
        self.assertEqual(side_rng_filling.size, self.reference_values["islr_side_cut_filling_rng_lobes"])
        self.assertEqual(side_az_filling.size, self.reference_values["islr_side_cut_filling_az_lobes"])
        self.assertEqual(main_cut_rng.size, self.reference_values["samples"])
        self.assertEqual(main_cut_az.size, self.reference_values["lines"])
        self.assertEqual(side_cut_rng.size, self.reference_values["samples"])
        self.assertEqual(side_cut_az.size, self.reference_values["lines"])

    def test_sslr_profile_cutting(self):
        """Testing sslr_profile_cutting functions on testing data"""
        sslr_mask = np.zeros_like(self.data)
        idxs = np.array([1, 2]) * sslr_mask.shape[0] // 3
        pad = sslr_mask.shape[0] // 10
        sslr_mask[idxs[0] : idxs[1], idxs[0] : idxs[1]] = 1
        sslr_mask[idxs[0] + pad : idxs[1] - pad, idxs[0] + pad : idxs[1] - pad] = 0
        masked_data = self.data * sslr_mask
        rng_cut, az_cut = masking.sslr_profile_cutting(
            masked_data=masked_data, peak_pos=[int(p) for p in self.peak_pos], side_lobes_directions=(np.inf, 0)
        )
        rng_cut_data = rng_cut[rng_cut != 0]
        az_cut_data = az_cut[az_cut != 0]

        # testing profile cutting
        self.assertEqual(rng_cut_data.size, self.reference_values["sslr_cut_filling"])
        self.assertEqual(az_cut_data.size, self.reference_values["sslr_cut_filling"])
        self.assertEqual(rng_cut.size, self.reference_values["samples"])
        self.assertEqual(az_cut.size, self.reference_values["lines"])

    def test_sslr_profile_cutting_lobes(self):
        """Testing sslr_profile_cutting functions on testing data, with side lobes"""
        sslr_mask = np.zeros_like(self.data)
        idxs = np.array([1, 2]) * sslr_mask.shape[0] // 3
        pad = sslr_mask.shape[0] // 10
        sslr_mask[idxs[0] : idxs[1], idxs[0] : idxs[1]] = 1
        sslr_mask[idxs[0] + pad : idxs[1] - pad, idxs[0] + pad : idxs[1] - pad] = 0
        masked_data = self.data * sslr_mask
        rng_cut_max, az_cut_max = masking.sslr_profile_cutting(
            masked_data=masked_data, peak_pos=[int(p) for p in self.peak_pos], side_lobes_directions=self.side_lobes
        )

        # testing profile cutting
        self.assertEqual(rng_cut_max, self.reference_values["sslr_cut_rng_max_lobes"])
        self.assertEqual(az_cut_max, self.reference_values["sslr_cut_az_max_lobes"])

    def test_masking_outliers(self):
        """Testing masking_outliers function"""
        expected_mask = np.array(
            [
                3.897805765002572257e-05,
                2.955835450619122501e-05,
                1.817007299607841723e-05,
                6.005430239230724393e-06,
                5.769923171025669229e-06,
                1.611308360029377787e-05,
                2.418410823233678670e-05,
                2.940449963072034646e-05,
                3.149021896935087593e-05,
                3.045775277363494066e-05,
                3.119642264309920629e-05,
                2.365728246617976346e-05,
                1.454257371496427464e-05,
                4.806497583302908167e-06,
                4.618007481996967955e-06,
                1.289624461515525779e-05,
                1.935595838141861507e-05,
                2.353414339742507025e-05,
                2.520346675332503890e-05,
                2.437712358108526638e-05,
                2.020126191269522509e-05,
                1.531928723717381136e-05,
                9.417052200555936782e-06,
                3.112450349640235408e-06,
                2.990393473183791463e-06,
                8.350970819359797878e-06,
                1.253396228495961862e-05,
                1.523954846045387284e-05,
                1.632051978576620307e-05,
                1.578542077639707175e-05,
                7.021147827260618863e-06,
                5.324369376789782927e-06,
                3.272989374766821932e-06,
                1.081762817801984421e-06,
                1.039340746515644800e-06,
                2.902462275736223467e-06,
                4.356302217373190018e-06,
                5.296655378459619730e-06,
                5.672357624430604143e-06,
                5.486378685924770426e-06,
                7.082467895620319138e-06,
                5.370870419381336835e-06,
                3.301574434808148292e-06,
                1.091210528001027555e-06,
                1.048417958275509872e-06,
                2.927811291244566504e-06,
                4.394348524948095682e-06,
                5.342914377397638002e-06,
                5.721897865691478307e-06,
                5.534294656980370110e-06,
                2.073521597206333015e-05,
                1.572420231833389053e-05,
                9.665961069293898825e-06,
                3.194717759763148554e-06,
                3.069434710360709526e-06,
                8.571701325599091225e-06,
                1.286525644227239188e-05,
                1.564235590874910078e-05,
                1.675189916336493239e-05,
                1.620265656784502929e-05,
                3.258293031612721552e-05,
                2.470871724245546564e-05,
                1.518891032451888321e-05,
                5.020119698116582755e-06,
                4.823252258974814941e-06,
                1.346941104249605996e-05,
                2.021622319837140718e-05,
                2.458010532620055579e-05,
                2.632362083125170486e-05,
                2.546055129580123626e-05,
                4.142511059666499531e-05,
                3.141403595501107785e-05,
                1.931079506758151421e-05,
                6.382452765460593598e-06,
                6.132160500148488893e-06,
                1.712466732407940529e-05,
                2.570239305409839577e-05,
                3.125052202906220379e-05,
                3.346718338895209873e-05,
                3.236989868767550018e-05,
                4.633565525081046160e-05,
                3.513786491049568735e-05,
                2.159990232935054538e-05,
                7.139030571924956381e-06,
                6.859068588712296351e-06,
                1.915463036753468711e-05,
                2.874916219949469917e-05,
                3.495496799622447502e-05,
                3.743439303819212887e-05,
                3.620703588939951919e-05,
                4.675881191976788366e-05,
                3.545875865397071829e-05,
                2.179716171135385647e-05,
                7.204227198152620273e-06,
                6.921708484499659666e-06,
                1.932955849874514969e-05,
                2.901171162597425022e-05,
                3.527419144824496750e-05,
                3.777625964128083201e-05,
                3.653769375140335614e-05,
            ]
        ).reshape((10, 10))
        masked_array = masking.masking_outliers(
            data=np.abs(self.data[10:20, 150:160]),
        )
        np.testing.assert_allclose(masked_array, expected_mask, atol=1e-10, rtol=0)


if __name__ == "__main__":
    unittest.main()
