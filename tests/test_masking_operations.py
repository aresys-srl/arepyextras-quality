# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Unittest for masking_operations.py functions"""

import unittest

import numpy as np

import arepyextras.quality.core.masking_operations as masking
from arepyextras.quality.core.generic_dataclasses import MaskingMethod

from . import _support as test_support

# import _test_support as test_support


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
            "main_lobe_mask_filling_size": 144,
            "rect_mask_shape": (80, 160),
            "rect_mask_filling_size": 1189,
            "pslr_filling_size": 2928,
            "islr_filling_size": 2223,
            "sslr_filling_size": 9216,
            "pslr_cut_filling": 51,
            "islr_main_cut_filling": 24,
            "islr_side_cut_filling": 208,
            "sslr_cut_filling": 50,
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

    def test_generate_peak_mask(self):
        """Testing generate_peak_mask function on testing data"""
        data = np.abs(self.data) ** 2
        # generating main lobe mask of 1s
        main_lobe_mask = masking.generate_peak_mask(data)
        main_lobe = main_lobe_mask[main_lobe_mask != 0]

        self.assertEqual(main_lobe.size, self.reference_values["main_lobe_mask_filling_size"])
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

    def test_pslr_masking(self):
        """Testing pslr_masking functions on testing data"""
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
        self.assertEqual(filled_mask.size, self.reference_values["pslr_filling_size"])
        self.assertFalse(any(filled_mask != 1))

    def test_islr_masking(self):
        """Testing islr_masking + islr_profile_cutting functions on testing data"""
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
        self.assertEqual(filled_main_mask.size, self.reference_values["main_lobe_mask_filling_size"])
        self.assertFalse(any(filled_main_mask != 1))
        self.assertEqual(filled_islr_mask.size, self.reference_values["islr_filling_size"])
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


if __name__ == "__main__":
    unittest.main()
