# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Unittest for signal_processing.py core functionalities"""

import unittest

import numpy as np

import arepyextras.quality.core.generic_dataclasses as gdt
import arepyextras.quality.core.signal_processing as sp


class SignalProcessingTest(unittest.TestCase):
    """Testing signal_processing.py core functionalities"""

    def setUp(self) -> None:
        pass

    def test_convert_to_db(self):
        """Testing convert_to_db function"""
        self.assertTrue(np.isnan(sp.convert_to_db(0)))
        self.assertEqual(sp.convert_to_db(10), 10)
        self.assertEqual(sp.convert_to_db(10, mode=gdt.DecibelConversion.POWER), 10)
        self.assertEqual(sp.convert_to_db(10, mode=gdt.DecibelConversion.AMPLITUDE), 20)
        np.testing.assert_array_equal(sp.convert_to_db([10, 100, 1000]), [10, 20, 30])
        np.testing.assert_array_equal(
            sp.convert_to_db([10, 100, 1000], mode=gdt.DecibelConversion.AMPLITUDE), [20, 40, 60]
        )

    def test_locate_max_2d(self):
        """Testing locate_max_2d function"""
        mat = np.zeros((10, 10))
        max_id = (4, 8)
        mat[max_id] = 1
        self.assertEqual(sp.locate_max_2d(mat), max_id)

    def test_crop_array_2d(self):
        """Testing crop_array_2d function"""
        mat = np.zeros((10, 10))
        mat[5, 5] = 1
        cropped = sp.crop_array_2d(mat, (5, 5))
        np.testing.assert_array_equal(sp.crop_array_2d(mat, (3, 3)).shape, (2, 2))
        np.testing.assert_array_equal(sp.crop_array_2d(mat, (4, 4)).shape, (4, 4))
        self.assertTrue(cropped.any())
        self.assertRaises(ValueError, sp.crop_array_2d, mat, (13, 13))
        self.assertFalse(sp.crop_array_2d(mat, (0, 0)).any())

    def test_modulate_data(self):
        """Testing modulate_data function"""
        mat = np.ones((10, 10))
        mod = sp.modulate_data(mat, np.array(1000))
        self.assertTrue(np.dtype(np.complex128) is sp.modulate_data(mat, np.array(0)).dtype)
        self.assertEqual(sp.modulate_data(mat, np.array(0)).sum(), 100)
        self.assertEqual(np.abs(sp.modulate_data(mat, np.array(1000)).sum()), 100)
        self.assertEqual(mat.shape, mod.shape)


if __name__ == "__main__":
    unittest.main()
