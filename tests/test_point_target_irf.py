# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Unittest for point_target_irf.py core functionalities"""

import unittest
from dataclasses import fields

from arepyextras.quality.point_targets_analysis.point_target_irf import (
    PointTargetIRFAnalysis,
)

from . import _support as test_support

# import _test_support as test_support


class PointTargetIRFTest(unittest.TestCase):
    """Testing point_target_irf.py core functionalities"""

    def setUp(self) -> None:
        # creating test data
        data, peak_pos, target_pos = test_support.generate_data_for_test(
            lines=test_support.default_input_data_generation["lines"],
            samples=test_support.default_input_data_generation["samples"],
            samples_start=test_support.default_input_data_generation["samples_start"],
            lines_step=test_support.default_input_data_generation["lines_step"],
            samples_step=test_support.default_input_data_generation["samples_step"],
            fc_hz=test_support.default_input_data_generation["fc_hz"],
        )
        self.settings = PointTargetIRFAnalysis(target_area=data, target_pos_real=peak_pos, target_pos_ref=target_pos)

        # benchmarking values
        self.irf_res = test_support.ref_data_irf_results
        self.rcs_res = test_support.ref_data_rcs_results

    def test_full_irf(self):
        """Testing IRF Point Target Analysis Feature"""
        irf_res, _ = self.settings.compute_irf()
        delta_dc = irf_res - self.irf_res
        delta = [getattr(delta_dc, field.name) for field in fields(delta_dc)]
        for value in delta:
            self.assertAlmostEqual(value, 0, None, "Wrong IRF evaluation", 1e-6)

    def test_full_rcs(self):
        """Testing RCS Point Target Analysis Feature"""
        self.settings.compute_irf()
        rcs_ref, _ = self.settings.compute_rcs()
        delta = [
            rcs_ref.clutter - self.rcs_res["clutter"],
            rcs_ref.rcs - self.rcs_res["rcs"],
            rcs_ref.scr - self.rcs_res["scr"],
        ]
        for value in delta:
            self.assertAlmostEqual(value, 0, None, "Wrong RCS evaluation", 1e-6)


if __name__ == "__main__":
    unittest.main()
