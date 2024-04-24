# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Unittest for point_target_analysis/custom_dataclasses.py core functionalities"""

import unittest
from dataclasses import fields

from arepyextras.quality.core.generic_dataclasses import MaskingMethod
from arepyextras.quality.point_targets_analysis.config import (
    IRFParameters,
    PointTargetAnalysisConfig,
    RCSParameters,
)


class PointTargetDataclasses(unittest.TestCase):
    """Testing point_target_analysis/custom_dataclasses.py core functionalities"""

    def setUp(self) -> None:
        # creating test data
        self.pta_bool_flags = {
            "perform_irf": True,
            "perform_rcs": False,
            "evaluate_pslr": True,
            "evaluate_islr": False,
            "evaluate_sslr": True,
            "evaluate_localization": False,
            "generate_static_graphs": False,
            "generate_interactive_graphs": True,
            "check_targets_in_scene": False,
            "ale_limits": (1.5, 4),
        }

        self.irf_params = {
            "peak_finding_roi_size": [34, 34],
            "analysis_roi_size": [128, 128],
            "oversampling_factor": 16,
            "masking_method": MaskingMethod.PEAK,
        }
        self.rcs_params = {
            "interpolation_factor": 8,
            "roi_dimension": 128,
            "calibration_factor": 1,
            "resampling_factor": 1,
        }

    def test_irf_parameters_from_dict(self):
        """Testing IRFParameters from dict method dataclass generation from dictionary"""
        dtc = IRFParameters.from_dict(self.irf_params)

        for key, item in self.irf_params.items():
            dataclass_key = [field.name for field in fields(dtc) if key in field.name][0]
            value = getattr(dtc, dataclass_key)
            if ("peak_finding_roi_size" in key) | ("analysis_roi_size" in key):
                self.assertEqual(tuple(item), value)
            elif "masking_method" in key:
                self.assertEqual(item, value)
            else:
                self.assertEqual(item, value)

    def test_rcs_parameters_from_dict(self):
        """Testing RCSParameters from dict method dataclass generation from dictionary"""
        dtc = RCSParameters.from_dict(self.rcs_params)

        for key, item in self.rcs_params.items():
            dataclass_key = [field.name for field in fields(dtc) if key in field.name][0]
            value = getattr(dtc, dataclass_key)

            self.assertEqual(item, value)

    def test_point_target_analysis_config_from_dict(self):
        """Testing PointTargetAnalysisConfig dataclass generation from dictionary"""
        total_dict = self.pta_bool_flags.copy()
        total_dict["irf_parameters"] = self.irf_params
        total_dict["rcs_parameters"] = self.rcs_params

        # loading it back via dataclass method
        config = PointTargetAnalysisConfig.from_dict(total_dict)

        self.assertTrue(isinstance(config.irf_parameters, IRFParameters))
        self.assertTrue(isinstance(config.rcs_parameters, RCSParameters))

        # check consistency with the default one
        for key, item in self.pta_bool_flags.items():
            dataclass_key = [field.name for field in fields(config) if key in field.name][0]
            value = getattr(config, dataclass_key)
            self.assertEqual(item, value)

        # irf params
        for key, item in self.irf_params.items():
            dataclass_key = [field.name for field in fields(IRFParameters) if key in field.name][0]
            value = getattr(config.irf_parameters, dataclass_key)
            if isinstance(item, list):
                self.assertEqual(tuple(item), value)
            else:
                self.assertEqual(item, value)

        # rcs params
        for key, item in self.rcs_params.items():
            dataclass_key = [field.name for field in fields(RCSParameters) if key in field.name][0]
            value = getattr(config.rcs_parameters, dataclass_key)
            if isinstance(item, list):
                self.assertEqual(tuple(item), value)
            else:
                self.assertEqual(item, value)


if __name__ == "__main__":
    unittest.main()
