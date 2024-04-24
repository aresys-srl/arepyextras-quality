# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Unittest for interferometric analysis configuration"""

import unittest
from dataclasses import asdict

from arepyextras.quality.interferometric_analysis.config import InterferometricConfig


class InterferometricConfigTesting(unittest.TestCase):
    """Testing InterferometricConfig core functionalities"""

    def setUp(self) -> None:
        # creating test data
        self._config = {
            "enable_coherence_computation": True,
            "coherence_kernel": 155,
            "azimuth_blocks_number": 16,
            "range_blocks_number": 47,
            "coherence_bins_number": 800,
        }

    def test_config_from_dict(self):
        """Testing InterferometricConfig generation from dictionary"""
        dtc = InterferometricConfig.from_dict(self._config)

        int_config_dict = asdict(dtc)
        self.assertDictEqual(self._config, int_config_dict)


if __name__ == "__main__":
    unittest.main()
