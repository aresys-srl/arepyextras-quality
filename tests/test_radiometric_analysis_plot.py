# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Unittest for radiometric_analysis/graphical_output.py core functionalities"""

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from arepytools.timing.precisedatetime import PreciseDateTime

from arepyextras.quality.core.generic_dataclasses import (
    SARPolarization,
    SARRadiometricQuantity,
)
from arepyextras.quality.radiometric_analysis.custom_dataclasses import (
    RadiometricAnalysisDirection,
    RadiometricProfilesOutput,
)
from arepyextras.quality.radiometric_analysis.graphical_output import (
    radiometric_2D_hist_plot,
)


@unittest.skipIf(sys.platform.startswith("win"), "skipping Windows on CI")
class Radiometric2DHistPlotTest(unittest.TestCase):
    """Testing Radiometric Analysis graphical output functionalities"""

    def test_radiometric_2D_hist_plot(self) -> None:
        """Testing radiometric_2D_hist_plot"""
        profiles = np.ones((3, 100))
        profiles[1, :] *= 15
        profiles[2, :] *= 125
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
            block_azimuth_times=np.tile(np.linspace(9, 18, 100), 3).reshape((3, 100)),
            hist_x_bins_axis=np.linspace(-3, 80, 100),
            hist_y_bins_axis=np.linspace(5, 35, 100),
            look_angles=np.ones((3, 100)) * 15,
            output_radiometric_quantity=SARRadiometricQuantity.BETA_NOUGHT,
            polarization=SARPolarization.HH,
            profiles=profiles,
        )
        with TemporaryDirectory() as temp_dir:
            tag = "test"
            radiometric_2D_hist_plot(data=data, out_dir=temp_dir, title=tag)
            out_file = Path(temp_dir).joinpath("graphs", tag.lower().replace(" ", "_")).with_suffix(".png")
            self.assertTrue(out_file.exists())
            self.assertTrue(out_file.is_file())


if __name__ == "__main__":
    unittest.main()
