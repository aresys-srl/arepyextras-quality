# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
Noise Equivalent Sigma Zero (NESZ) custom dataclasses
-----------------------------------------------------
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

from arepyextras.quality.core.generic_dataclasses import SARPolarization


@dataclass
class NESZOutput:
    """Dataclass to store NESZ output for a single channel"""

    channel: str
    swath: str
    polarization: SARPolarization
    azimuth_blocks_num: int
    elevation_angles_deg: np.ndarray
    nesz_profiles: np.ndarray
    axis_deg: np.ndarray


@dataclass
class NESZConfig:
    """Dataclass to manage, enable and customize different part of the NESZ procedure"""

    rng_multilook_length: int = 7  # sort of moving average window parameter for range
    az_multilook_length: int = 7  # sort of moving average window parameter for azimuth
    az_block_size: int = 2000  # azimuth block size for partitioning the whole scene
    look_angle_step: float = 0.01  # look angle step in degrees for graphs purposes
    look_angle_margin: float = 0.5  # look angle margin in degrees to remove near and far range values
    pixel_margin: int = 150  # pixel margin to remove near and far range values
    burst_center_block_size: int = 100  # dimension of the central block in case of topsar/scansar acquisitions
    incidence_compensation: bool = False  # perform incidence compensation

    @staticmethod
    def from_dict(arg: dict) -> NESZConfig:
        """Creating a NESZConfig object by conversion from a dictionary.

        Parameters
        ----------
        arg : dict
            dictionary with keys equal to the NESZConfig ones

        Returns
        -------
        NESZConfig
            NESZConfig object

        Raises
        ------
        ValueError
            invalid dictionary structure
        """
        nesz_obj = NESZConfig()
        valid_fields = [f.name for f in fields(nesz_obj)]
        try:
            for key, value in arg.items():
                if key not in valid_fields:
                    raise ValueError("Invalid dictionary structure.")
                setattr(nesz_obj, key, value)

            return nesz_obj

        except Exception as err:
            raise ValueError("Invalid dictionary structure.") from err
