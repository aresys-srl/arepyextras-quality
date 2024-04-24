# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Collecting all dataclasses used in Radiometric Analysis application"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Union

import numpy as np
from arepytools.timing.precisedatetime import PreciseDateTime

from arepyextras.quality.core.generic_dataclasses import (
    SARPolarization,
    SARRadiometricQuantity,
)


class RadiometricAnalysisDirection(Enum):
    """Enum class for radiometric analysis direction"""

    RANGE = auto()
    AZIMUTH = auto()
    ALL = auto()


@dataclass
class RadiometricProfilesOutput:
    """Dataclass to collect Radiometric Profiles output"""

    swath: str | None = None
    channel: Union[str, int] | None = None
    polarization: SARPolarization | None = None
    direction: RadiometricAnalysisDirection | None = None
    output_radiometric_quantity: SARRadiometricQuantity | None = None
    azimuth_block_centers: np.ndarray | None = None
    range_block_centers: np.ndarray | None = None
    blocks_num: int | None = None
    azimuth_start_time: PreciseDateTime | None = None
    profiles: np.ndarray | None = None
    look_angles: np.ndarray | None = None
    block_azimuth_times: np.ndarray | None = None
    hist_2d: np.ndarray | None = None
    hist_x_bins_axis: np.ndarray | None = None
    hist_y_bins_axis: np.ndarray | None = None
