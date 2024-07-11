# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Collecting Point Target Analysis specific dataclasses"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Union

import numpy as np
from arepytools.timing.precisedatetime import PreciseDateTime

from arepyextras.quality.core.generic_dataclasses import (
    SAROrbitDirection,
    SARPolarization,
)


@dataclass
class IRFDataOutput:
    """Input Response dataclass containing all relevant information computed in
    the Point Target IRF analysis class."""

    # resolutions
    range_resolution: float = np.nan
    azimuth_resolution: float = np.nan
    # pslr
    range_pslr: float = np.nan
    azimuth_pslr: float = np.nan
    pslr_2d: float = np.nan
    # islr
    range_islr: float = np.nan
    azimuth_islr: float = np.nan
    islr_2d: float = np.nan
    # sslr
    range_sslr: float = np.nan
    azimuth_sslr: float = np.nan
    sslr_2d: float = np.nan
    # localization errors
    slant_range_localization_error: float = np.nan
    azimuth_localization_error: float = np.nan
    ground_range_localization_error: float = np.nan

    @staticmethod
    def from_dict(arg: dict) -> "IRFDataOutput":
        """Creating a IRFDataOutput object by conversion from a dictionary.

        Args:
            arg (dict): dictionary with keys equal to the IRFDataOutput ones

        Returns:
            IRFDataOutput: IRFDataOutput object
        """
        irf_obj = IRFDataOutput()
        for fld in fields(irf_obj):
            if fld.name in arg.keys():
                setattr(irf_obj, fld.name, arg[fld.name])

        return irf_obj

    def __str__(self) -> str:
        message = (
            "Range:"
            + "\n"
            + f"\tSlant Resolution [px]: {self.range_resolution}"
            + "\n"
            + "\tGround Resolution [px]: "
            + f"{self.ground_range_localization_error}"
            + "\n"
            + f"\tPSLR [dB]: {self.range_pslr}"
            + "\n"
            + f"\tISLR [dB]: {self.range_islr}"
            + "\n"
            + f"\tSSLR [dB]: {self.range_sslr}"
            + "\n"
            + "Azimuth:"
            + "\n"
            + f"\tResolution [px]: {self.azimuth_resolution}"
            + "\n"
            + f"\tLocalization Err [px]: {self.azimuth_localization_error}"
            + "\n"
            + f"\tPSLR [dB]: {self.azimuth_pslr}"
            + "\n"
            + f"\tISLR [dB]: {self.azimuth_islr}"
            + "\n"
            + f"\tSSLR [dB]: {self.azimuth_sslr}"
        )
        return message

    def __sub__(self, other: "IRFDataOutput") -> "IRFDataOutput":
        """Subtraction operation override.

        Parameters
        ----------
        other : IRFDataOutput
            object to be subtracted to self, actually another IRFDataOutput object

        Returns
        -------
        IRFDataOutput
            IRFDataOutput dataclass containing the subtraction between the two
        """
        if isinstance(other, IRFDataOutput):
            for fld in fields(self):
                setattr(self, fld.name, getattr(self, fld.name) - getattr(other, fld.name))

            return self
        else:
            return NotImplemented


@dataclass
class RCSDataOutput:
    """Dataclass to collect the output of RCS analysis"""

    rcs: float = np.nan
    rcs_error: float = np.nan
    peak_phase_error: float = np.nan
    peak_value_complex: complex = np.nan
    clutter: float = np.nan
    scr: float = np.nan


@dataclass
class GenericInfoOutput:
    """Dataclass to collect generic output for the whole Point Target Analysis"""

    swath: str = None
    burst: int = None
    product_type: str = None
    polarization: Union[SARPolarization, str] = None
    incidence_angle: float = None
    squint_angle: float = None
    range_position: float = None
    azimuth_position: float = None


@dataclass
class PTAdditionalInfo:
    """Additional info for other needs"""

    orbit_direction: SAROrbitDirection = None  # sensor orbit direction
    peak_azimuth_time: PreciseDateTime = None  # azimuth time at signal peak coordinates
    peak_azimuth_from_burst_start: float = (
        None  # azimuth pixel position at signal peak coordinates relative to burst start
    )
    peak_range_time: float = None  # range time at signal peak coordinates
    look_angle: float = None  # look angle at nominal target coordinates
    ground_velocity: float = None  # ground velocity at nominal target coordinates
    doppler_rate_theoretical: float = None  # theoretical doppler rate at nominal target coordinates
    doppler_rate_real: float = None  # real (annotated) doppler rate at nominal target coordinates
    doppler_frequency: float = None  # doppler frequency from mechanical squint angle at signal peak coordinates
    steering_doppler_frequency: float = (
        None  # doppler frequency due to electrical antenna steering at signal peak coordinates
    )


@dataclass
class PointTargetAnalysisOutput:
    """Dataclass to collect output for the whole Point Target Analysis"""

    target: int = None
    channel: int = None
    info: GenericInfoOutput = field(default_factory=GenericInfoOutput)
    irf: IRFDataOutput = field(default_factory=IRFDataOutput)
    rcs: RCSDataOutput = field(default_factory=RCSDataOutput)
    additional_info: PTAdditionalInfo = field(default_factory=PTAdditionalInfo)


@dataclass
class IRFGraphDataOutput:
    """Dataclass needed to store data for generating graphical output for IRF"""

    image: np.ndarray = None
    rng_axis: np.ndarray = None
    rng_profile: np.ndarray = None
    rng_resolution: float = None
    rng_step_distance: float = None
    az_axis: np.ndarray = None
    az_profile: np.ndarray = None
    az_resolution: float = None
    az_step_distance: float = None
    side_lobes_directions: np.ndarray = None


@dataclass
class RCSGraphDataOutput:
    """Dataclass needed to store data for generating graphical output for RCS"""

    image: np.ndarray = None
    data_type: str = None
    roi_background: list = None
    roi_peak: list = None
    rng_step_distance: float = None
    az_step_distance: float = None
    roi_size: np.ndarray = None
    interp_factor: int = None
    rcs_lin: float = None
    rcs_db: float = None


@dataclass
class PointTargetGraphicalData:
    """Dataclass to collect data for graphical output of Point Target Analysis"""

    target: int = None
    channel: int = None
    swath: str = None
    burst: int = None
    polarization: SARPolarization = None
    irf: IRFGraphDataOutput = None
    rcs: RCSGraphDataOutput = None
