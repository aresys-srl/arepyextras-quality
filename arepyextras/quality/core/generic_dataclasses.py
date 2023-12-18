# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Collecting generic purpose dataclasses and enum classes used in the Arepyextras Quality module"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Type, Union

from arepytools.timing.precisedatetime import PreciseDateTime


def convert_to_enum_field(fld: Any, *, enum_type: Type) -> Enum:
    """Function to convert a string value to its corresponding Enum attribute.

    Parameters
    ----------
    fld : Any
        field name, string or Enum type already
    enum_type : Type
        Enum type class of conversion

    Returns
    -------
    Enum
        Enum type attribute corresponding to the input name

    Raises
    ------
    ValueError
        if the input value is not in the target Enum class, this error is raised
    """

    if isinstance(fld, enum_type):
        return fld

    fld = fld.upper()
    if fld in [m.name for m in enum_type]:
        return enum_type[fld]

    raise ValueError(f"{fld} is an invalid {enum_type.__name__} value")


class SAROrbitDirection(Enum):
    """Sensor orbit direction"""

    ASCENDING = auto()
    DESCENDING = auto()
    NOT_AVAILABLE = auto()


class SARPolarization(Enum):
    """Polarization enum class"""

    HH = "H/H"
    VV = "V/V"
    HV = "H/V"
    VH = "V/H"


class CoordinatesType(Enum):
    """Enum class for nominal point target coordinates type"""

    LLH = 0
    ECEF = 1
    NORMALIZED = 2


class DecibelConversion(Enum):
    """Enum class for decibel multiplying factor setting"""

    INTENSITY = auto()
    POWER = auto()


class SARProjection(Enum):
    """Enum class for managing swath projection of product folder"""

    SLANT_RANGE = "SLANT RANGE"
    GROUND_RANGE = "GROUND RANGE"


class TargetDataType(Enum):
    """Enum class for IRF data type labelling"""

    DETECTED = auto()
    COMPLEX = auto()


class MaskingMethod(Enum):
    """Enum class for masking method IRF setting"""

    PEAK = auto()
    RESOLUTION = auto()


class GetFrequencyMethod(Enum):
    """Enum class for get local frequency settings"""

    AUTOCORRELATION = auto()
    FFT = auto()
    POWER_BALANCE = auto()


class SARSideLooking(Enum):
    """Enum class for sensor side looking direction"""

    RIGHT_LOOKING = "RIGHT"
    LEFT_LOOKING = "LEFT"


class SARImageType(Enum):
    """Enum class for image type"""

    RAW = auto()
    RGC = auto()
    SLC = auto()
    GRD = auto()
    NESZ_MAP = auto()
    INT = auto()
    OCN = auto()

    @staticmethod
    def from_str(label: str) -> SARImageType:
        """Generating a Enum SAR Image Type from input string.

        Parameters
        ----------
        label : str
            label string

        Returns
        -------
        SARImageType
            SARImageType enum type corresponding to input string

        Raises
        ------
        NotImplementedError
            if input string is not in the implemented enum types, this raise an error
        """

        if "RAW" in label:
            return SARImageType.RAW
        if "AZIMUTH FOCUSED" in label:
            return SARImageType.SLC
        if "RANGE FOCUSED" in label:
            return SARImageType.RGC
        if "NESZ" in label:
            return SARImageType.NESZ_MAP
        if "MULTILOOK" in label:
            return SARImageType.GRD
        if "INTERFEROMETRY" in label:
            return SARImageType.INT

        raise RuntimeError(f"{label} label not supported")


class SARRadiometricQuantity(Enum):
    """Enum class for radiometric analysis input/output quantity types"""

    BETA_NOUGHT = auto()
    SIGMA_NOUGHT = auto()
    GAMMA_NOUGHT = auto()


@dataclass
class SARSamplingFrequencies:
    """SAR signal sampling frequencies"""

    range_freq_hz: float
    range_bandwidth_freq_hz: float
    azimuth_freq_hz: float
    azimuth_bandwidth_freq_hz: float


@dataclass
class SARCoordinates:
    """SAR Coordinates dataclass to gather azimuth and range index and times in swath"""

    azimuth: PreciseDateTime = None
    range: float = None
    azimuth_index_subpx: float = None
    range_index_subpx: float = None


@dataclass
class PointTargetVisibility:
    """Point Target Visibility dataclass containing burst and swath association"""

    id: str
    channel: int
    burst: Union[list[int], int]
    swath: str
    polarization: str


@dataclass
class LocationData:
    """Location Manager class satisfying the LocationData protocol"""

    abs_azimuth_time: PreciseDateTime
    abs_range_time: float
    incidence_angle: float
    look_angle: float
    ground_velocity: float
    azimuth_step_m: float
    range_step_m: float
    ground_range_step_m: float
