# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Collecting all dataclasses used in Radiometric Analysis application"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import Enum, auto
from pathlib import Path
from typing import Union

import numpy as np
import toml
from arepytools.io.metadata import EPolarization
from arepytools.timing.precisedatetime import PreciseDateTime

from arepyextras.quality.core.generic_dataclasses import (
    SARRadiometricQuantity,
    convert_to_enum_field,
)


class RadiometricAnalysisDirection(Enum):
    """Enum class for radiometric analysis direction"""

    RANGE = auto()
    AZIMUTH = auto()
    ALL = auto()


class RadiometricAnalysisAxes(Enum):
    """Enum class for radiometric analysis output axes to represent data versus"""

    NATURAL = auto()  # Time/Distance
    INCIDENCE_ANGLE = auto()
    LOOK_ANGLE = auto()


class RadiometricAnalysisValue(Enum):
    """Enum class for radiometric analysis value to be represented"""

    AMPLITUDE = auto()
    PHASE = auto()


@dataclass
class RadiometricAnalysisParameters:
    """Dataclass to store configuration parameters for Radiometric Analysis functions"""

    smoothening_order: int = 3
    smoothening_window_length: int = 71
    radiometric_correction_exponent: float = 0.5
    outliers_kernel_size: tuple[int, int] = (5, 5)
    outliers_filter_kernel_size: tuple[int, int] = (10, 10)
    outliers_percentile_boundaries: tuple[int, int] = (20, 90)
    az_average_band: int = 1000
    rng_average_band: int = 1000

    @staticmethod
    def from_dict(arg: dict) -> RadiometricAnalysisParameters:
        """Creating a RadiometricAnalysisParameters object by conversion from a dictionary.

        Args:
            arg (dict): dictionary with keys equal to the RadiometricAnalysisParameters ones

        Returns:
            RadiometricAnalysisParameters: RadiometricAnalysisParameters object
        """
        rap_obj = RadiometricAnalysisParameters()
        for fld in fields(rap_obj):
            if fld.name in arg.keys():
                if isinstance(arg[fld.name], list):
                    setattr(rap_obj, fld.name, tuple(arg[fld.name]))
                else:
                    setattr(rap_obj, fld.name, arg[fld.name])

        return rap_obj


@dataclass
class RadiometricAnalysisConfig:
    """Radiometric Analysis configuration setup dataclass"""

    input_type: SARRadiometricQuantity = SARRadiometricQuantity.BETA_NOUGHT
    output_type: SARRadiometricQuantity = SARRadiometricQuantity.BETA_NOUGHT
    value: RadiometricAnalysisValue = RadiometricAnalysisValue.AMPLITUDE
    direction: RadiometricAnalysisDirection = RadiometricAnalysisDirection.RANGE
    axis: RadiometricAnalysisAxes = RadiometricAnalysisAxes.NATURAL
    outlier_removal: bool = False
    smoothening_filter: bool = True
    parameters: RadiometricAnalysisParameters = field(default_factory=RadiometricAnalysisParameters)

    @staticmethod
    def from_toml(toml_file: Path) -> RadiometricAnalysisConfig:
        """Generating an RadiometricAnalysisConfig dataclass from a configuration toml file.

        Parameters
        ----------
        arg : Path
            path to the toml file

        Returns
        -------
        AnalysisConfig
            output dataclass

        Raises
        ------
        ValueError
            if input toml is not valid, this error is raised
        """

        # loading toml file
        with open(toml_file, "r", encoding="UTF-8") as f_in:
            config = toml.load(f_in)

        # accessing each field of the dictionary and storing its values in a new dataclass
        config = config["radiometric_analysis"]
        try:
            out = RadiometricAnalysisConfig.from_dict(config)

            return out

        except Exception as err:
            raise ValueError("Invalid toml file.") from err

    @staticmethod
    def from_dict(arg: dict) -> RadiometricAnalysisConfig:
        """Creating a RadiometricAnalysisConfig object by conversion from a dictionary.

        Parameters
        ----------
        arg : dict
            dictionary with keys equal to the RadiometricAnalysisConfig ones

        Returns
        -------
        RadiometricAnalysisConfig
            RadiometricAnalysisConfig object

        Raises
        ------
        ValueError
            invalid dictionary structure
        """
        ra_obj = RadiometricAnalysisConfig()

        try:
            if "parameters" in arg:
                ra_obj.parameters = RadiometricAnalysisParameters.from_dict(arg["parameters"])

            if "input_type" in arg:
                ra_obj.input_type = convert_to_enum_field(arg["input_type"], enum_type=SARRadiometricQuantity)
            if "output_type" in arg:
                ra_obj.output_type = convert_to_enum_field(arg["output_type"], enum_type=SARRadiometricQuantity)
            if "value" in arg:
                ra_obj.value = convert_to_enum_field(arg["value"], enum_type=RadiometricAnalysisValue)
            if "direction" in arg:
                ra_obj.direction = convert_to_enum_field(arg["direction"], enum_type=RadiometricAnalysisDirection)
            if "axis" in arg:
                ra_obj.axis = convert_to_enum_field(arg["axis"], enum_type=RadiometricAnalysisAxes)
            if "outlier_removal" in arg:
                ra_obj.outlier_removal = arg["outlier_removal"]
            if "smoothening_filter" in arg:
                ra_obj.smoothening_filter = arg["smoothening_filter"]

            return ra_obj

        except Exception as err:
            raise ValueError("Invalid dictionary structure.") from err


@dataclass
class RadiometricAnalysisOutput:
    """Dataclass to collect generic output from Radiometric Analysis"""

    swath: str = None
    burst: int = None
    channel: int = None
    polarization: EPolarization = None
    profile: np.ndarray = None
    smoothed_profile: np.ndarray = None
    axis: np.ndarray = None
    time: Union[float, PreciseDateTime] = None
    direction: RadiometricAnalysisDirection = None
    value_type: RadiometricAnalysisValue = None
    axis_format: RadiometricAnalysisAxes = None
    radiometric_type: SARRadiometricQuantity = None
