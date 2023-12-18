# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Collecting Point Target Analysis specific dataclasses"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Union

import numpy as np
import toml
from arepytools.timing.precisedatetime import PreciseDateTime

from arepyextras.quality.core.generic_dataclasses import (
    MaskingMethod,
    SAROrbitDirection,
    SARPolarization,
    convert_to_enum_field,
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
class IRFParameters:
    """IRF analysis detailed setup parameters"""

    peak_finding_roi_size: tuple[int, int] = (33, 33)
    analysis_roi_size: tuple[int, int] = (128, 128)
    oversampling_factor: int = 16
    zero_doppler_abs_squint_threshold_deg: float = 1.0
    masking_method: MaskingMethod = MaskingMethod.PEAK

    @staticmethod
    def from_dict(arg: dict) -> IRFParameters:
        """Creating a IRFParameters object by conversion from a dictionary.

        Args:
            arg (dict): dictionary with keys equal to the IRFParameters ones

        Returns:
            IRFParameters: IRFParameters object
        """
        irf_obj = IRFParameters()
        for fld in fields(irf_obj):
            if fld.name in arg.keys():
                if fld.name == "masking_method":
                    setattr(irf_obj, fld.name, convert_to_enum_field(arg[fld.name], enum_type=MaskingMethod))
                elif isinstance(arg[fld.name], list):
                    setattr(irf_obj, fld.name, tuple(arg[fld.name]))
                else:
                    setattr(irf_obj, fld.name, arg[fld.name])

        return irf_obj


@dataclass
class RCSParameters:
    """RCS analysis detailed setup parameters"""

    interpolation_factor: int = 8
    roi_dimension: int = 128
    calibration_factor: float = 1.0
    resampling_factor: float = 1.0

    @staticmethod
    def from_dict(arg: dict) -> RCSParameters:
        """Creating a RCSParameters object by conversion from a dictionary.

        Args:
            arg (dict): dictionary with keys equal to the RCSParameters ones

        Returns:
            RCSParameters: RCSParameters object
        """
        rcs_obj = RCSParameters()
        for fld in fields(rcs_obj):
            if fld.name in arg.keys():
                if isinstance(arg[fld.name], list):
                    setattr(rcs_obj, fld.name, tuple(arg[fld.name]))
                else:
                    setattr(rcs_obj, fld.name, arg[fld.name])

        return rcs_obj


@dataclass
class PointTargetAnalysisConfig:
    """Dataclass to manage, enable and customize different part of the Point Target Analysis procedure"""

    perform_irf: bool = True
    perform_rcs: bool = True
    evaluate_pslr: bool = True
    evaluate_islr: bool = True
    evaluate_sslr: bool = True
    evaluate_localization: bool = True
    generate_static_graphs: bool = True
    generate_interactive_graphs: bool = False
    irf_parameters: IRFParameters = field(default_factory=IRFParameters)
    rcs_parameters: RCSParameters = field(default_factory=RCSParameters)

    @staticmethod
    def from_toml(toml_file: Path) -> PointTargetAnalysisConfig:
        """Generating an PointTargetAnalysisConfig dataclass from a configuration toml file.

        Parameters
        ----------
        arg : Path
            path to the toml file

        Returns
        -------
        PointTargetAnalysisConfig
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
        config = config["point_target_analysis"]
        try:
            out = PointTargetAnalysisConfig.from_dict(config)

            return out

        except Exception as err:
            raise ValueError("Invalid toml file.") from err

    @staticmethod
    def from_dict(arg: dict) -> PointTargetAnalysisConfig:
        """Creating a PointTargetAnalysisConfig object by conversion from a dictionary.

        Args:
            arg (dict): dictionary with keys equal to the PointTargetAnalysisConfig ones

        Returns:
            PointTargetAnalysisConfig: PointTargetAnalysisConfig object
        """
        pta_obj = PointTargetAnalysisConfig()
        dict_in = arg.copy()

        try:
            if "irf_parameters" in dict_in:
                pta_obj.irf_parameters = IRFParameters.from_dict(dict_in.pop("irf_parameters"))
            if "rcs_parameters" in dict_in:
                pta_obj.rcs_parameters = RCSParameters.from_dict(dict_in.pop("rcs_parameters"))

            dtc_fields = [f.name for f in fields(pta_obj)]
            for key, value in dict_in.items():
                if key in dtc_fields:
                    setattr(pta_obj, key, value)

            return pta_obj

        except Exception as err:
            raise ValueError("Invalid dictionary structure.") from err


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
    irf: IRFGraphDataOutput = None
    rcs: RCSGraphDataOutput = None
