# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Collecting Point Target Analysis specific dataclasses"""

from __future__ import annotations

from dataclasses import dataclass, field, fields

from arepyextras.quality.core.generic_dataclasses import (
    MaskingMethod,
    convert_to_enum_field,
)


@dataclass
class IRFParameters:
    """IRF analysis detailed setup parameters"""

    peak_finding_roi_size: tuple[int, int] = (33, 33)
    analysis_roi_size: tuple[int, int] = (128, 128)
    oversampling_factor: int = 16
    zero_doppler_abs_squint_threshold_deg: float = 1.0
    masking_method: MaskingMethod = MaskingMethod.PEAK

    @classmethod
    def from_dict(cls, arg: dict) -> IRFParameters:
        """Creating a IRFParameters object by conversion from a dictionary.

        Args:
            arg (dict): dictionary with keys equal to the IRFParameters ones

        Returns:
            IRFParameters: IRFParameters object
        """
        irf_obj = cls()
        for fld in fields(cls):
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

    @classmethod
    def from_dict(cls, arg: dict) -> RCSParameters:
        """Creating a RCSParameters object by conversion from a dictionary.

        Args:
            arg (dict): dictionary with keys equal to the RCSParameters ones

        Returns:
            RCSParameters: RCSParameters object
        """
        rcs_obj = cls()
        for fld in fields(cls):
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
    check_targets_in_scene: bool = True
    ale_limits: tuple[float, float] | None = None
    irf_parameters: IRFParameters = field(default_factory=IRFParameters)
    rcs_parameters: RCSParameters = field(default_factory=RCSParameters)

    @classmethod
    def from_dict(cls, arg: dict) -> PointTargetAnalysisConfig:
        """Creating a PointTargetAnalysisConfig object by conversion from a dictionary.

        Args:
            arg (dict): dictionary with keys equal to the PointTargetAnalysisConfig ones

        Returns:
            PointTargetAnalysisConfig: PointTargetAnalysisConfig object
        """
        pta_obj = cls()
        dict_in = arg.copy()

        try:
            if "irf_parameters" in dict_in:
                pta_obj.irf_parameters = IRFParameters.from_dict(dict_in.pop("irf_parameters"))
            if "rcs_parameters" in dict_in:
                pta_obj.rcs_parameters = RCSParameters.from_dict(dict_in.pop("rcs_parameters"))

            dtc_fields = [f.name for f in fields(cls)]
            for key, value in dict_in.items():
                if key in dtc_fields:
                    if isinstance(value, list):
                        setattr(pta_obj, key, tuple(value))
                    else:
                        setattr(pta_obj, key, value)

            return pta_obj

        except Exception as err:
            raise ValueError("Invalid dictionary structure.") from err
