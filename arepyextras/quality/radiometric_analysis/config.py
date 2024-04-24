# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Collecting all dataclasses used in Radiometric Analysis application"""

from __future__ import annotations

from dataclasses import dataclass, field, fields

from arepyextras.quality.core.generic_dataclasses import (
    SARRadiometricQuantity,
    convert_to_enum_field,
)


@dataclass
class Radiometric2DHistogramParameters:
    """Radiometric 2D Histogram configuration parameters"""

    x_bins_step: int | None = None  # resampling step of x axis
    y_bins_num: int | None = None  # number of bins along y axis
    y_bins_center_margin: float | None = (
        None  # +- margin defining the extension of the y binning axis around y center value
    )

    @classmethod
    def from_dict(cls, arg: dict) -> Radiometric2DHistogramParameters:
        """Creating a Radiometric2DHistogramParameters object by conversion from a dictionary.

        Parameters
        ----------
        arg : dict
            dictionary with keys equal to the ProfileExtractionParameters ones

        Returns
        -------
        Radiometric2DHistogramParameters
            Radiometric2DHistogramParameters object

        Raises
        ------
        ValueError
            invalid dictionary structure
        """
        h_obj = cls()

        try:
            for fld in fields(cls):
                if fld.name in arg.keys():
                    setattr(h_obj, fld.name, arg[fld.name])

            return h_obj

        except Exception as err:
            raise ValueError("Invalid dictionary structure.") from err


@dataclass
class ProfileExtractionParameters:
    """Dataclass to store configuration parameters for Radiometric Analysis functions"""

    outlier_removal: bool = True
    smoothening_filter: bool = True
    filtering_kernel_size: tuple[int, int] | None = None
    outliers_percentile_boundaries: tuple[int, int] = (20, 90)
    outliers_kernel_size: tuple[int, int] = (5, 5)

    @classmethod
    def from_dict(cls, arg: dict) -> ProfileExtractionParameters:
        """Creating a ProfileExtractionParameters object by conversion from a dictionary.

        Parameters
        ----------
        arg : dict
            dictionary with keys equal to the ProfileExtractionParameters ones

        Returns
        -------
        ProfileExtractionParameters
            ProfileExtractionParameters object

        Raises
        ------
        ValueError
            invalid dictionary structure
        """
        pf_obj = cls()

        try:
            for fld in fields(cls):
                if fld.name in arg.keys():
                    if isinstance(arg[fld.name], list):
                        setattr(pf_obj, fld.name, tuple(arg[fld.name]))
                    else:
                        setattr(pf_obj, fld.name, arg[fld.name])

            return pf_obj

        except Exception as err:
            raise ValueError("Invalid dictionary structure.") from err


@dataclass
class RadiometricProfilesConfig:
    """Radiometric Profiles configuration setup dataclass"""

    input_quantity: SARRadiometricQuantity = SARRadiometricQuantity.BETA_NOUGHT
    azimuth_block_size: int = 2000
    range_pixel_margin: int = 150
    radiometric_correction_exponent: float = 1.0
    histogram_parameters: Radiometric2DHistogramParameters = field(default_factory=Radiometric2DHistogramParameters)
    profile_extraction_parameters: ProfileExtractionParameters = field(default_factory=ProfileExtractionParameters)

    @classmethod
    def from_dict(cls, arg: dict) -> RadiometricProfilesConfig:
        """Creating a RadiometricProfilesConfig object by conversion from a dictionary.

        Parameters
        ----------
        arg : dict
            dictionary with keys equal to the RadiometricProfilesConfig ones

        Returns
        -------
        RadiometricProfilesConfig
            RadiometricProfilesConfig object

        Raises
        ------
        ValueError
            invalid dictionary structure
        """
        ra_obj = cls()

        try:
            if "histogram_parameters" in arg:
                ra_obj.histogram_parameters = Radiometric2DHistogramParameters.from_dict(arg["histogram_parameters"])

            if "profile_extraction_parameters" in arg:
                ra_obj.profile_extraction_parameters = ProfileExtractionParameters.from_dict(
                    arg["profile_extraction_parameters"]
                )

            if "input_quantity" in arg:
                ra_obj.input_quantity = convert_to_enum_field(arg["input_quantity"], enum_type=SARRadiometricQuantity)
            if "azimuth_block_size" in arg:
                ra_obj.azimuth_block_size = arg["azimuth_block_size"]
            if "range_pixel_margin" in arg:
                ra_obj.range_pixel_margin = arg["range_pixel_margin"]
            if "radiometric_correction_exponent" in arg:
                ra_obj.radiometric_correction_exponent = arg["radiometric_correction_exponent"]

            return ra_obj

        except Exception as err:
            raise ValueError("Invalid dictionary structure.") from err
