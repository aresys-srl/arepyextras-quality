# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""CLI default configuration file generation"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import appdirs
import toml
from jsonschema import validate

from arepyextras.quality import config_schema
from arepyextras.quality.nesz_analysis.custom_dataclasses import NESZConfig
from arepyextras.quality.point_targets_analysis.custom_dataclasses import (
    PointTargetAnalysisConfig,
)
from arepyextras.quality.radiometric_analysis.custom_dataclasses import (
    RadiometricAnalysisConfig,
)

# syncing with logger
log = logging.getLogger("quality_analysis")

USER_AREPYEXTRAS_QUALITY_CONFIG_FILE = Path(
    appdirs.user_config_dir(), "AREPYEXTRAS_QUALITY", "arepyextras_quality_default_config.toml"
)
ENVIRONMENT_VARIABLE = "AREPYEXTRAS_QUALITY_CONFIG_FILE"


def toml_schema_validation(toml_content: dict):
    """Validation of input config file for Arepyextras Quality tool.

    Parameters
    ----------
    toml_content : dict
        dictionary containing the parsed toml content
    """

    json_schema = json.loads(config_schema)

    validate(toml_content, json_schema)


@dataclass
class DefaultConfig:
    """Default configuration setup for CLI tool equivalent to toml configuration file"""

    point_target_analysis: PointTargetAnalysisConfig = field(default_factory=PointTargetAnalysisConfig)
    radiometric_analysis: RadiometricAnalysisConfig = field(default_factory=RadiometricAnalysisConfig)
    nesz_analysis: NESZConfig = field(default_factory=NESZConfig)

    @staticmethod
    def from_toml(toml_file: Path) -> DefaultConfig:
        """Generating a DefaultConfig dataclass from a configuration toml file.

        Parameters
        ----------
        arg : Path
            path to the toml file

        Returns
        -------
        DefaultConfig
            output dataclass

        Raises
        ------
        ValueError
            if input toml is not valid, this error is raised
        """

        # loading toml file
        with open(toml_file, "r", encoding="UTF-8") as f_in:
            config = toml.load(f_in)

        # validating toml content
        toml_schema_validation(config)

        # accessing each field of the dictionary and storing its values in a new dataclass
        pta_config = config["point_target_analysis"] if "point_target_analysis" in config else None
        ra_config = config["radiometric_analysis"] if "radiometric_analysis" in config else None
        nesz_config = config["nesz_analysis"] if "nesz_analysis" in config else None

        try:
            out = DefaultConfig()
            if pta_config is not None:
                pta = PointTargetAnalysisConfig.from_dict(pta_config)
                out.point_target_analysis = pta
            if ra_config is not None:
                rad = RadiometricAnalysisConfig.from_dict(ra_config)
                out.radiometric_analysis = rad
            if nesz_config is not None:
                nesz = NESZConfig.from_dict(nesz_config)
                out.nesz_analysis = nesz

            return out

        except Exception as err:
            raise ValueError("Invalid toml file.") from err

    def dump_to_toml(self, out_file: Path) -> None:
        """Dumping to disk a .toml file from the dataclass instance.

        Parameters
        ----------
        out_file : Path
            path to the output .toml file
        """

        dtc_dict = asdict(self)
        # converting enum classes to their name in lower case
        dtc_dict["point_target_analysis"]["irf_parameters"]["masking_method"] = dtc_dict["point_target_analysis"][
            "irf_parameters"
        ]["masking_method"].name.lower()
        dtc_dict["radiometric_analysis"]["input_type"] = dtc_dict["radiometric_analysis"]["input_type"].name.lower()
        dtc_dict["radiometric_analysis"]["output_type"] = dtc_dict["radiometric_analysis"]["output_type"].name.lower()
        dtc_dict["radiometric_analysis"]["value"] = dtc_dict["radiometric_analysis"]["value"].name.lower()
        dtc_dict["radiometric_analysis"]["direction"] = dtc_dict["radiometric_analysis"]["direction"].name.lower()
        dtc_dict["radiometric_analysis"]["axis"] = dtc_dict["radiometric_analysis"]["axis"].name.lower()

        with open(out_file, "w", encoding="UTF-8") as f_out:
            toml.dump(dtc_dict, f_out)


def default_settings_filename(create_if_missing: bool = False) -> Path:
    """Getting the location of Arepyextras Quality CLI tool configuration file.

    Parameters
    ----------
    create_if_missing : bool, optional
        create the file if it is missing, by default False

    Returns
    -------
    Path
        path to the configuration toml file
    """

    filename = Path(
        os.getenv(
            key=ENVIRONMENT_VARIABLE,
            default=str(USER_AREPYEXTRAS_QUALITY_CONFIG_FILE),
        )
    )

    # creating the file if none is found
    if create_if_missing and not filename.exists():
        log.info("Default configuration file is missing. Creating a new one.")
        # creating all the folder structure up to the file to be generated
        filename.parent.mkdir(exist_ok=True, parents=True)
        # dumping the DefaultConfig with default attributes values
        default_conf = DefaultConfig()
        default_conf.dump_to_toml(filename)

    return filename
