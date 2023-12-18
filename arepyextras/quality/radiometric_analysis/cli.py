# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Command Line Interface command for Radiometric Analysis"""
import logging
import sys
import time
from pathlib import Path

import art
import click
from arepytools.io.metadata import EPolarization
from arepytools.timing.precisedatetime import PreciseDateTime

import arepyextras.quality.radiometric_analysis.custom_dataclasses as rdt
import arepyextras.quality.radiometric_analysis.custom_errors as ra_err
from arepyextras.quality.configuration.arepyextras_quality_init_config import (
    DefaultConfig,
)
from arepyextras.quality.core import custom_errors as c_err
from arepyextras.quality.radiometric_analysis.analysis import (
    radiometric_analysis as rdma,
)

# syncing with logger
log = logging.getLogger("quality_analysis")

# creating a decorator to pass a DefaultConfig dataclass object between commands
share_config = click.make_pass_decorator(DefaultConfig)

# list of managed known errors
known_errors = ra_err.known_errors + c_err.known_errors


class AzimuthType(click.ParamType):
    """Custom click type for validating input azimuth times in the form of Precise Date Time"""

    name = "azimuth_times"

    def convert(self, value, param, ctx):
        try:
            str_in = value.split(", ")
            pdt_list = list(map(PreciseDateTime.from_utc_string, str_in))

            return pdt_list

        except ValueError:
            self.fail(f"{value!r} is not compliant to Precise Date Time standard", param, ctx)


class RangeType(click.ParamType):
    """Custom click type for validating input range times in the form of floats"""

    name = "range_times"

    def convert(self, value, param, ctx):
        try:
            str_in = value.split(", ")
            pdt_list = list(map(float, str_in))

            return pdt_list

        except ValueError:
            self.fail(f"{value!r} is not composed by floats", param, ctx)


class RadiometricInput(click.ParamType):
    """Custom click type to split input values from string to single values"""

    name = "radiometric_input"

    def convert(self, value, param, ctx):
        try:
            str_in = value.split(", ")

            return str_in

        except ValueError:
            self.fail(f"{value!r} wrong input format", param, ctx)


class PolarizationInput(click.ParamType):
    """Custom click type to switch user input string to EPolarization enum class"""

    name = "polarization_input"

    def convert(self, value, param, ctx):
        try:
            str_in = value.upper().strip()
            if "/" not in str_in:
                str_in = str_in[0] + "/" + str_in[1]
            pol = EPolarization(str_in)

            return pol

        except ValueError:
            self.fail(
                f"{value!r} wrong input format, polarization must be in the form (case insensitive): 'hh' or 'h/h'",
                param,
                ctx,
            )


@click.command(name="radiometric_analysis")
@click.option(
    "--product_folder",
    "-pf",
    required=True,
    type=click.Path(path_type=Path, exists=True, dir_okay=True),
    help="Path to the product folder to be analyzed (SLC, GRD)",
)
@click.option(
    "--azimuth_times",
    "-az",
    type=RadiometricInput(),
    help='Azimuth times/pixel indexes specified using a string " ", separated by commas',
)
@click.option(
    "--range_times",
    "-rng",
    type=RadiometricInput(),
    help='Range times/pixel indexes specified using a string " ", separated by commas',
)
@click.option(
    "--pixels",
    "-p",
    default=False,
    is_flag=True,
    type=bool,
    help="Flag to prompt azimuth and range values in pixel coordinates",
)
@click.option(
    "--swath_name",
    "-s",
    type=click.STRING,
    help="Swath name to be provided for multiple swath products",
)
@click.option(
    "--polarization",
    "-pol",
    type=PolarizationInput(),
    help="Polarization to be analyzed",
)
@click.option(
    "--output_directory",
    "-out",
    default=None,
    required=True,
    type=click.Path(path_type=Path, exists=True, dir_okay=True),
    help="Path to the folder where to save output data",
)
@share_config
def radiometric_analysis(
    config: DefaultConfig,
    product_folder: Path,
    azimuth_times: str,
    range_times: str,
    pixels: bool,
    swath_name: str,
    polarization: EPolarization,
    output_directory: Path,
):
    """Radiometric Analysis at provided azimuth/range times"""

    try:
        import arepyextras.quality.radiometric_analysis.graphical_output as ragpo

    except ImportError:
        log.critical('Install cli requirements "pip install arepyextras-quality[graphs]"')
        sys.exit(1)

    # inheriting configuration settings from group command in CLI main
    config = config.radiometric_analysis

    log.info(f"Selected product is: {product_folder}")

    if pixels:
        log.info("Azimuth and Range values provided as pixel coordinates.")
        if azimuth_times is not None:
            azimuth_times = list(map(int, azimuth_times))
        if range_times is not None:
            range_times = list(map(int, range_times))
    else:
        log.info("Azimuth and Range values provided as times.")
        if azimuth_times is not None:
            azimuth_times = list(map(PreciseDateTime.from_utc_string, azimuth_times))
        if range_times is not None:
            range_times = list(map(float, range_times))

    if azimuth_times is not None:
        log.info(f"Selected azimuth values are: {azimuth_times}")
    if range_times is not None:
        log.info(f"Selected range values are: {range_times}")

    if range_times is not None and azimuth_times is None:
        config.direction = rdt.RadiometricAnalysisDirection.AZIMUTH

    if azimuth_times is not None and range_times is None:
        config.direction = rdt.RadiometricAnalysisDirection.RANGE

    if polarization is not None:
        log.info(f"Only polarization {polarization.value} will be analyzed")

    if swath_name is not None:
        swath_name = swath_name.upper()
        log.info(f"Only Swath {swath_name} will be analyzed")

    txt = art.text2art("Radiometric  Analysis", font="small")
    click.echo(txt + "\n")

    try:
        start = time.perf_counter_ns()
        output_profiles, projection = rdma(
            product_folder=product_folder,
            azimuth_times=azimuth_times,
            range_times=range_times,
            is_pixel=pixels,
            analysis_config=config,
            swath_name=swath_name,
            selected_polarization=polarization,
        )
        if (output_profiles is not None) and (projection is not None):
            log.info("Generating graphical output...")
            grph_out_dir = output_directory.joinpath("graphs")
            Path.mkdir(grph_out_dir, exist_ok=True)

            ragpo.radiometric_profiles(data=output_profiles, config=config, projection=projection, out_dir=grph_out_dir)

            elapsed = (time.perf_counter_ns() - start) / 1e9
            log.info(f"Radiometric Analysis completed in {elapsed} s.")
        else:
            log.critical("Radiometric analysis failed. Please check your input data")

    except known_errors as err:
        log.error(type(err).__name__)
        log.critical(err)
        log.info("Radiometrical Analysis failed.")

    except Exception as err:
        log.critical(err)
        log.info("Radiometrical Analysis failed.")
