# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Command Line Interface command for NESZ Analysis"""
import logging
import sys
import time
from pathlib import Path

import art
import click

from arepyextras.quality.configuration.arepyextras_quality_init_config import (
    DefaultConfig,
)
from arepyextras.quality.nesz_analysis.analysis import (
    nesz_productfolder_wrapper,
    save_to_netcdf,
)

# syncing with logger
log = logging.getLogger("quality_analysis")

# creating a decorator to pass a DefaultConfig dataclass object between commands
share_config = click.make_pass_decorator(DefaultConfig)


@click.command(name="nesz_analysis")
@click.option(
    "--product_folder",
    "-pf",
    required=True,
    type=click.Path(path_type=Path, exists=True, dir_okay=True),
    help="Path to the product folder to be analyzed (SLC, GRD)",
)
@click.option(
    "--output_directory",
    "-out",
    default=None,
    required=True,
    type=click.Path(path_type=Path, exists=True, dir_okay=True),
    help="Path to the folder where to save output data",
)
@click.option(
    "--graphs",
    "-g",
    default=False,
    is_flag=True,
    type=bool,
    help="Flag to generate graphical output at the end of the analysis",
)
@share_config
def nesz_analysis(config: DefaultConfig, product_folder: Path, output_directory: Path, graphs: bool):
    """NESZ Analysis CLI"""

    if graphs:
        try:
            import arepyextras.quality.nesz_analysis.graphical_output as gpo

        except ImportError:
            log.critical('Install cli requirements "pip install arepyextras-quality[graphs]"')
            sys.exit(1)

    # inheriting configuration settings from group command in CLI main
    config_nesz = config.nesz_analysis

    log.info(f"Selected product is: {product_folder}")

    txt = art.text2art("NESZ  Analysis", font="small")
    click.echo(txt + "\n")

    try:
        start = time.perf_counter_ns()
        analysis_output = nesz_productfolder_wrapper(pf_path=product_folder, config=config_nesz)
        save_to_netcdf(data=analysis_output, out_path=output_directory)
        if graphs:
            log.info("Generating graphical output...")
            for item in analysis_output:
                gpo.nesz_graphs(data=item, output_dir=output_directory)

        elapsed = (time.perf_counter_ns() - start) / 1e9
        log.info(f"NESZ analysis completed in {elapsed} s.")
    except Exception as err:
        log.error(type(err).__name__)
        log.critical(err)
        log.info("NESZ Analysis failed.")
