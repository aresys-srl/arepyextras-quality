# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Command Line Interface for Arepyextras Quality Tool"""
import logging
from pathlib import Path
from typing import Optional

import click

from arepyextras.quality.__init__ import __version__ as VERSION
from arepyextras.quality.configuration.arepyextras_quality_init_config import (
    DefaultConfig,
    default_settings_filename,
)
from arepyextras.quality.nesz_analysis.cli import nesz_analysis
from arepyextras.quality.point_targets_analysis.cli import target_analysis
from arepyextras.quality.radiometric_analysis.cli import radiometric_analysis

version_option = click.version_option(VERSION, help="Show CLI version and exit")

# creating a decorator to pass a DefaultConfig dataclass object between commands
share_config = click.make_pass_decorator(DefaultConfig)

# syncing with logger
log = logging.getLogger("quality_analysis")


@click.group(
    context_settings=dict(
        help_option_names=["-h", "--help"],
    )
)
@version_option
@click.pass_context
@click.option(
    "-cfg",
    "--config",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    help="Path to the configuration file with settings.",
)
def arepyextras_quality(ctx: click.Context, config: Optional[Path]):
    """CLI tool for SAR products quality analysis"""
    click.echo("Starting application...\n")
    if config is None:
        log.info("Configuration not provided. Searching for default one...")
        config = default_settings_filename(create_if_missing=True)
        log.info(f"Default configuration used is: {config}")
    else:
        log.info("Using the custom configuration file provided.")

    ctx.ensure_object(DefaultConfig)
    ctx.obj = DefaultConfig.from_toml(config)


arepyextras_quality.add_command(target_analysis)
arepyextras_quality.add_command(radiometric_analysis)
arepyextras_quality.add_command(nesz_analysis)
