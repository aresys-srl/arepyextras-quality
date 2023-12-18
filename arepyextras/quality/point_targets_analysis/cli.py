# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Command Line Interface command for Point Target Analysis"""
import logging
import sys
import time
from pathlib import Path

import art
import click

from arepyextras.quality.configuration.arepyextras_quality_init_config import (
    DefaultConfig,
)
from arepyextras.quality.point_targets_analysis.analysis import (
    point_target_productfolder_wrapper,
)

# syncing with logger
log = logging.getLogger("quality_analysis")

# creating a decorator to pass a DefaultConfig dataclass object between commands
share_config = click.make_pass_decorator(DefaultConfig)


@click.command(name="target_analysis")
@click.option(
    "--product_folder",
    "-pf",
    required=True,
    type=click.Path(path_type=Path, exists=True, dir_okay=True),
    help="Path to the product folder to be analyzed (SLC, GRD)",
)
@click.option(
    "--point_target_data",
    "-pt",
    required=True,
    type=click.Path(path_type=Path, exists=True, dir_okay=True),
    help="Path to the point target XML file or binary folder",
)
@click.option(
    "--output_directory",
    "-out",
    default=None,
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
def target_analysis(
    config: DefaultConfig, product_folder: Path, point_target_data: Path, output_directory: Path, graphs: bool
):
    """Point Target Analysis (IRF, Localization and RCS)"""

    # inheriting configuration settings from group command in CLI main
    config = config.point_target_analysis

    if graphs:
        try:
            import arepyextras.quality.point_targets_analysis.graphical_output as ptgpo

        except ImportError:
            log.critical('Install cli requirements "pip install arepyextras-quality[graphs]"')
            sys.exit(1)

    if output_directory is None:
        log.info("Output directory not specified. Output data will be saved in Product Folder.")
        output_directory = product_folder.joinpath("PTA Output")
        Path.mkdir(output_directory, exist_ok=True)
    else:
        log.info(f"Output folder is: {output_directory}")

    if point_target_data.is_dir():
        log.info(f"Binary Point Target folder has been provided:\n {point_target_data}")
    else:
        log.info(f"Selected point target file is: {point_target_data}")

    log.info(f"Selected product is: {product_folder}")

    txt = art.text2art("Point  Target  Analysis", font="doom")
    click.echo(txt + "\n")

    start = time.perf_counter_ns()
    results_df, graph_data = point_target_productfolder_wrapper(
        pf_path=product_folder, point_targets_path=point_target_data, config=config
    )

    # saving results to csv file
    results_df.to_csv(output_directory.joinpath("point_target_analysis_results.csv"), index=False)
    click.echo(results_df)

    # graphical output management
    if graphs:
        log.info("Generating graphical output...")
        grph_out_dir = output_directory.joinpath("graphs")
        Path.mkdir(grph_out_dir, exist_ok=True)

        for item in graph_data:
            this_graph = f"Target {item.target}, Channel {item.channel}"
            log.info("Generating graphical output for " + this_graph)
            data_val = results_df.query(f"target == {item.target} & channel == {item.channel}").to_dict("records")[0]
            label = (
                f"target_{data_val['target']}_{data_val['swath']}_"
                + f"polarization_{data_val['polarization'].replace('/','')}"
            )

            if config.generate_static_graphs:
                try:
                    # IRF graphs
                    ptgpo.irf_parameters(
                        data_graph=item.irf,
                        data_values=data_val,
                        label=label,
                        out_dir=grph_out_dir,
                    )
                except Exception:
                    log.error("Could not generate IRF images for " + this_graph)

                if config.perform_rcs:
                    # RCS graphs
                    try:
                        ptgpo.rcs_parameters(data_graph=item.rcs, label=label, out_dir=grph_out_dir)
                    except Exception:
                        log.error("Could not generate RCS images for " + this_graph)

            if config.generate_interactive_graphs:
                # interactive full plot
                try:
                    ptgpo.interactive_graphs(
                        irf_data_graph=item.irf,
                        rcs_data_graph=item.rcs,
                        data_values=data_val,
                        label=label,
                        out_dir=grph_out_dir,
                    )
                except Exception:
                    log.error("Could not generate interactive plot for " + this_graph)

    elapsed = (time.perf_counter_ns() - start) / 1e9
    log.info(f"Point Target Analysis completed in {elapsed} s.")
