# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Saving radiometric output data to netCDF4 file"""

from pathlib import Path

import numpy as np
import pandas as pd
from netCDF4 import Dataset

import arepyextras.quality.radiometric_analysis.custom_dataclasses as rdt


def dataclass2netcdf(data: list[rdt.RadiometricAnalysisOutput], out_path: Path) -> None:
    """Function to save the radiometric analysis computed profiles and metadata into a netCDF4 file.

    Parameters
    ----------
    data : list[rdt.RadiometricAnalysisOutput]
        list of RadiometricAnalysisOutput dataclasses
    out_path : Path
        output folder path
    """

    # data conversion to dataframe
    data_frame = pd.DataFrame(data)
    data_frame.rename(columns={"profile": "original_profile"}, inplace=True)

    # file init
    rootgrp = Dataset(out_path.joinpath("results_dump").with_suffix(".nc"), "w", format="NETCDF4")
    rootgrp.set_auto_mask(False)

    # defining groups
    az_grp = rootgrp.createGroup("azimuth")
    az_grp.set_auto_mask(False)
    rng_grp = rootgrp.createGroup("range")
    rng_grp.set_auto_mask(False)

    # creating dimensions
    rootgrp.createDimension("swath", data_frame["swath"].unique().size)  # swath_dim
    rootgrp.createDimension("polarization", data_frame["polarization"].unique().size)  # polarization_dim
    rootgrp.createDimension("profiles", 2)  # profiles_dim
    rootgrp.createDimension("axis", 1)  # axis_dim
    rootgrp.createDimension("range_times", None)  # rng_times_dim
    rootgrp.createDimension("azimuth_times", None)  # az_times_dim
    rootgrp.createDimension("profile_len_az", None)  # profile_len_az_dim
    rootgrp.createDimension("profile_len_rng", None)  # profile_len_rng_dim

    # creating main variables
    swath = rootgrp.createVariable("swath", str, ("swath",))
    polarization = rootgrp.createVariable("polarization", str, ("polarization",))
    profiles = rootgrp.createVariable("profiles", str, ("profiles",))
    profiles.unit = data_frame.value_type.unique()[0].name.capitalize()
    profiles.radiometric_type = " ".join(
        list(map(str.capitalize, data_frame.radiometric_type.unique()[0].name.split("_")))
    )
    axis = rootgrp.createVariable("axis", np.float64, ("axis",))
    axis.unit = data_frame.axis_format.unique()[0].name.capitalize()
    times_rng = rootgrp.createVariable("range_times", np.float64, ("range_times"))
    times_rng.unit = "s"
    times_az = rootgrp.createVariable("azimuth_times", str, ("azimuth_times"))
    times_az.unit = "PreciseDateTime"
    radiometric_output_rng = rng_grp.createVariable(
        "radiometric_output", np.float32, ("swath", "polarization", "profiles", "range_times", "profile_len_rng")
    )
    radiometric_axis_rng = rng_grp.createVariable(
        "output_axis", np.float64, ("swath", "polarization", "axis", "range_times", "profile_len_rng")
    )
    radiometric_output_az = az_grp.createVariable(
        "radiometric_output", np.float32, ("swath", "polarization", "profiles", "azimuth_times", "profile_len_az")
    )
    radiometric_axis_az = az_grp.createVariable(
        "output_axis", np.float64, ("swath", "polarization", "axis", "azimuth_times", "profile_len_az")
    )

    # filling common variables
    swaths = data_frame["swath"].unique().tolist()
    for swt_num, swt in enumerate(swaths):
        swath[swt_num] = swt
    pols = data_frame["polarization"].unique().tolist()
    for p_num, pol in enumerate(pols):
        polarization[p_num] = pol
    profs = ["smoothed_profile", "original_profile"]
    for p_num, prf in enumerate(profs):
        profiles[p_num] = prf
    times = data_frame["time"].unique().tolist()
    az_times = list(filter(lambda elm: isinstance(elm, str), times))
    rng_times = list(filter(lambda elm: isinstance(elm, float), times))
    for tr_num, trng in enumerate(rng_times):
        times_rng[tr_num] = trng
    for az_num, azmth in enumerate(az_times):
        times_az[az_num] = azmth

    # filling specific profile variables
    for s_num, swt in enumerate(swath):
        for p_num, pol in enumerate(pols):
            selected_data = data_frame.query("swath == @swt & polarization == @pol")
            rng_data = selected_data.query("direction == @rdt.RadiometricAnalysisDirection.AZIMUTH")
            az_data = selected_data.query("direction == @rdt.RadiometricAnalysisDirection.RANGE")

            for pp_num, prf in enumerate(profs):
                for az_num, azmth in enumerate(az_times):
                    radiometric_output_az[s_num, p_num, pp_num, az_num, :] = az_data[az_data["time"] == azmth][
                        prf
                    ].to_list()[0]
                for rng_num, rng in enumerate(rng_times):
                    radiometric_output_rng[s_num, p_num, pp_num, rng_num, :] = rng_data[rng_data["time"] == rng][
                        prf
                    ].to_list()[0]
            for az_num, azmth in enumerate(az_times):
                radiometric_axis_az[s_num, p_num, 0, az_num, :] = az_data[az_data["time"] == azmth]["axis"].to_list()[0]
            for rng_num, rng in enumerate(rng_times):
                radiometric_axis_rng[s_num, p_num, 0, rng_num, :] = rng_data[rng_data["time"] == rng]["axis"].to_list()[
                    0
                ]

    rootgrp.close()
