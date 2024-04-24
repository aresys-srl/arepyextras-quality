# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Quality interferometry module custom dataclasses"""
from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass
class InterferometricConfig:
    """Interferometric analysis configuration"""

    # if set to True, it means that the input product contains the interferogram and not the coherence map already computed
    enable_coherence_computation: bool = False
    # coherence computation kernel, it can be a single value (square) or rectangular kernel shape
    coherence_kernel: int | tuple[int, int] = 15
    # number of blocks by which divide the coherence array along azimuth direction to compute histograms
    azimuth_blocks_number: int | None = None
    # number of blocks by which divide the coherence array along range direction to compute histograms
    range_blocks_number: int | None = None
    # number of coherence bins to compute histograms
    coherence_bins_number: int = 80

    @classmethod
    def from_dict(cls, arg: dict) -> InterferometricConfig:
        """Creating a InterferometricConfig object by conversion from a dictionary.

        Args:
            arg (dict): dictionary with keys equal to the InterferometricConfig ones

        Returns:
            InterferometricConfig: InterferometricConfig object
        """
        inter_obj = cls()
        dict_in = arg.copy()

        try:
            dtc_fields = [f.name for f in fields(inter_obj)]
            for key, value in dict_in.items():
                if key in dtc_fields:
                    if isinstance(value, list):
                        setattr(inter_obj, key, tuple(value))
                    else:
                        setattr(inter_obj, key, value)

            return inter_obj

        except Exception as err:
            raise ValueError("Invalid dictionary structure.") from err
