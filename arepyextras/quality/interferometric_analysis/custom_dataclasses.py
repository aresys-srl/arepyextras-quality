# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Quality interferometry module custom dataclasses"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from arepyextras.quality.core.generic_dataclasses import SARPolarization


class CoherenceGraphMode(Enum):
    """Coherence graphs complex coherence plot method"""

    MAGNITUDE = "magnitude"
    PHASE = "phase"


@dataclass
class InterferometricCoherenceOutput:
    """Interferometric Coherence computation output"""

    channel_name: str
    swath: str
    burst: int
    polarization: SARPolarization
    coherence: np.ndarray
    coherence_histograms: InterferometricCoherence2DHistograms


@dataclass
class InterferometricCoherence2DHistograms:
    """Interferometric Coherence 2D histograms output"""

    coherence_bin_edges: np.ndarray
    azimuth_histogram: np.ndarray
    range_histogram: np.ndarray
