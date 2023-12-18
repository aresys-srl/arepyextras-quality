# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Custom errors definition for better arepyextras-quality module troubleshooting"""


class AzimuthExceedsBoundariesError(ValueError):
    """Selected Azimuth index in ROI extraction from Swath exceeds boundaries"""


class RangeExceedsBoundariesError(ValueError):
    """Selected Range index in ROI extraction from Swath exceeds boundaries"""


class SideLobesDirectionsEstimationError(RuntimeError):
    """Could not evaluate the side lobes directions values"""


class CoordinatesOutOfBounds(RuntimeError):
    """Input pixel/time coordinate is out of swath bounds"""


known_errors = [AzimuthExceedsBoundariesError, RangeExceedsBoundariesError]
