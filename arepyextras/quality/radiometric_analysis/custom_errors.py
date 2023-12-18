# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Definition of custom errors for troubleshooting of Radiometric Analysis"""


class SwathNotFoundError(RuntimeError):
    """Swath provided as input was not found"""


class PolarizationNotFoundError(RuntimeError):
    """Selected polarization not found in product folder channels"""


class MultipleSwathError(RuntimeError):
    """There is more than one swath but none has been selected"""


class PixelTimesMismatchError(RuntimeError):
    """Mismatch between the pixel flag and the input azimuth/range type"""


class TimesDirectionMismatchError(RuntimeError):
    """Mismatch between the requested analysis direction and the input times/pixels"""


class InputMissingError(RuntimeError):
    """No input times/pixel have been provided"""


known_errors = [
    TimesDirectionMismatchError,
    MultipleSwathError,
    PixelTimesMismatchError,
    SwathNotFoundError,
    PolarizationNotFoundError,
]
