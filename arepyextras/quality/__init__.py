# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
Quality package
---------------
"""

import pkgutil

__version__ = "1.1.0"

# configuration json schema for validation
config_schema = pkgutil.get_data(__name__, "resources/config_schema.json")
