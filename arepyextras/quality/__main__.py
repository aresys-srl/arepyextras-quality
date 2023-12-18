# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Main CLI script"""
import logging
import sys

import arepyextras.quality.configuration.custom_logger as clg


def main():
    """Main function to launch the python SQT CLI program"""
    # setup custom logger
    log = logging.getLogger("quality_analysis")
    log.setLevel("DEBUG")
    log.addHandler(clg.MyHandler())

    try:
        from arepyextras.quality.cli import arepyextras_quality

        arepyextras_quality()

    except ImportError:
        print('Install cli requirements "pip install arepyextras-quality[cli]"')
        sys.exit(1)


if __name__ == "__main__":
    main()
