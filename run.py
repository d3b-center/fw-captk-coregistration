#!/usr/bin/env python
"""The run script."""
import logging
import sys

from flywheel_gear_toolkit import GearToolkitContext

from gear_functions.main import run

log = logging.getLogger(__name__)


def main(context: GearToolkitContext) -> None:  # pragma: no cover
    """Run and exit with the gear's exit code."""
    sys.exit(run(context))


if __name__ == "__main__":  # pragma: no cover
    with GearToolkitContext() as gear_context:
        gear_context.init_logging()
        main(gear_context)
