from ._maximize_coverage import maximize_coverage, MAXIMIZE_COVERAGE
from ._maximize_capacitated_coverage import (
    maximize_capacitated_coverage,
    MAXIMIZE_CAPACITATED_COVERAGE,
)

__version__ = "0.1.0"
"""
The :mod:`location_allocation` module gathers popular location allocation problems.
"""

__all__ = [
    "MAXIMIZE_COVERAGE",
    "maximize_coverage",
    "MAXIMIZE_CAPACITATED_COVERAGE",
    "maximize_capacitated_coverage",
]
