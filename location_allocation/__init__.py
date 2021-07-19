from ._maximize_coverage import maximize_coverage, MAXIMIZE_COVERAGE
from ._maximize_coverage_capacitated import (
    maximize_coverage_capacitated,
    MAXIMIZE_COVERAGE_CAPACITATED,
)
from ._maximize_coverage_minimize_cost import (
    maximize_coverage_minimize_cost,
    MAXIMIZE_COVERAGE_MINIMIZE_COST,
)

from ._maximize_coverage_minimize_facilities import (
    maximize_coverage_minimize_facilities,
    MAXIMIZE_COVERAGE_MINIMIZE_FACILITIES,
)

from .utils import custom_logger

custom_logger("la", debug=True)

__version__ = "0.1.0"
"""
The :mod:`location_allocation` module gathers popular location allocation problems.
"""

__all__ = [
    "MAXIMIZE_COVERAGE",
    "maximize_coverage",
    "MAXIMIZE_COVERAGE_CAPACITATED",
    "maximize_coverage_capacitated",
    "MAXIMIZE_COVERAGE_MINIMIZE_COST",
    "maximize_coverage_minimize_cost",
    "MAXIMIZE_COVERAGE_MINIMIZE_FACILITIES",
    "maximize_coverage_minimize_facilities",
]
