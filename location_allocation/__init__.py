from ._maximize_coverage import MAXIMIZE_COVERAGE
from ._maximize_coverage_capacitated import MAXIMIZE_COVERAGE_CAPACITATED
from ._maximize_coverage_minimize_cost import MAXIMIZE_COVERAGE_MINIMIZE_COST
from ._maximize_coverage_minimize_facilities import (
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
    "MAXIMIZE_COVERAGE_CAPACITATED",
    "MAXIMIZE_COVERAGE_MINIMIZE_COST",
    "MAXIMIZE_COVERAGE_MINIMIZE_FACILITIES",
]
