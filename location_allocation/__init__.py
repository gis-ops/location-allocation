from ._maximize_coverage import MaximizeCoverage
from ._maximize_coverage_capacitated import MaximizeCoverageCapacitated
from ._maximize_coverage_minimize_cost import MaximizeCoverageMinimizeCost
from ._maximize_coverage_minimize_facilities import (
    MaximizeCoverageMinimizeFacilities,
)

__version__ = "0.1.0"
"""
The :mod:`location_allocation` module gathers popular location allocation problems.
"""

__all__ = [
    "MaximizeCoverage",
    "MaximizeCoverageCapacitated",
    "MaximizeCoverageMinimizeCost",
    "MaximizeCoverageMinimizeFacilities",
]
