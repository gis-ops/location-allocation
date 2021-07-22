"""
Tests for maximize coverage algorithm
"""

from scipy.spatial import distance_matrix
import numpy as np
from mip import OptimizationStatus

from location_allocation import (
    MAXIMIZE_COVERAGE_MINIMIZE_FACILITIES,
    utils,
)


def test_maximize_coverage_minimize_cost_near(demand, facilities):

    cost_matrix = distance_matrix(demand, facilities)

    mcmclp = MAXIMIZE_COVERAGE_MINIMIZE_FACILITIES(
        demand,
        facilities,
        cost_matrix,
        cost_cutoff=5,
        max_facilities=2,
    ).optimize()

    assert mcmclp.model.status == OptimizationStatus.OPTIMAL

    res = sum([len(sub) for sub in mcmclp.result.solution.values()])
    assert res == 51

    opt_facilities = facilities[list(mcmclp.result.solution.keys())]
    assert len(opt_facilities) == 2
    assert np.alltrue(opt_facilities[0] == [-5, 10])
    assert np.alltrue(opt_facilities[1] == [5, 10])
    #utils.plot_result(demand, mcmclp.result.solution, opt_facilities)


def test_maximize_coverage_minimize_cost_far(demand, facilities):

    cost_matrix = distance_matrix(demand, facilities)

    mcmclp = MAXIMIZE_COVERAGE_MINIMIZE_FACILITIES(
        demand,
        facilities,
        cost_matrix,
        cost_cutoff=15,
        max_facilities=2,
    ).optimize()

    assert mcmclp.model.status == OptimizationStatus.OPTIMAL

    opt_facilities = facilities[list(mcmclp.result.solution.keys())]

    res = sum([len(sub) for sub in mcmclp.result.solution.values()])
    assert res == 119
    # utils.plot_result(demand, mcmclp.result.solution, opt_facilities)
