"""
Tests for maximize capacitated coverage algorithm
"""

from scipy.spatial import distance_matrix
from mip import OptimizationStatus
import numpy as np

from location_allocation import (
    MAXIMIZE_CAPACITATED_COVERAGE,
    maximize_capacitated_coverage,
    utils,
)


def test_maximize_capacitated_coverage_full(demand, facilities, capacities):

    cost_matrix = distance_matrix(demand, facilities)

    mcclp = MAXIMIZE_CAPACITATED_COVERAGE(
        demand,
        facilities,
        capacities,
        cost_matrix,
        facilities_to_choose=2,
        cost_cutoff=3,
    )
    mcclp.optimize()

    assert mcclp.model.status == OptimizationStatus.OPTIMAL
    for k, v in mcclp.result.solution.items():
        assert len(v) == 15

    opt_facilities = facilities[list(mcclp.result.solution.keys())]
    assert np.alltrue(opt_facilities[0] == [-1, 2])
    assert np.alltrue(opt_facilities[1] == [1, 2])

    # utils.plot_result(demand, mcclp.result.solution, opt_facilities)


def test_maximize_capacitated_coverage_partial(demand, facilities, capacities):

    cost_matrix = distance_matrix(demand, facilities)

    mcclp = MAXIMIZE_CAPACITATED_COVERAGE(
        demand,
        facilities,
        capacities,
        cost_matrix,
        facilities_to_choose=2,
        cost_cutoff=0.5,
    )
    mcclp.optimize()

    assert mcclp.model.status == OptimizationStatus.OPTIMAL
    for k, v in mcclp.result.solution.items():
        assert len(v) == 8

    opt_facilities = facilities[list(mcclp.result.solution.keys())]
    assert np.alltrue(opt_facilities[0] == [-1, 2])
    assert np.alltrue(opt_facilities[1] == [1, 2])

    # utils.plot_result(demand, mcclp.result.solution, opt_facilities)


def test_maximize_capacitated_coverage_fnc_full(demand, facilities, capacities):

    cost_matrix = distance_matrix(demand, facilities)

    model, result = maximize_capacitated_coverage(
        demand,
        facilities,
        capacities,
        cost_matrix,
        facilities_to_choose=2,
        cost_cutoff=3,
    )

    assert model.status == OptimizationStatus.OPTIMAL
    for k, v in result.solution.items():
        assert len(v) == 15

    opt_facilities = facilities[list(result.solution.keys())]
    assert np.alltrue(opt_facilities[0] == [-1, 2])
    assert np.alltrue(opt_facilities[1] == [1, 2])

    # utils.plot_result(demand, mcclp.result.solution, opt_facilities)


def test_maximize_capacitated_coverage_fnc_partial(demand, facilities, capacities):

    cost_matrix = distance_matrix(demand, facilities)

    model, result = maximize_capacitated_coverage(
        demand,
        facilities,
        capacities,
        cost_matrix,
        facilities_to_choose=2,
        cost_cutoff=0.5,
    )

    assert model.status == OptimizationStatus.OPTIMAL
    for k, v in result.solution.items():
        assert len(v) == 8

    opt_facilities = facilities[list(result.solution.keys())]
    assert np.alltrue(opt_facilities[0] == [-1, 2])
    assert np.alltrue(opt_facilities[1] == [1, 2])

    # utils.plot_result(demand, mcclp.result.solution, opt_facilities)
