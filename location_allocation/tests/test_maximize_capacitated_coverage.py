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

    cost_cutoff = 15

    mcclp = MAXIMIZE_CAPACITATED_COVERAGE(
        demand,
        facilities,
        cost_matrix,
        cost_cutoff,
        capacities=capacities,
        facilities_to_choose=2,
    )
    mcclp.optimize()

    assert mcclp.model.status == OptimizationStatus.OPTIMAL
    for k, v in mcclp.result.solution.items():
        assert len(v) == 15

    opt_facilities = facilities[list(mcclp.result.solution.keys())]
    assert np.alltrue(opt_facilities[0] == [-5, 10])
    assert np.alltrue(opt_facilities[1] == [5, 10])

    # utils.plot_result(demand, mcclp.result.solution, opt_facilities)


def test_maximize_capacitated_coverage_partial(demand, facilities, capacities):

    cost_matrix = distance_matrix(demand, facilities)

    mcclp = MAXIMIZE_CAPACITATED_COVERAGE(
        demand,
        facilities,
        cost_matrix,
        cost_cutoff=2.5,
        capacities=capacities,
        facilities_to_choose=2,
    )
    mcclp.optimize()

    assert mcclp.model.status == OptimizationStatus.OPTIMAL
    for k, v in mcclp.result.solution.items():
        assert len(v) == 8

    opt_facilities = facilities[list(mcclp.result.solution.keys())]
    assert np.alltrue(opt_facilities[0] == [-5, 10])
    assert np.alltrue(opt_facilities[1] == [5, 10])

    # utils.plot_result(demand, mcclp.result.solution, opt_facilities)


def test_maximize_capacitated_coverage_fnc_full(demand, facilities, capacities):

    cost_matrix = distance_matrix(demand, facilities)

    model, result = maximize_capacitated_coverage(
        demand,
        facilities,
        cost_matrix,
        cost_cutoff=15,
        capacities=capacities,
        facilities_to_choose=2,
    )

    assert model.status == OptimizationStatus.OPTIMAL
    opt_facilities = facilities[list(result.solution.keys())]
    assert np.alltrue(opt_facilities[0] == [-5, 10])
    assert np.alltrue(opt_facilities[1] == [5, 10])

    # for debugging
    print("objective value: ", model.objective_value)
    for v in model.vars:
        if v.name[0] == "z" and v.x == 1:
            print(v.name, " ", v.x)
        if v.name[0] == "x" and v.x == 1:
            print(v.name, " ", v.x)

    assert model.objective_value == 30
    for k, v in result.solution.items():
        assert len(v) == 15

    # utils.plot_result(demand, mcclp.result.solution, opt_facilities)


def test_maximize_capacitated_coverage_fnc_partial(demand, facilities, capacities):

    cost_matrix = distance_matrix(demand, facilities)

    model, result = maximize_capacitated_coverage(
        demand,
        facilities,
        cost_matrix,
        cost_cutoff=2.5,
        capacities=capacities,
        facilities_to_choose=2,
    )

    assert model.status == OptimizationStatus.OPTIMAL
    for k, v in result.solution.items():
        assert len(v) == 8

    opt_facilities = facilities[list(result.solution.keys())]
    assert np.alltrue(opt_facilities[0] == [-5, 10])
    assert np.alltrue(opt_facilities[1] == [5, 10])

    # utils.plot_result(demand, mcclp.result.solution, opt_facilities)
