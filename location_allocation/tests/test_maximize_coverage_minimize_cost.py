"""
Tests for maximize coverage algorithm
"""

from scipy.spatial import distance_matrix
import numpy as np
from mip import OptimizationStatus

from location_allocation import (
    MAXIMIZE_COVERAGE_MINIMIZE_COST,
    maximize_coverage_minimize_cost,
    utils,
)


def test_maximize_coverage_minimize_cost_near(demand, facilities):

    cost_matrix = distance_matrix(demand, facilities)

    mcmclp = MAXIMIZE_COVERAGE_MINIMIZE_COST(
        demand,
        facilities,
        cost_matrix,
        cost_cutoff=5,
        facilities_to_choose=2,
    ).optimize()

    assert mcmclp.model.status == OptimizationStatus.OPTIMAL

    assert len(mcmclp.result.solution[0]) == 26
    assert len(mcmclp.result.solution[1]) == 26

    opt_facilities = facilities[list(mcmclp.result.solution.keys())]
    assert len(opt_facilities) == 2
    assert np.alltrue(opt_facilities[0] == [-5, 10])
    assert np.alltrue(opt_facilities[1] == [5, 10])
    # utils.plot_result(demand, mcmclp.result.solution, opt_facilities)


def test_maximize_coverage_minimize_cost_far(demand, facilities):

    cost_matrix = distance_matrix(demand, facilities)

    mcmclp = MAXIMIZE_COVERAGE_MINIMIZE_COST(
        demand,
        facilities,
        cost_matrix,
        cost_cutoff=7.5,
        facilities_to_choose=2,
    ).optimize()

    assert mcmclp.model.status == OptimizationStatus.OPTIMAL

    solution_keys = list(mcmclp.result.solution.keys())

    assert len(mcmclp.result.solution[solution_keys[0]]) == 22
    assert len(mcmclp.result.solution[solution_keys[1]]) == 38

    opt_facilities = facilities[solution_keys]
    assert len(opt_facilities) == 2
    assert np.alltrue(opt_facilities[0] == [5, 0])
    assert np.alltrue(opt_facilities[1] == [-5, 10])
    # utils.plot_result(demand, mcmclp.result.solution, opt_facilities)


def test_maximize_coverage_minimize_cost_func_near(demand, facilities):

    cost_matrix = distance_matrix(demand, facilities)

    model, result = maximize_coverage_minimize_cost(
        demand,
        facilities,
        cost_matrix,
        cost_cutoff=5,
        facilities_to_choose=2,
    )

    assert model.status == OptimizationStatus.OPTIMAL

    assert len(result.solution[0]) == 26
    assert len(result.solution[1]) == 26

    opt_facilities = facilities[list(result.solution.keys())]
    assert len(opt_facilities) == 2
    assert np.alltrue(opt_facilities[0] == [-5, 10])
    assert np.alltrue(opt_facilities[1] == [5, 10])
    # utils.plot_result(demand, result.solution, opt_facilities)


def test_maximize_coverage_minimize_cost_func_far(demand, facilities):

    cost_matrix = distance_matrix(demand, facilities)

    model, result = maximize_coverage_minimize_cost(
        demand,
        facilities,
        cost_matrix,
        cost_cutoff=7.5,
        facilities_to_choose=2,
    )

    assert model.status == OptimizationStatus.OPTIMAL

    solution_keys = list(result.solution.keys())

    amount = sum(
        [len(result.solution[solution_keys[0]]), len(result.solution[solution_keys[1]])]
    )
    assert 58 == amount or amount == 60

    opt_facilities = facilities[solution_keys]
    assert len(opt_facilities) == 2
    # utils.plot_result(demand, result.solution, opt_facilities)
