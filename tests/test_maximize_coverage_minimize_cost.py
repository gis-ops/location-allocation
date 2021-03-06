import numpy as np
from mip import OptimizationStatus
from scipy.spatial import distance_matrix

from location_allocation import MaximizeCoverageMinimizeCost


def test_maximize_coverage_minimize_cost_near(demand, facilities):

    dist_matrix = distance_matrix(demand, facilities)

    mcmclp = MaximizeCoverageMinimizeCost(
        demand,
        facilities,
        dist_matrix,
        dist_cutoff=5,
        facilities_to_site=2,
    ).optimize()

    assert mcmclp.model.status == OptimizationStatus.OPTIMAL

    assert len(mcmclp.result.solution[0]) == 26
    assert len(mcmclp.result.solution[1]) == 26

    opt_facilities = facilities[list(mcmclp.result.solution.keys())]
    assert len(opt_facilities) == 2
    assert [5, 0] in opt_facilities
    assert [-5, 10] in opt_facilities
    # utils.plot_result(demand, mcmclp.result.solution, opt_facilities)


def test_maximize_coverage_minimize_cost_far(demand, facilities):

    dist_matrix = distance_matrix(demand, facilities)

    mcmclp = MaximizeCoverageMinimizeCost(
        demand,
        facilities,
        dist_matrix,
        dist_cutoff=7.5,
        facilities_to_site=2,
    ).optimize()

    assert mcmclp.model.status == OptimizationStatus.OPTIMAL

    solution_keys = list(mcmclp.result.solution.keys())

    assert len(mcmclp.result.solution[solution_keys[0]]) > 20
    assert len(mcmclp.result.solution[solution_keys[0]]) < 30
    assert len(mcmclp.result.solution[solution_keys[1]]) > 25
    assert len(mcmclp.result.solution[solution_keys[1]]) < 40

    opt_facilities = facilities[solution_keys]
    assert len(opt_facilities) == 2
    assert [5, 0] in opt_facilities
    assert [-5, 10] in opt_facilities
    # utils.plot_result(demand, mcmclp.result.solution, opt_facilities)
