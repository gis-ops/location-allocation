import numpy as np
from mip import OptimizationStatus
from scipy.spatial import distance_matrix

from location_allocation import MaximizeCoverageMinimizeFacilities


def test_maximize_coverage_minimize_facilities_near(demand, facilities):

    dist_matrix = distance_matrix(demand, facilities)

    mcmclp = MaximizeCoverageMinimizeFacilities(
        demand,
        facilities,
        dist_matrix,
        dist_cutoff=5,
        max_facilities=2,
    ).optimize()

    assert mcmclp.model.status == OptimizationStatus.OPTIMAL

    res = sum([len(sub) for sub in mcmclp.result.solution.values()])
    assert res == 51

    opt_facilities = facilities[list(mcmclp.result.solution.keys())]
    assert len(opt_facilities) == 2
    assert [5, 0] in opt_facilities
    assert [-5, 10] in opt_facilities


def test_maximize_coverage_minimize_facilities_far(demand, facilities):

    dist_matrix = distance_matrix(demand, facilities)

    mcmclp = MaximizeCoverageMinimizeFacilities(
        demand,
        facilities,
        dist_matrix,
        dist_cutoff=15,
        max_facilities=2,
    ).optimize()

    assert mcmclp.model.status == OptimizationStatus.OPTIMAL
    # opt_facilities = facilities[list(mcmclp.result.solution.keys())]
    res = sum([len(sub) for sub in mcmclp.result.solution.values()])
    assert res == 119


def test_maximize_coverage_minimize_facilities_force_minimization(demand, facilities):

    dist_matrix = distance_matrix(demand, facilities)

    mcmclp = MaximizeCoverageMinimizeFacilities(
        demand,
        facilities,
        dist_matrix,
        dist_cutoff=5,
        facility_minimisation_weight=10,
        coverage_maximisation_weight=1,
    ).optimize()

    assert mcmclp.model.status == OptimizationStatus.OPTIMAL
    assert len(list(mcmclp.result.solution.keys())) == 2
    res = sum([len(sub) for sub in mcmclp.result.solution.values()])
    assert res == 51
    # for plotting
    # opt_facilities_indices = list(mcmclp.result.solution.keys())
    # opt_facilities = facilities[opt_facilities_indices]
    # other_facilities = np.delete(facilities, [opt_facilities_indices], axis=0)
    # utils.plot_result(demand, mcmclp.result.solution, opt_facilities, other_facilities)


def test_maximize_coverage_minimize_facilities_force_coverage(demand, facilities):

    dist_matrix = distance_matrix(demand, facilities)

    mcmclp = MaximizeCoverageMinimizeFacilities(
        demand,
        facilities,
        dist_matrix,
        dist_cutoff=5,
        facility_minimisation_weight=10,
        coverage_maximisation_weight=100,
    ).optimize()

    assert mcmclp.model.status == OptimizationStatus.OPTIMAL
    res = sum([len(sub) for sub in mcmclp.result.solution.values()])
    assert res == 56
    assert len(list(mcmclp.result.solution.keys())) == 4
    # for plotting
    # opt_facilities_indices = list(mcmclp.result.solution.keys())
    # opt_facilities = facilities[opt_facilities_indices]
    # other_facilities = np.delete(facilities, [opt_facilities_indices], axis=0)
    # utils.plot_result(demand, mcmclp.result.solution, opt_facilities, other_facilities)
