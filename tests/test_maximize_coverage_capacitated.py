import numpy as np
from mip import OptimizationStatus
from scipy.spatial import distance_matrix

from location_allocation import MaximizeCoverageCapacitated

# from ..examples import utils


def test_maximize_coverage_capacitated_full(demand, facilities, capacities):

    dist_matrix = distance_matrix(demand, facilities)

    dist_cutoff = 15

    mcclp = MaximizeCoverageCapacitated(
        demand,
        facilities,
        dist_matrix,
        dist_cutoff,
        capacities,
        facilities_to_site=2,
    )
    mcclp.optimize()

    assert mcclp.model.status == OptimizationStatus.OPTIMAL

    assert mcclp.model.objective_value == 30 or mcclp.model.objective_value == 29

    # note: this could fail due to cbc instability
    for k, v in mcclp.result.solution.items():
        assert len(v) == 15 or len(v) == 14

    opt_facilities = facilities[list(mcclp.result.solution.keys())]
    assert [5, 0] in opt_facilities
    assert [-5, 10] in opt_facilities
    # utils.plot_result(demand, mcclp.result.solution, opt_facilities)


def test_maximize_coverage_capacitated_partial(demand, facilities, capacities):

    dist_matrix = distance_matrix(demand, facilities)

    mcclp = MaximizeCoverageCapacitated(
        demand,
        facilities,
        dist_matrix,
        dist_cutoff=2.5,
        capacities=capacities,
        facilities_to_site=2,
    )
    mcclp.optimize()

    assert mcclp.model.status == OptimizationStatus.OPTIMAL
    for k, v in mcclp.result.solution.items():
        assert len(v) == 8

    opt_facilities = facilities[list(mcclp.result.solution.keys())]
    assert [5, 0] in opt_facilities
    assert [-5, 10] in opt_facilities

    # utils.plot_result(demand, mcclp.result.solution, opt_facilities)
