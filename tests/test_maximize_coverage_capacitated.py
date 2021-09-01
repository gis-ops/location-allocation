import numpy as np
from mip import OptimizationStatus
from scipy.spatial import distance_matrix

from location_allocation import MaximizeCoverageCapacitated, utils


def test_maximize_coverage_capacitated_full(demand, facilities, capacities):

    cost_matrix = distance_matrix(demand, facilities)

    cost_cutoff = 15

    mcclp = MaximizeCoverageCapacitated(
        demand,
        facilities,
        cost_matrix,
        cost_cutoff,
        capacities,
        facilities_to_choose=2,
    )
    mcclp.optimize()

    assert mcclp.model.status == OptimizationStatus.OPTIMAL

    assert mcclp.model.objective_value == 30 or mcclp.model.objective_value == 29

    # note: this could fail due to cbc instability
    for k, v in mcclp.result.solution.items():
        assert len(v) == 15 or len(v) == 14

    opt_facilities = facilities[list(mcclp.result.solution.keys())]
    assert np.alltrue(opt_facilities[0] == [-5, 10])
    assert np.alltrue(opt_facilities[1] == [5, 10])
    # utils.plot_result(demand, mcclp.result.solution, opt_facilities)


def test_maximize_coverage_capacitated_partial(demand, facilities, capacities):

    cost_matrix = distance_matrix(demand, facilities)

    mcclp = MaximizeCoverageCapacitated(
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
