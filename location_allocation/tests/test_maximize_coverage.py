"""
Tests for maximize coverage algorithm
"""

from scipy.spatial import distance_matrix
import numpy as np
from mip import OptimizationStatus

from location_allocation import MAXIMIZE_COVERAGE, maximize_coverage


def test_maximize_coverage_near(demand, facilities):

    cost_matrix = distance_matrix(demand, facilities)

    mclp = MAXIMIZE_COVERAGE(
        demand, facilities, cost_matrix, facilities_to_choose=2, cost_cutoff=1
    ).optimize()

    assert len(mclp.result.opt_facilities) == 2
    assert np.alltrue(mclp.result.opt_facilities[0] == [-1, 2])
    assert np.alltrue(mclp.result.opt_facilities[1] == [1, 2])
    assert mclp.model.status == OptimizationStatus.OPTIMAL


def test_maximize_coverage_far(demand, facilities):

    cost_matrix = distance_matrix(demand, facilities)

    mclp = MAXIMIZE_COVERAGE(
        demand, facilities, cost_matrix, facilities_to_choose=2, cost_cutoff=5
    ).optimize()

    assert len(mclp.result.opt_facilities) == 2
    assert np.alltrue(mclp.result.opt_facilities[0] == [-1, 2])
    assert np.alltrue(mclp.result.opt_facilities[1] == [1, 2])
    assert mclp.model.status == OptimizationStatus.OPTIMAL


def test_maximize_coverage_fnc_near(demand, facilities):

    cost_matrix = distance_matrix(demand, facilities)

    model, result = maximize_coverage(
        demand, facilities, cost_matrix, facilities_to_choose=2, cost_cutoff=1
    )

    assert len(result.opt_facilities) == 2
    assert np.alltrue(result.opt_facilities[0] == [-1, 2])
    assert np.alltrue(result.opt_facilities[1] == [1, 2])
    assert model.status == OptimizationStatus.OPTIMAL


def test_maximize_coverage_fnc_far(demand, facilities):

    cost_matrix = distance_matrix(demand, facilities)

    model, result = maximize_coverage(
        demand, facilities, cost_matrix, facilities_to_choose=2, cost_cutoff=5
    )

    assert len(result.opt_facilities) == 2
    assert np.alltrue(result.opt_facilities[0] == [-1, 2])
    assert np.alltrue(result.opt_facilities[1] == [1, 2])
    assert model.status == OptimizationStatus.OPTIMAL
