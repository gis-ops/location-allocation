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
        demand,
        facilities,
        cost_matrix,
        cost_cutoff=1,
        facilities_to_choose=2,
        max_gap=0.9,
    ).optimize()

    assert len(mclp.result.opt_facilities) == 2
    assert np.alltrue(mclp.result.opt_facilities[0] == [-5, 10])
    assert np.alltrue(mclp.result.opt_facilities[1] == [5, 10])
    assert mclp.model.status == OptimizationStatus.OPTIMAL


def test_maximize_coverage_far(demand, facilities):

    cost_matrix = distance_matrix(demand, facilities)

    mclp = MAXIMIZE_COVERAGE(
        demand,
        facilities,
        cost_matrix,
        cost_cutoff=5,
        facilities_to_choose=2,
        max_gap=0.9,
    ).optimize()

    assert len(mclp.result.opt_facilities) == 2
    assert np.alltrue(mclp.result.opt_facilities[0] == [-5, 10])
    assert np.alltrue(mclp.result.opt_facilities[1] == [5, 10])
    assert mclp.model.status == OptimizationStatus.OPTIMAL


def test_maximize_coverage_fnc_near(demand, facilities):

    cost_matrix = distance_matrix(demand, facilities)

    model, result = maximize_coverage(
        demand,
        facilities,
        cost_matrix,
        cost_cutoff=1,
        facilities_to_choose=2,
        max_gap=0.9,
    )

    assert len(result.opt_facilities) == 2
    assert np.alltrue(result.opt_facilities[0] == [-5, 10])
    assert np.alltrue(result.opt_facilities[1] == [5, 10])
    assert model.status == OptimizationStatus.OPTIMAL


def test_maximize_coverage_fnc_far(demand, facilities):

    cost_matrix = distance_matrix(demand, facilities)

    model, result = maximize_coverage(
        demand,
        facilities,
        cost_matrix,
        cost_cutoff=5,
        facilities_to_choose=2,
        max_gap=0.9,
    )

    assert len(result.opt_facilities) == 2
    assert np.alltrue(result.opt_facilities[0] == [-5, 10])
    assert np.alltrue(result.opt_facilities[1] == [5, 10])
    assert model.status == OptimizationStatus.OPTIMAL
