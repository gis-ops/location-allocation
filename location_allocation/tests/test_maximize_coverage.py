"""
Tests for maximize coverage algorithm
"""

import pytest

from scipy.spatial import distance_matrix
import numpy as np
from mip import OptimizationStatus

from location_allocation import MAXIMIZE_COVERAGE
from location_allocation import maximize_coverage


def test_maximize_coverage_near(demand, mclp_facilities):

    cost_matrix = distance_matrix(demand, mclp_facilities)

    mclp = MAXIMIZE_COVERAGE(
        demand, mclp_facilities, cost_matrix, facilities_to_choose=2, cost_cutoff=2
    ).optimize()

    assert len(mclp.result.opt_facilities) == 2
    assert np.alltrue(mclp.result.opt_facilities[0] == [2.5, 2.5])
    assert np.alltrue(mclp.result.opt_facilities[1] == [2.5, -2.5])
    assert mclp.model.status == OptimizationStatus.OPTIMAL


def test_maximize_coverage_far(demand, mclp_facilities):

    cost_matrix = distance_matrix(demand, mclp_facilities)

    mclp = MAXIMIZE_COVERAGE(
        demand, mclp_facilities, cost_matrix, facilities_to_choose=2, cost_cutoff=20
    ).optimize()

    assert len(mclp.result.opt_facilities) == 2
    assert np.alltrue(mclp.result.opt_facilities[0] == [-10, 10])
    assert np.alltrue(mclp.result.opt_facilities[1] == [10, 10])
    assert mclp.model.status == OptimizationStatus.OPTIMAL


def test_maximize_coverage_fnc_near(demand, mclp_facilities):

    cost_matrix = distance_matrix(demand, mclp_facilities)

    model, result = maximize_coverage(
        demand, mclp_facilities, cost_matrix, facilities_to_choose=2, cost_cutoff=2
    )

    assert len(result.opt_facilities) == 2
    assert np.alltrue(result.opt_facilities[0] == [2.5, 2.5])
    assert np.alltrue(result.opt_facilities[1] == [2.5, -2.5])
    assert model.status == OptimizationStatus.OPTIMAL


def test_maximize_coverage_fnc_far(demand, mclp_facilities):

    cost_matrix = distance_matrix(demand, mclp_facilities)

    model, result = maximize_coverage(
        demand, mclp_facilities, cost_matrix, facilities_to_choose=2, cost_cutoff=20
    )

    assert len(result.opt_facilities) == 2
    assert np.alltrue(result.opt_facilities[0] == [-10, 10])
    assert np.alltrue(result.opt_facilities[1] == [10, 10])
    assert model.status == OptimizationStatus.OPTIMAL
