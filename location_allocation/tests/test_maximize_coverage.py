"""
Tests for maximize coverage algorithm
"""

from scipy.spatial import distance_matrix
import numpy as np
from mip import OptimizationStatus

from location_allocation import MAXIMIZE_COVERAGE


def test_maximize_coverage_near(demand, facilities):

    cost_matrix = distance_matrix(demand, facilities)

    mclp = MAXIMIZE_COVERAGE(
        demand,
        facilities,
        cost_matrix,
        cost_cutoff=1,
        facilities_to_choose=2,
        max_gap=0.1,
    ).optimize()
    print(mclp.config)
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
        max_gap=0.1,
    ).optimize()

    assert len(mclp.result.opt_facilities) == 2
    assert np.alltrue(mclp.result.opt_facilities[0] == [-5, 10])
    assert np.alltrue(mclp.result.opt_facilities[1] == [5, 10])
    assert mclp.model.status == OptimizationStatus.OPTIMAL
