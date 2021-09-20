import numpy as np
import pytest
from mip import OptimizationStatus
from scipy.spatial import distance_matrix

from location_allocation import MaximizeCoverage


def test_maximize_coverage_near(demand, facilities):

    dist_matrix = distance_matrix(demand, facilities)

    mclp = MaximizeCoverage(
        demand,
        facilities,
        dist_matrix,
        dist_cutoff=1,
        facilities_to_site=2,
        max_gap=0.1,
    ).optimize()
    print(mclp.config)
    assert len(mclp.result.solution["opt_facilities"]) == 2
    assert [5, 0] in mclp.result.solution["opt_facilities"]
    assert [-5, 10] in mclp.result.solution["opt_facilities"]

    assert mclp.model.status == OptimizationStatus.OPTIMAL


def test_maximize_coverage_far(demand, facilities):

    dist_matrix = distance_matrix(demand, facilities)

    mclp = MaximizeCoverage(
        demand,
        facilities,
        dist_matrix,
        dist_cutoff=5,
        facilities_to_site=2,
        max_gap=0.1,
    ).optimize()

    assert len(mclp.result.solution["opt_facilities"]) == 2
    assert [5, 0] in mclp.result.solution["opt_facilities"]
    assert [-5, 10] in mclp.result.solution["opt_facilities"]
    assert mclp.model.status == OptimizationStatus.OPTIMAL


def test_maximize_coverage_too_many_facilities_to_site(demand, facilities):

    dist_matrix = distance_matrix(demand, facilities)

    with pytest.raises(ValueError):
        MaximizeCoverage(
            demand,
            facilities,
            dist_matrix,
            dist_cutoff=5,
            facilities_to_site=30,
            max_gap=0.1,
        ).optimize()


def test_maximize_coverage_wrong_types(demand, facilities):

    with pytest.raises(ValueError):
        MaximizeCoverage(
            [],
            [],
            [],
            dist_cutoff=5,
            facilities_to_site=30,
            max_gap=0.1,
        ).optimize()
