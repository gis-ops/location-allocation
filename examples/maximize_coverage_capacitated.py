"""Example for Maximize Capacitated Coverage"""
from scipy.spatial import distance_matrix
from sklearn.datasets import make_moons

from location_allocation import MAXIMIZE_COVERAGE_CAPACITATED, utils

points, _ = make_moons(3000, noise=0.15)
facilities = utils.generate_candidate_facilities(points, 50)
capacities = utils.generate_facility_capacities(facilities.shape[0])

cost_matrix = distance_matrix(points, facilities)

mcclp = MAXIMIZE_COVERAGE_CAPACITATED(
    points,
    facilities,
    cost_matrix,
    cost_cutoff=0.2,
    capacities=capacities,
    facilities_to_choose=5,
)
mcclp.optimize()

opt_facilities = facilities[list(mcclp.result.solution.keys())]
utils.plot_result(points, mcclp.result.solution, opt_facilities)
