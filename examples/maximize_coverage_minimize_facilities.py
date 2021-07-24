"""Example for Maximize Coverage Minimize Facilities"""
from scipy.spatial import distance_matrix
from sklearn.datasets import make_moons

from location_allocation import MAXIMIZE_COVERAGE_MINIMIZE_FACILITIES, utils

points, _ = make_moons(10000, noise=0.15)
facilities = utils.generate_candidate_facilities(points, 100)

cost_matrix = distance_matrix(points, facilities)

mcmflp = MAXIMIZE_COVERAGE_MINIMIZE_FACILITIES(
    points, facilities, cost_matrix, cost_cutoff=0.2, max_gap=0.1
)
mcmflp.optimize()

opt_facilities = facilities[list(mcmflp.result.solution.keys())]
utils.plot_result(points, mcmflp.result.solution, opt_facilities)
