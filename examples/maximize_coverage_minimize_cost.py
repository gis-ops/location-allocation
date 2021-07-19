from sklearn.datasets import make_moons
from scipy.spatial import distance_matrix

from location_allocation import MAXIMIZE_COVERAGE_MINIMIZE_COST, utils

points, _ = make_moons(3000, noise=0.15)
facilities = utils.generate_candidate_facilities(points, 50)

cost_matrix = distance_matrix(points, facilities)

mcmclp = MAXIMIZE_COVERAGE_MINIMIZE_COST(
    points,
    facilities,
    cost_matrix,
    cost_cutoff=0.2,
    facilities_to_choose=3,
)
mcmclp.optimize()

opt_facilities = facilities[list(mcmclp.result.solution.keys())]
utils.plot_result(points, mcmclp.result.solution, opt_facilities)