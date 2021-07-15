from sklearn.datasets import make_moons
from scipy.spatial import distance_matrix

from location_allocation import MAXIMIZE_CAPACITATED_COVERAGE, utils

points, _ = make_moons(3000, noise=0.15)
facilities = utils.generate_candidate_facilities(points, 50)
facilities_capacities = utils.generate_facility_capacities(facilities.shape[0])

cost_matrix = distance_matrix(points, facilities)

mcclp = MAXIMIZE_CAPACITATED_COVERAGE(
    points,
    facilities,
    facilities_capacities,
    cost_matrix,
    facilities_to_choose=5,
    cost_cutoff=0.2,
)
mcclp.optimize()

opt_facilities = facilities[list(mcclp.result.solution.keys())]
utils.plot_result(points, mcclp.result.solution, opt_facilities)
