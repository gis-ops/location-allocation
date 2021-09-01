"""Example for Maximize Capacitated Coverage"""
from scipy.spatial import distance_matrix
from sklearn.datasets import make_moons
import numpy as np

from location_allocation import MaximizeCoverageCapacitated, utils

points, _ = make_moons(10000, noise=0.15)
facilities = utils.generate_candidate_facilities(points, 100)
capacities = utils.generate_facility_capacities(facilities.shape[0])

cost_matrix = distance_matrix(points, facilities)

mcclp = MaximizeCoverageCapacitated(
    points,
    facilities,
    cost_matrix,
    cost_cutoff=0.2,
    capacities=capacities,
    facilities_to_choose=5,
)
mcclp.optimize()

opt_facilities_indices = list(mcclp.result.solution.keys())
opt_facilities = facilities[opt_facilities_indices]
other_facilities = np.delete(facilities, [opt_facilities_indices], axis=0)
utils.plot_result(points, mcclp.result.solution, opt_facilities, other_facilities)
