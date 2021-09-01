"""Example for Maximize Coverage"""
import numpy as np
from scipy.spatial import distance_matrix
from sklearn.datasets import make_moons

from location_allocation import MaximizeCoverage, utils

points, _ = make_moons(3000, noise=0.15)
facilities = utils.generate_candidate_facilities(points, 50)
cost_matrix = distance_matrix(points, facilities)

mclp = MaximizeCoverage(
    points,
    facilities,
    cost_matrix,
    cost_cutoff=0.2,
    facilities_to_choose=5,
)
mclp.optimize()
print(mclp.result.opt_facilities)

# removing sites from the distance matrix which are not part of the solution
cost_matrix = np.delete(
    cost_matrix,
    [x for x in range(50) if x not in mclp.result.opt_facilities_indexes],
    1,
)
point_allocations = {}
for s_idx, x in enumerate(cost_matrix.T):
    point_allocations[mclp.result.opt_facilities_indexes[s_idx]] = []
    for idx, y in enumerate(x):
        if y == 1.0:
            point_allocations[mclp.result.opt_facilities_indexes[s_idx]].append(idx)

utils.plot_result(points, point_allocations, mclp.result.opt_facilities)
