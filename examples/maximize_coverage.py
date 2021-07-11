from sklearn.datasets import make_moons
from scipy.spatial import ConvexHull, distance_matrix
from shapely.geometry import Polygon, Point
import numpy as np

from utils import plot_result
from location_allocation import MAXIMIZE_COVERAGE


def generate_candidate_facilities(points, M=100):
    """
    Generate M candidate sites with the convex hull of a point set
    Input:
        points: a Numpy array with shape of (N,2)
        M: the number of candidate sites to generate
    Return:
        sites: a Numpy array with shape of (M,2)
    """
    hull = ConvexHull(points)
    polygon_points = points[hull.vertices]
    poly = Polygon(polygon_points)
    min_x, min_y, max_x, max_y = poly.bounds
    sites = []
    while len(sites) < M:
        random_point = Point(
            [np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y)]
        )
        if random_point.within(poly):
            sites.append(random_point)
    return np.array([(p.x, p.y) for p in sites])


points, _ = make_moons(3000, noise=0.15)
facilities = generate_candidate_facilities(points, 50)
# facilities_to_choose = 5
cost_matrix = distance_matrix(points, facilities)
# cost_cutoff = 0.2

mclp = MAXIMIZE_COVERAGE(
    points, facilities, cost_matrix, facilities_to_choose=5, cost_cutoff=0.2
)
mclp.optimize()
print(mclp.result.opt_facilities)

# removing sites from the distance matrix which are not part of the solution
cost_matrix = np.delete(
    cost_matrix,
    [x for x in range(50) if x not in mclp.result.opt_facilities_indexes],
    1,
)
print(cost_matrix)
point_allocations = {}
for s_idx, x in enumerate(cost_matrix.T):
    point_allocations[mclp.result.opt_facilities_indexes[s_idx]] = []
    for idx, y in enumerate(x):
        if y == 1.0:
            point_allocations[mclp.result.opt_facilities_indexes[s_idx]].append(idx)

plot_result(points, point_allocations, mclp.result.opt_facilities)
