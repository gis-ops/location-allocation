from matplotlib import pyplot as plt
from matplotlib import colors
from itertools import cycle
import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, Point
import random
import logging


def generate_facility_capacities(M):

    capacities = []
    while len(capacities) < M:
        random_capacity = random.randint(M * 0.5, M * 1.8)
        capacities.append(random_capacity)
    return np.array(capacities)


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


def plot_result(points, point_allocations, opt_sites):
    """
    Plot the result
    Input:
        points: input points, Numpy array in shape of [N,2]
        opt_sites: locations K optimal sites, Numpy array in shape of [K,2]
    """
    fig = plt.figure(figsize=(8, 8))
    plt.scatter(points[:, 0], points[:, 1], c="black", s=4)
    ax = plt.gca()

    plt.scatter(opt_sites[:, 0], opt_sites[:, 1], c="C1", s=200, marker="*")

    mpl_colors = []
    for name, hex in colors.cnames.items():
        mpl_colors.append(name)
    cycol = cycle(mpl_colors)

    for k, v in point_allocations.items():
        color = next(cycol)
        for point_idx in v:
            plt.scatter(points[point_idx][0], points[point_idx][1], c=color, marker="+")

    # for site in opt_sites:
    #     circle = plt.Circle(site, radius, color='C1',fill=False,lw=2)
    #     ax.add_artist(circle)
    ax.axis("equal")
    ax.tick_params(
        axis="both",
        left=False,
        top=False,
        right=False,
        bottom=False,
        labelleft=False,
        labeltop=False,
        labelright=False,
        labelbottom=False,
    )
    plt.show()


def custom_logger(name, debug, report=False):
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(module)s - %(message)s"
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    if report:
        logger.addHandler(logging.FileHandler("log.txt", mode="w"))

    return logger
