"""Common functions used across all models"""
import logging
import random
from itertools import cycle

import numpy as np
from matplotlib import colors
from matplotlib import pyplot as plt
from scipy.spatial import ConvexHull
from shapely.geometry import Point, Polygon


def generate_facility_capacities(M):
    capacities = []
    while len(capacities) < M:
        random_capacity = random.randint(M * 0.5, M * 1.8)
        capacities.append(random_capacity)
    return np.array(capacities)


def generate_candidate_facilities(points, M=100):
    """
    Generate M candidate sites with the convex hull of a point set

    :param points:  a Numpy array with shape of (n_points,2)
    :type points: ndarray
    :param M: the number of candidate sites to generate
    :type points: int
    :return: a Numpy array with shape of (M,2)
    :rtype: ndarray
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


def plot_result(points, point_allocations, opt_sites, other_sites):
    """
    Plots the result with matplotlib

    :param points:  a numpy array with shape of (n_points,2)
    :type points: ndarray
    :param point_allocations: facilitiy to point allocations
    :type point_allocations: dict
    :param opt_sites: coordinates of optimital sites as numpy array
        in shape of [n_opt_sites,2]
    :type opt_sites: list
    :param other_sites: coordinates of other sites as numpy array
        in shape of [n_other_sites,2]
    :type other_sites: list
    """
    plt.scatter(points[:, 0], points[:, 1], c="black", s=4)
    ax = plt.gca()

    plt.scatter(opt_sites[:, 0], opt_sites[:, 1], c="C1", s=200, marker="*")
    plt.scatter(other_sites[:, 0], other_sites[:, 1], c="black", s=200, marker="*")

    cycol = cycle(plt.get_cmap("tab20").colors)
    for k, v in point_allocations.items():
        color = next(cycol)
        for point_idx in v:
            plt.scatter(points[point_idx][0], points[point_idx][1], c=color, marker="+")

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
    """
    Custom Logger

    :param name: name of the logger
    :type name: str
    :param debug: if debug is true or false
    :type debug: bool
    :param report: if a logfile should be written, defaults to False
    :type report: bool, optional
    :return: returns the logging object
    :rtype: object
    """
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
