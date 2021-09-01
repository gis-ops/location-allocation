"""Common functions used across all models"""
import logging
import random
from itertools import cycle

import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import ConvexHull
from shapely.geometry import Point, Polygon


def generate_facility_capacities(M):
    capacities = []
    while len(capacities) < M:
        random_capacity = random.randint(M * 0.5, M * 1.8)
        capacities.append(random_capacity)

    return np.array(capacities)


def generate_candidate_facilities(points, M: int = 100) -> np.ndarray:
    """
    Generate M candidate sites with the convex hull of a point set

    :param points:  a Numpy array with shape of (n_points,2)
    :param M: the number of candidate sites to generate
    :return: a Numpy array with shape of (M,2)
    """
    hull = ConvexHull(points)
    polygon_points = points[hull.vertices]
    poly = Polygon(polygon_points)
    min_x, min_y, max_x, max_y = poly.bounds
    sites = []
    while len(sites) < M:
        random_point = Point([np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y)])
        if random_point.within(poly):
            sites.append(random_point)

    return np.array([(p.x, p.y) for p in sites])


def plot_result(points: np.ndarray, point_allocations: dict, opt_sites: list, other_sites: list):
    """
    Plots the result with matplotlib

    :param points:  a numpy array with shape of (n_points,2)
    :param point_allocations: facility to point allocations
    :param opt_sites: coordinates of optimital sites as numpy array
        in shape of [n_opt_sites,2]
    :param other_sites: coordinates of other sites as numpy array
        in shape of [n_other_sites,2]
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
