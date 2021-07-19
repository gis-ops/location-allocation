# -*- coding: utf-8 -*-
"""
Maximum coverage location problem (MCLP) solved with Mixed Integer Programming.
Inspired by https://github.com/cyang-kth/maximum-coverage-location

Summary: (Una Help)
The result of this computation is a subset of candidate facilities such 
that as many demand points as possible are allocated to these within the cost cutoff value.

Problem Objective: (Una Help)

Notes: (Una Help)
- Demand points exceeding the facilities cost cutoffs are not considered in the computation
- Demand points within the cost cutoff of one candidate facility has all its weight allocated to it
- Demand points within he cost cutoff of two or more facilities has all its demand weight 
allocated to the nearest facility only
"""

# Author: Can Yang <cyang@kth.se>
#         GIS-OPS UG <enquiry@gis-ops.com>
#
# License: GPL License

import numpy as np
from mip import *
import time

from .common import CONFIG


class RESULT:
    def __init__(self, time_elapsed, opt_facilities, opt_facilities_indexes):
        self.time_elapsed = time_elapsed
        self.opt_facilities = opt_facilities
        self.opt_facilities_indexes = opt_facilities_indexes


def maximize_coverage(
    points, facilities, cost_matrix, cost_cutoff, max_seconds=200, **kwargs
):
    """Solve Maximum coverage location problem with MIP.

    Parameters
    ----------

    points : ndarray
        Numpy array of shape (n_points, 2).
    facilities : ndarray
        Numpy array of shape (n_facilities, 2).
    cost_matrix : ndarray
        Numpy array of shape (n_points, n_facilities).
        The distance matrix of points to facilities.
    facilities_to_choose : int
        The amount of facilites to choose, must be lesser equal n_facilities.
    cost_cutoff : int
        Cost cutoff which can be used to exclude points from the distance matrix which
        feature a greater cost.
    max_seconds : int, default=200
        The maximum amount of seconds given to the solver.

    Returns
    -------
    model : <mip.model.Model>
        MIP Model class documented at https://docs.python-mip.com/en/latest/classes.html
    result : <location_allocation._maximize_coverage.RESULT>
        A result object containing the optimal facilities coordinates (opt_facilities)
        and elapsed time in seconds (time_elapsed).

    Notes
    -----
    For an example, see :ref:`examples/maximize_coverage.py
    <sphx_glr_auto_examples_maximize_coverage.py>`.

    References
    ----------
    Church, R. and C. ReVelle, "The maximal covering location problem".
    In: Papers of the Regional Science Association, pp. 101-118. 1974.
    """

    mclp = MAXIMIZE_COVERAGE(
        points=points,
        facilities=facilities,
        cost_matrix=cost_matrix,
        cost_cutoff=cost_cutoff,
        **kwargs
    )
    mclp.optimize(max_seconds=max_seconds)
    return mclp.model, mclp.result


class MAXIMIZE_COVERAGE:
    """
    points : ndarray
        Numpy array of shape (n_points, 2).
    facilities : ndarray
        Numpy array of shape (n_facilities, 2).
    cost_matrix : ndarray
        Numpy array of shape (n_points, n_facilities).
        The distance matrix of points to facilities.
    facilities_to_choose : int
        The amount of facilites to choose, must be lesser equal n_facilities.
    cost_cutoff : int
        Cost cutoff which can be used to exclude points from the distance matrix which
        feature a greater cost.
    max_seconds : int, default=200
        The maximum amount of seconds given to the solver.
    """

    def __init__(self, points, facilities, cost_matrix, cost_cutoff, **kwargs):

        self.config = CONFIG(
            self.__class__.__name__,
            points,
            facilities,
            cost_matrix,
            cost_cutoff,
            **kwargs
        )

        I = self.config.points.shape[0]
        J = self.config.facilities.shape[0]

        mask1 = self.config.cost_matrix <= self.config.cost_cutoff
        self.config.cost_matrix[mask1] = 1
        self.config.cost_matrix[~mask1] = 0

        # Build model
        self.model = mip.Model()
        # Add variables
        x = {}
        y = {}
        for i in range(I):
            y[i] = self.model.add_var(var_type=mip.BINARY, name="y%d" % i)
        for j in range(J):
            x[j] = self.model.add_var(var_type=mip.BINARY, name="x%d" % j)

        # Add constraints
        self.model.add_constr(
            mip.xsum(x[j] for j in range(J)) == self.config.facilities_to_choose
        )

        for i in range(I):
            self.model.add_constr(
                mip.xsum(x[j] for j in np.where(self.config.cost_matrix[i] == 1)[0])
                >= y[i]
            )

        self.model.objective = mip.maximize(mip.xsum(y[i] for i in range(I)))

    def optimize(self, max_seconds=200):
        """Optimize MCLP.

        Parameters
        ----------
        max_seconds : int
            The maximum amount of seconds given to the solver.

        Returns
        -------
        self : object
            Returns an instance of self consisting of the configuration,
            mip model and optimized facility locations.
        """
        start = time.time()
        self.model.optimize(max_seconds=max_seconds)
        solution = []
        if (
            self.model.status == mip.OptimizationStatus.FEASIBLE
            or self.model.status == mip.OptimizationStatus.OPTIMAL
        ):
            for v in self.model.vars:
                if v.x == 1 and v.name[0] == "x":
                    solution.append(int(v.name[1:]))
        self.result = RESULT(
            float(time.time() - start), self.config.facilities[solution], solution
        )
        return self
