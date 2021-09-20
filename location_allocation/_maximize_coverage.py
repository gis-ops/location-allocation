# -*- coding: utf-8 -*-
"""
Maximize Coverage Location Problem
"""

import time
from typing import List
import mip as mip
import numpy as np

from .common import Config, Result


class MaximizeCoverage:
    def __init__(
        self,
        points: np.ndarray,
        facilities: np.ndarray,
        dist_matrix: np.ndarray,
        dist_cutoff,
        facilities_to_site,
        max_gap: float = 0.1,
    ):
        """
        **Maximum Coverage Location Problem**

        The result of this computation is a subset of candidate facilities such
        that as many demand points as possible are allocated to these within the cost cutoff value.

        **Problem Objective**

        Let K be the number of facilities to select for location coverage.
        The problem aims to maximize the number of locations covered by at least one facility
        from the subset of K selected facilities.

        **Notes**

        * Allocation of a demand point to a facility is disregarded if the maximum distance between the facility and the demand point is exceeded.
        I.e., if the demand point is not within the distance distance cutoff of the facility.

        * Demand points within the distance cutoff of one candidate facility have all its weight allocated to it

        * Demand points within the distance cutoff of 2 or more facilities is allocated to the nearest facility

        References:
        Church, R. and C. ReVelle, "The maximal covering location problem".
        In: Papers of the Regional Science Association, pp. 101-118. 1974.

        :param points:  Numpy array of shape (n_points, 2).
        :param facilities: Numpy array of shape (n_facilities, 2).
        :param dist_matrix: Numpy array of shape (n_points, n_facilities).
            The distance matrix of points to facilities.
        :param dist_cutoff: Distance cutoff which excludes from consideration (location, facility) pairs that are more
         than a maximum distance apart.
        :param facilities_to_site: The amount of facilites to choose,
            must be less than n_facilities.
        :param max_gap: Value indicating the tolerance for the maximum percentage deviation
            from the optimal solution cost, defaults to 0.1
        """
        self.config = Config(
            self.__class__.__name__,
            points,
            facilities,
            dist_matrix,
            dist_cutoff,
            facilities_to_site=facilities_to_site,
            max_gap=max_gap,
        )

        I = self.config.points.shape[0]
        J = self.config.facilities.shape[0]

        mask1 = self.config.dist_matrix <= self.config.dist_cutoff
        self.config.dist_matrix[mask1] = 1
        self.config.dist_matrix[~mask1] = 0

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
        self.model.add_constr(mip.xsum(x[j] for j in range(J)) == self.config.facilities_to_site)

        for i in range(I):
            self.model.add_constr(
                mip.xsum(x[j] for j in np.where(self.config.dist_matrix[i] == 1)[0]) >= y[i]
            )

        self.model.objective = mip.maximize(mip.xsum(y[i] for i in range(I)))

    def optimize(self, max_seconds=200) -> "MaximizeCoverage":
        """
        Optimize the maximum coverage problem

        :param max_seconds: The amount of time given to the solver, defaults to 200.
        :type max_seconds: int, optional
        :return: Returns an instance of ``self`` consisting of

            * the configuration class:`location_allocation.common.Config`,

            * mip model <mip.model.Model> (https://docs.python-mip.com/en/latest/classes.html)

            * optimized facility locations. class:`location_allocation._maximize_coverage.Result`.
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

        # opt_facilities: the optimal facilities coordinates
        # opt_facilities_indexes: the optimial facilities indices
        solution = {
            "opt_facilities": self.config.facilities[solution],
            "opt_facilities_indexes": solution,
        }
        self.result = Result(float(time.time() - start), solution)

        return self
