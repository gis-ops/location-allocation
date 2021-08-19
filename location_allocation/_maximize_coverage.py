# -*- coding: utf-8 -*-
"""
Maximize Coverage Location Problem
"""

import time

import mip as mip
import numpy as np

from .common import CONFIG, RESULT


class MAXIMIZE_COVERAGE:
    def __init__(
        self,
        points,
        facilities,
        cost_matrix,
        cost_cutoff,
        facilities_to_choose,
        max_gap=0.1,
    ):
        """
        **Maximum Coverage Location Problem**

        The result of this computation is a subset of candidate facilities such
        that as many demand points as possible are allocated to these within the cost cutoff value.

        **Problem Objective**

        Let K be the number of facilities to select for location coverage.
        The problem aims to maximize the number of locations covered by at least one facility
        from the subset of K selected facilities.

        **Notes** (Una to verify)

        * Demand points exceeding the facilities cost cutoffs are not considered

        * Demand points within the cost cutoff of one candidate facility has all its weight allocated to it

        * Demand points within the cost cutoff of 2 or more facilities is allocated to the nearest facility

        References:
        Church, R. and C. ReVelle, "The maximal covering location problem".
        In: Papers of the Regional Science Association, pp. 101-118. 1974.

        :param points:  Numpy array of shape (n_points, 2).
        :type points: ndarray
        :param facilities: Numpy array of shape (n_facilities, 2).
        :type facilities: ndarray
        :param cost_matrix: Numpy array of shape (n_points, n_facilities).
            The distance matrix of points to facilities.
        :type cost_matrix: ndarray
        :param cost_cutoff: Cost cutoff which can be used to exclude points
            from the distance matrix which feature a greater cost.
        :type cost_cutoff: int
        :param facilities_to_choose: The amount of facilites to choose,
            must be less than n_facilities.
        :type facilities_to_choose: int
        :param max_gap: Value indicating the tolerance for the maximum percentage deviation
            from the optimal solution cost, defaults to 0.1
        :type max_gap: float, optional
        """
        self.config = CONFIG(
            self.__class__.__name__,
            points,
            facilities,
            cost_matrix,
            cost_cutoff,
            facilities_to_choose=facilities_to_choose,
            max_gap=max_gap,
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
        """
        Optimize the maximum coverage problem

        :param max_seconds: The amount of time given to the solver, defaults to 200.
        :type max_seconds: int, optional
        :return: Returns an instance of self consisting of

            * the configuration <location_allocation.common.CONFIG>,

            * mip model <mip.model.Model> (https://docs.python-mip.com/en/latest/classes.html)

            * optimized facility locations. <location_allocation._maximize_coverage.RESULT>.
        :rtype: :class:`location_allocation._maximize_coverage.MAXIMIZE_COVERAGE`
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

        # opt_facilities: the optimial facilities coordinates
        # opt_facilities_indexes: the optimial facilities indices
        solution = {
            "opt_facilities": self.config.facilities[solution],
            "opt_facilities_indexes": solution,
        }
        self.result = RESULT(float(time.time() - start), solution)
        return self
