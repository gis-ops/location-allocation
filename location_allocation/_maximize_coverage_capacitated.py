# -*- coding: utf-8 -*-

"""
Maximum Capicitated Coverage Location Problem
"""

# Author: GIS-OPS UG <enquiry@gis-ops.com>
#
# License: GPL License

import logging
import random
import time
from typing import Dict, List
import mip as mip
import numpy as np

from .common import Config, Result

logger = logging.getLogger(__name__)


class MaximizeCoverageCapacitated:
    def __init__(
        self,
        points: np.ndarray,
        facilities: np.ndarray,
        dist_matrix: np.ndarray,
        dist_cutoff: int,
        capacities: np.ndarray,
        facilities_to_site: int,
        max_gap: float = 0.1,
    ):
        """
        **Maximum Capacitated Coverage Location Problem**

        The result of this computation is a subset of candidate facilities such
        that as many demand points as possible are allocated to these within the distance cutoff value
        and considering the capacity of the facility itself.

        **Problem Objective**

        Let K be the number of facilities to select for location coverage and let C_f
        be the maximum number of locations that can be allocated to a facility f.
        The problem aims to maximize the number of location covered by at least one facility
        from the subset of K selected facilities, such that the number of locations assigned
        to each facility f does not exceed the facility capacity threshold C_f.

        **Notes**

        * Allocation of a demand point to a facility is disregarded if the maximum distance between the facility and the demand point is exceeded.
        I.e., if the demand point is not within the distance cutoff of the facility.

        * Demand points within the distance cutoff of 2 or more facilities is allocated to the nearest facility.

        * If the total demand of a facility is greater than the capacity of the facility,
          only the demand points that maximize the total captured demand.

        :param points: Numpy array of shape (n_points, 2).
        :param facilities: Numpy array of shape (n_facilities, 2).
        :param dist_matrix: Numpy array of shape (n_points, n_facilities).
            The distance matrix of points to facilities.
        :param dist_cutoff: Distance cutoff which excludes from consideration (location, facility) pairs that are more
         than a maximum distance apart.
        :param capacities: Numpy array of shape (n_capacities, ).
            Must be the same length as facilities with capacities as integers.
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
            capacities=capacities,
            facilities_to_site=facilities_to_site,
            max_gap=max_gap,
        )

        self.result = {}

        I = self.config.points.shape[0]
        J = self.config.facilities.shape[0]
        mask1 = self.config.dist_matrix <= self.config.dist_cutoff
        self.config.dist_matrix[mask1] = 1
        self.config.dist_matrix[~mask1] = 0

        self.model = mip.Model()
        x = {}  # is facility at site used
        y = {}  # is point covered by any facility
        z = {}  # is point i covered by facility j

        for i in range(I):
            y[i] = self.model.add_var(var_type=mip.BINARY, name="y%d" % i)
        for j in range(J):
            x[j] = self.model.add_var(var_type=mip.BINARY, name="x%d" % j)

        for i in range(I):
            for j in range(J):
                z[i, j] = self.model.add_var(var_type=mip.BINARY, name="z" + str(i) + "_" + str(j))

        initial_solution = self.generate_initial_solution(
            self.config.dist_matrix,
            I,
            J,
            self.config.capacities,
            self.config.facilities_to_site,
        )

        bigM = 1000000
        epsilon = 0.001

        # exactly K allocated facilities
        self.model.add_constr(mip.xsum(x[j] for j in range(J)) == self.config.facilities_to_site)

        # a point cannot be allocated to facility if its not within the facility radius
        for i in range(I):
            for j in np.where(self.config.dist_matrix[i] == 0)[0]:
                self.model.add_constr(z[i, j] == 0)

        # if point is covered, it needs to be covered by at least one facility
        for i in range(I):
            self.model.add_constr(
                mip.xsum(z[i, j] for j in np.where(self.config.dist_matrix[i] == 1)[0]) >= y[i]
            )

        # the number of points allocated to facility must not exceed facility capacity
        for j in range(J):
            self.model.add_constr(mip.xsum(z[i, j] for i in range(I)) <= self.config.capacities[j])

        # if at least one point is allocated to facility, the facility must be considered as allocated: for all j \in J sum(z_ij) >= 1 TRUEIFF xj = true
        for j in range(J):
            self.model.add_constr(-1 + mip.xsum(z[i, j] for i in range(I)) >= -bigM * (1 - x[j]))

        for j in range(J):
            self.model.add_constr(-1 + epsilon + mip.xsum(z[i, j] for i in range(I)) <= bigM * x[j])
        self.model.objective = mip.maximize(mip.xsum(y[i] for i in range(I)))
        self.model.start = [(z[i, j], 1.0) for (i, j) in initial_solution]
        self.model.max_mip_gap = self.config.max_gap

    @staticmethod
    def generate_initial_solution(
        D: np.ndarray, I: int, J: int, C: np.ndarray, K: int
    ) -> List[List[int]]:
        """
        Generate initial solution to use as the starting point for the milp solver.

        :param D: Numpy array of shape (n_points, n_facilities).
        :param I: n_points
        :param J: n_facilities
        :param C: Capacity for each facility of shape (n_facilities_capacities, )
        :param K: facilities_to_site
        :return: a list of pairs (i, j) denoting that point i is covered by facility j
        """
        Is = list(range(0, I))  # list of points
        max_number_of_trials = 1000  # if a feasible solution is not found after this many trials,
        # the least infeasible solution is returned
        number_of_trials = 0
        best_solution = []  # least infeasible solution
        best_number_of_assigned_facilities = 0
        while True:
            number_of_trials += 1
            Js = random.sample(list(range(0, J)), K)  # random selection of K facilities
            solution = []

            random.shuffle(Is)
            random.shuffle(Js)
            assigned_facilities = 0
            for j in Js:
                points_assigned_to_facility = 0
                for i in Is:
                    if points_assigned_to_facility < C[j] and D[i, j] == 1:
                        points_assigned_to_facility += 1
                        solution.append([i, j])

                if points_assigned_to_facility != 0:
                    assigned_facilities += 1

            if assigned_facilities > best_number_of_assigned_facilities:
                best_number_of_assigned_facilities = assigned_facilities
                best_solution = solution

            if assigned_facilities == K or number_of_trials > max_number_of_trials:
                if number_of_trials > max_number_of_trials:
                    logger.debug("Feasible solution not found, return least infeasible solution")
                else:
                    logger.debug("Feasible solution found")
                return best_solution

    def optimize(self, max_seconds: int = 200) -> "MaximizeCoverageCapacitated":
        """
        Optimize Maximize Capacitated Coverage Problem

        :param max_seconds: The amount of time given to the solver, defaults to 200.
        :return: Returns an instance of self consisting of

            * the configuration :class:`location_allocation.common.Config`

            * mip model <mip.model.Model> (https://docs.python-mip.com/en/latest/classes.html)

            * points to facility allocations :class:`location_allocation._maximize_coverage_capacitated.Result`
        """
        start = time.time()
        self.model.optimize(max_seconds=max_seconds)

        solution = {}
        if (
            self.model.status == mip.OptimizationStatus.FEASIBLE
            or self.model.status == mip.OptimizationStatus.OPTIMAL
        ):
            for v in self.model.vars:
                if v.name[0] == "z" and v.x == 1:
                    site_point = v.name.split("_")
                    point_ix = int(site_point[0][1:])
                    site_ix = int(site_point[1])
                    if site_ix not in solution:
                        solution[site_ix] = []
                    solution[site_ix].append(point_ix)

        self.result = Result(float(time.time() - start), solution)

        return self
