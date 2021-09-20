# -*- coding: utf-8 -*-
"""
Maximum Coverage Minimizing Cost Location Problem
"""

import logging
import random
import time
from typing import List

import mip as mip
import numpy as np

from .common import Config, Result

logger = logging.getLogger(__name__)


class MaximizeCoverageMinimizeCost:
    def __init__(
        self,
        points: np.ndarray,
        facilities: np.ndarray,
        dist_matrix: np.ndarray,
        dist_cutoff: int,
        facilities_to_site: int,
        load_distribution_weight: int = 10,
        maximum_coverage_weight: int = 100,
        total_distance_weight: int = 1,
        max_gap: float = 0.1,
    ):
        """
        **Maximum Coverage Minimum Cost Coverage Location Problem**

        The result of this computation is a given subset of candidate facilities such
        that as many demand points as possible are allocated to these within the distance cutoff value,
         while minimizing the distance of these demand points to the corresponding facility.

        **Problem Objective**

        Let K be the number of facilities to select for location coverage. The problem is to allocate
        locations to a selection of K facilities such that:

            1. the location coverage is maximized;
            2. the total distance from location to facility is minimized; and
            3. the difference between f_max and f_min is minimized,
               where f_max and f_min are the maximum and the minimum number of
               locations assigned to a facility in the given solution.

        The objective is thus to minimize a weighted sum of the three objective terms::

            minimize(total_distance * W1 - total_coverage * W2 + coverage_difference * W3)

        where :code:`W1`, :code:`W2` and :code:`W3` are the corresponding penalty weights.

        **Notes**

        * Allocation of a demand point to a facility is disregarded if the maximum distance between the facility and the demand point is exceeded.
        I.e., if the demand point is not within the distance cutoff of the facility.

        * Demand points within the cost cutoff of 2 or more facilities is allocated to the nearest facility.


        :param points: Numpy array of shape (n_points, 2).
        :param facilities: Numpy array of shape (n_facilities, 2).
        :param dist_matrix: Numpy array of shape (n_points, n_facilities).
            The distance matrix of points to facilities.
        :param dist_cutoff: Excludes from consideration (location, facility) pairs that are more
         than a maximum distance apart.
        :param facilities_to_site: The amount of facilites to choose,
            must be less than n_facilities.
        :param load_distribution_weight: penalty weight that is multiplied by (maxLoad - minLoad), where maxLoad and minLoad are
          the maximum and the minimum number of locations assigned to a facility.
          It thus penalises uneven distribution of loads assigned to facilities. This penalty defaults to 10
        :param maximum_coverage_weight: penalty weight that is multiplied by the number of sites that are not assigned to any of the facility.
                  This penalty defaults to 100
        :param total_distance_weight: penalty weight that is multiplied by the total distance. This weight defaults to 1.
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
            load_distribution_weight=load_distribution_weight,
            maximum_coverage_weight=maximum_coverage_weight,
            total_distance_weight=total_distance_weight,
            max_gap=max_gap,
        )

        self.result = None

        I = self.config.points.shape[0]
        J = self.config.facilities.shape[0]
        Dist = self.config.dist_matrix
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

        maxLoad = self.model.add_var(var_type=mip.INTEGER, lb=0, ub=I, name="maxLoad")
        minLoad = self.model.add_var(var_type=mip.INTEGER, lb=-I, ub=0, name="minLoad")

        initialSolution = self.generate_initial_solution(
            self.config.dist_matrix, I, J, self.config.facilities_to_site
        )

        bigM = 1000000
        epsilon = 0.001

        # Add constraints

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

        # if at least one point is allocated to facility, the facility must be considered as allocated: for all j \in J sum(z_ij) >= 1 TRUEIFF xj = true
        for j in range(J):
            self.model.add_constr(-1 + mip.xsum(z[i, j] for i in range(I)) >= -bigM * (1 - x[j]))

        for j in range(J):
            self.model.add_constr(-1 + epsilon + mip.xsum(z[i, j] for i in range(I)) <= bigM * x[j])

        # maxLoad >= z_ij
        for j in range(J):
            self.model.add_constr(mip.xsum(z[i, j] for i in range(I)) <= maxLoad)

        # -bigM - z_ij + bigM * x_j <= minLoad
        for j in range(J):
            self.model.add_constr(-bigM + bigM * x[j] - mip.xsum(z[i, j] for i in range(I)) <= minLoad)

        load_distribution_weight = self.config.load_distribution_weight
        maximum_coverage_weight = self.config.maximum_coverage_weight
        total_distance_weight = self.config.total_distance_weight

        # objective function: minimize the distance + maximize coverage + minimize difference between max facility and min facility allocation
        # note that minLoad takes on a negative value, and hence (maxLoad + minLoad) in the milp model objective
        self.model.objective = mip.minimize(
            (maxLoad + minLoad) * load_distribution_weight
            + mip.xsum(-y[i] * maximum_coverage_weight for i in range(I))
            + mip.xsum(Dist[i, j] * z[i, j] * total_distance_weight for i in range(I) for j in range(J))
        )

        self.model.start = [(z[i, j], 1.0) for (i, j) in initialSolution]
        self.model.max_mip_gap = self.config.max_gap

    @staticmethod
    def generate_initial_solution(D: np.ndarray, I: int, J: int, K: int) -> List[List[int]]:
        """
        Generate initial solution to use as the starting point for the milp solver.

        :param D: Numpy array of shape (n_points, n_facilities).
        :param I: n_points
        :param J: n_facilities
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
                    if D[i, j] == 1:
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

    def optimize(self, max_seconds: int = 200) -> "MaximizeCoverageMinimizeCost":
        """
        Optimize Maximize Coverage Minimize Cost Problem

        :param max_seconds: The amount of time given to the solver, defaults to 200.
        :return: Returns an instance of self consisting of

            * the configuration <location_allocation.common.CONFIG>,

            * mip model <mip.model.Model> (https://docs.python-mip.com/en/latest/classes.html)

            * points to facility allocations <location_allocation._maximize_coverage_minimize_cost.RESULT>.
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
