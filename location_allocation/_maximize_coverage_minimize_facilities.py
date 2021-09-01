# -*- coding: utf-8 -*-
"""
Maximize Coverage Minimize Facilities Location Problem
"""
import logging
import random
import time
from typing import Dict, List, Optional

import mip as mip
import numpy as np

from .common import Config, Result

logger = logging.getLogger(__name__)


class MaximizeCoverageMinimizeFacilities:
    def __init__(
        self,
        points: np.ndarray,
        facilities: np.ndarray,
        cost_matrix: np.ndarray,
        cost_cutoff: int,
        max_facilities: Optional[int] = None,
        facility_minimisation_weight: int = 10,
        coverage_maximisation_weight: int = 1,
        max_gap: float = 0.1,
    ):
        """
        **Maximum Coverage Minimize Facilities Location Problem**

        The problem is to allocate as many locations to facilities, while minimizing the number
        of facilities selected for allocation. The number of facilities to allocate is determined by
        the solver, hence the cost does not play a role in this model. For this model 2 weights
        decide at what point is it beneficial to add an extra facility. In other words, how many locations
        should be covered in order for it to be economical to add a facility.

        **Problem Objective**

        The objective can be modelled as::

            minimize(total_facilities_selected * W1 - total_coverage * W2)

        **Notes** (Una to verify)

        * Demand points exceeding the facilities cost cutoffs are not considered.

        * Demand points within the cost cutoff of one candidate facility has all its weight allocated to it.

        * Demand points within the cost cutoff of 2 or more facilities is allocated to the nearest facility.

        * The number of facilities will be reduced if the cost of having one extra facility outweighs the
          benefit of having X locations covered. This decision is dependent on the penalty weights for number
          of facilities and location coverage. As an example: if :code:`facility_minimisation_weight` is set to 10 and
          :code:`coverage_maximisation_weight` is set to 1 a facility covering less than 10 demand points will be removed.

        :param points:  Numpy array of shape (n_points, 2).
        :param facilities: Numpy array of shape (n_facilities, 2).
        :param cost_matrix: Numpy array of shape (n_points, n_facilities).
            The distance matrix of points to facilities.
        :param cost_cutoff: Cost cutoff which can be used to exclude points
            from the distance matrix which feature a greater cost.
        :param facility_minimisation_weight: This value controls the importance
            of minimizing facilities, defaults to 10
        :param coverage_maximisation_weight: This value controls the importance
            of coverage of demand points, defaults to 1
        :param max_facilities: The amount of facilites to choose, must be less than n_facilities,
            defaults to None.
        :param max_gap: Value indicating the tolerance for the maximum percentage deviation
            from the optimal solution cost, defaults to 0.1
        """
        self.config = Config(
            self.__class__.__name__,
            points,
            facilities,
            cost_matrix,
            cost_cutoff,
            facility_minimisation_weight=facility_minimisation_weight,
            coverage_maximisation_weight=coverage_maximisation_weight,
            max_facilities=max_facilities,
            max_gap=max_gap,
        )

        self.result = {}

        I = self.config.points.shape[0]
        J = self.config.facilities.shape[0]

        mask1 = self.config.cost_matrix <= self.config.cost_cutoff
        self.config.cost_matrix[mask1] = 1
        self.config.cost_matrix[~mask1] = 0

        self.model = mip.Model()
        # Add variables
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

        if self.config.max_facilities is not None:
            K = self.config.max_facilities
            self.model.add_constr(mip.xsum(x[j] for j in range(J)) <= self.config.max_facilities)
        else:
            K = J
        # K should not be too large otherwise it becomes fairly slow
        initialSolution = self.generate_initial_solution(
            self.config.cost_matrix, I, J, 10 if K > 10 else K
        )

        bigM = 1000000
        epsilon = 0.001

        # Add constraints
        # a point cannot be allocated to facility if its not within the facility radius
        for i in range(I):
            for j in np.where(self.config.cost_matrix[i] == 0)[0]:
                self.model.add_constr(z[i, j] == 0)

        # if point is covered, it needs to be covered by at least one facility
        for i in range(I):
            self.model.add_constr(
                mip.xsum(z[i, j] for j in np.where(self.config.cost_matrix[i] == 1)[0]) >= y[i]
            )

        # if at least one point is allocated to facility, the facility must be considered as allocated: for all j \in J sum(z_ij) >= 1 TRUEIFF xj = true
        for j in range(J):
            self.model.add_constr(-1 + mip.xsum(z[i, j] for i in range(I)) >= -bigM * (1 - x[j]))

        for j in range(J):
            self.model.add_constr(-1 + epsilon + mip.xsum(z[i, j] for i in range(I)) <= bigM * x[j])

        facility_minimisation_weight = self.config.facility_minimisation_weight
        coverage_maximisation_weight = self.config.coverage_maximisation_weight

        # objective: minimize number of used facilities while maximizing the coverage
        self.model.objective = mip.minimize(
            facility_minimisation_weight * mip.xsum(x[j] for j in range(J))
            - coverage_maximisation_weight * mip.xsum(y[i] for i in range(I))
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
        :param K: n_facilities or max_facilities if provided by the user
        :return: a list of pairs (i, j) denoting that point i is covered by facility j
        :rtype: list
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
                logger.debug("Feasible solution found")
                return best_solution

    def optimize(self, max_seconds: int = 200) -> "MaximizeCoverageMinimizeFacilities":
        """
        Optimize Maximize Coverage Minimize Facilities Problem

        :param max_seconds: The amount of time given to the solver.
        :return: Returns an instance of self consisting of

            * configuration :class:`location_allocation.common.Config`,

            * mip model <mip.model.Model> (https://docs.python-mip.com/en/latest/classes.html)

            * points to facility allocations :class:`location_allocation._maximize_coverage_minimize_facilities.Result`.
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
