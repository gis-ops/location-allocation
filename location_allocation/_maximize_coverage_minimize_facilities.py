# -*- coding: utf-8 -*-
"""
Maximize Coverage Minimize Facilities Location Problem

The problem is to allocate as many locations to facilities, while minimizing the number
of facilities selected for allocation. The number of facilities to allocate is determined by
the solver, hence the cost does not play a role in this model. For this model 2 weights
decide at what point is it beneficial to add an extra facility. In other words, how many locations
should be covered in order for it to be economical to add a facility.

Problem Objective:
The objective can be modelled as: minimize(total_facilities_selected * W1 - total_coverage * W2).

Notes: (Una to verify)
- Demand points exceeding the facilities cost cutoffs are not considered.
- Demand points within the cost cutoff of one candidate facility has all its weight allocated to it.
- Demand points within the cost cutoff of 2 or more facilities is allocated to the nearest facility.
- The number of facilities will be reduced if the cost of having one extra facility outweighs the
benefit of having X locations covered. This decision is dependent on the penalty weights for number
of facilities and location coverage. As an example: if facility_minimisation_weight is set to 10 and
coverage_maximisation_weight is set to 1 a facility covering less than 10 demand points will be removed.
"""
import logging
import random
import time

import mip as mip
import numpy as np

from .common import CONFIG

logger = logging.getLogger("la")


class RESULT:
    def __init__(self, time_elapsed, solution):
        """
        Result class

        :param time_elapsed: the time the solver occupied to compute the result
        :type time_elapsed: int
        :param solution: the solution object
        :type solution: object
        """
        self.time_elapsed = time_elapsed
        self.solution = solution


def generate_initial_solution(D, I, J, K):
    """
    Generate initial solution to use as the starting point for the milp solver.

    :param D: Numpy array of shape (n_points, n_facilities).
    :type D: ndarray
    :param I: n_points
    :type I: int
    :param J: n_facilities
    :type J: int
    :param K: n_facilities or max_facilities if provided by the user
    :type K: int
    :return: a list of pairs (i, j) denoting that point i is covered by facility j
    :rtype: list
    """
    Is = list(range(0, I))  # list of points
    max_number_of_trials = (
        1000  # if a feasible solution is not found after this many trials,
    )
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


class MAXIMIZE_COVERAGE_MINIMIZE_FACILITIES:
    def __init__(
        self,
        points,
        facilities,
        cost_matrix,
        cost_cutoff,
        max_facilities=None,
        facility_minimisation_weight=10,
        coverage_maximisation_weight=1,
        max_gap=0.1,
    ):
        """
        Maximum Coverage Minimize Facilities Location Problem Class

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
        :param facility_minimisation_weight: This value controls the importance
            of minimizing facilities, defaults to 10
        :type facility_minimisation_weight: int, optional
        :param coverage_maximisation_weight: This value controls the importance
            of coverage of demand points, defaults to 1
        :type coverage_maximisation_weight: int, optional
        :param max_facilities: The amount of facilites to choose, must be less than n_facilities,
            defaults to None.
        :type max_facilities: integer, optional
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
            facility_minimisation_weight=facility_minimisation_weight,
            coverage_maximisation_weight=coverage_maximisation_weight,
            max_facilities=max_facilities,
            max_gap=max_gap,
        )

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
                z[i, j] = self.model.add_var(
                    var_type=mip.BINARY, name="z" + str(i) + "_" + str(j)
                )

        if self.config.max_facilities is not None:
            K = self.config.max_facilities
            self.model.add_constr(
                mip.xsum(x[j] for j in range(J)) <= self.config.max_facilities
            )
        else:
            K = J
        # K should not be too large otherwise it becomes fairly slow
        initialSolution = generate_initial_solution(
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
                mip.xsum(z[i, j] for j in np.where(self.config.cost_matrix[i] == 1)[0])
                >= y[i]
            )

        # if at least one point is allocated to facility, the facility must be considered as allocated: for all j \in J sum(z_ij) >= 1 TRUEIFF xj = true
        for j in range(J):
            self.model.add_constr(
                -1 + mip.xsum(z[i, j] for i in range(I)) >= -bigM * (1 - x[j])
            )

        for j in range(J):
            self.model.add_constr(
                -1 + epsilon + mip.xsum(z[i, j] for i in range(I)) <= bigM * x[j]
            )

        facility_minimisation_weight = self.config.facility_minimisation_weight
        coverage_maximisation_weight = self.config.coverage_maximisation_weight

        # objective: minimize number of used facilities while maximizing the coverage
        self.model.objective = mip.minimize(
            facility_minimisation_weight * mip.xsum(x[j] for j in range(J))
            - coverage_maximisation_weight * mip.xsum(y[i] for i in range(I))
        )

        self.model.start = [(z[i, j], 1.0) for (i, j) in initialSolution]
        self.model.max_mip_gap = self.config.max_gap

    def optimize(self, max_seconds=200):
        """
        Optimize Maximize Coverage Minimize Facilities Problem

        :param max_seconds: The amount of time given to the solver, defaults to 200.
        :type max_seconds: int, optional
        :return: Returns an instance of self consisting of
            the configuration <location_allocation.common.CONFIG>,
            mip model <mip.model.Model> (https://docs.python-mip.com/en/latest/classes.html)
            and points to facility allocations <location_allocation._maximize_coverage_minimize_facilities.RESULT>.
        :rtype: :class:`location_allocation._maximize_coverage_minimize_facilities.MAXIMIZE_COVERAGE_MINIMIZE_FACILITIES`
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

        self.result = RESULT(float(time.time() - start), solution)
        return self
