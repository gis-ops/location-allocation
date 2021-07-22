# -*- coding: utf-8 -*-
"""
Maximum Coverage Minimizing Cost Location Problem:
The result of this computation is a given subset of candidate facilities such 
that as many demand points as possible are allocated to these within the cost cutoff value
and thereby minimizing the distance of these demand points. 

Problem Objective:
Let K be the number of facilities to select for location coverage. The problem is to allocate 
locations to a selection of K facilities such that: 
    (i) the location coverage is maximized; 
    (ii) the total distance from location to facility is minimized; and 
    (iii) the difference between f_max and f_min is minimized, 
    where f_max and f_min are the maximum and the minimum number of 
    locations assigned to a facility in the given solution. 
The objective is thus to minimize a weighted sum of the three objective terms: 
i.e., minimize(total_distance * W1 - total_coverage * W2 + coverage_difference *W3), 
where W1, W2 and W3 are the corresponding penalty weights. 

Notes: (Una to verify)
- Demand points exceeding the facilities cost cutoffs are not considered.
- Demand points within the cost cutoff of one candidate facility has all its weight allocated to it.
- Demand points within the cost cutoff of 2 or more facilities is allocated to the nearest facility.
- Una help: Some examples like in minimize_facilities
"""

import numpy as np
from mip import *
import time
import random
import logging
from .common import CONFIG

logger = logging.getLogger("la")


class RESULT:
    def __init__(self, time_elapsed, solution):
        self.time_elapsed = time_elapsed
        self.solution = solution


def generate_initial_solution(D, I, J, K):
    """
    Generate initial solution to use as
    the starting point for the milp solver.

    :param D: Numpy array of shape (n_points, n_facilities).
    :type D: ndarray
    :param I: n_points
    :type I: int
    :param J: n_facilities
    :type J: int
    :param K: facilities_to_choose
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
            if number_of_trials > max_number_of_trials:
                logger.debug(
                    "Feasible solution not found, return least infeasible solution"
                )
            else:
                logger.debug("Feasible solution found")
            return best_solution


class MAXIMIZE_COVERAGE_MINIMIZE_COST:
    def __init__(
        self,
        points,
        facilities,
        cost_matrix,
        cost_cutoff,
        facilities_to_choose,
        load_distribution_weight=10,
        maximum_coverage_weight=100,
        total_distance_weight=1,
        max_gap=0.1,
    ):
        """
        Maximum Coverage Minimum Cost Coverage Location Problem Class

        :param points: Numpy array of shape (n_points, 2).
        :type points: ndarray
        :param facilities: Numpy array of shape (n_facilities, 2).
        :type facilities: ndarray
        :param cost_matrix: Numpy array of shape (n_points, n_facilities).
            The distance matrix of points to facilities.
        :type cost_matrix: ndarray
        :param cost_cutoff: Cost cutoff which can be used to
            exclude points from the distance matrix which
            feature a greater cost.
        :type cost_cutoff: int
        :param facilities_to_choose: The amount of facilites to choose,
            must be lesser equal n_facilities.
        :type facilities_to_choose: int
        :param load_distribution_weight: Una Help, defaults to 10
        :type load_distribution_weight: int, optional
        :param maximum_coverage_weight: Una Help, defaults to 100
        :type maximum_coverage_weight: int, optional
        :param total_distance_weight: Una Help, defaults to 1
        :type total_distance_weight: int, optional
        :param max_gap: Una Help, defaults to 0.1
        :type max_gap: float, optional
        """

        self.config = CONFIG(
            self.__class__.__name__,
            points,
            facilities,
            cost_matrix,
            cost_cutoff,
            facilities_to_choose=facilities_to_choose,
            load_distribution_weight=load_distribution_weight,
            maximum_coverage_weight=maximum_coverage_weight,
            total_distance_weight=total_distance_weight,
            max_gap=max_gap,
        )

        I = self.config.points.shape[0]
        J = self.config.facilities.shape[0]
        Dist = self.config.cost_matrix
        mask1 = self.config.cost_matrix <= self.config.cost_cutoff
        self.config.cost_matrix[mask1] = 1
        self.config.cost_matrix[~mask1] = 0

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
                z[i, j] = self.model.add_var(
                    var_type=mip.BINARY, name="z" + str(i) + "_" + str(j)
                )

        maxLoad = self.model.add_var(var_type=mip.INTEGER, lb=0, ub=I, name="maxLoad")
        minLoad = self.model.add_var(var_type=mip.INTEGER, lb=-I, ub=0, name="minLoad")

        initialSolution = generate_initial_solution(
            self.config.cost_matrix, I, J, self.config.facilities_to_choose
        )

        bigM = 1000000
        epsilon = 0.001

        # Add constraints

        # exactly K allocated facilities
        self.model.add_constr(
            mip.xsum(x[j] for j in range(J)) == self.config.facilities_to_choose
        )

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

        # maxLoad >= z_ij
        for j in range(J):
            self.model.add_constr(mip.xsum(z[i, j] for i in range(I)) <= maxLoad)

        # -bigM - z_ij + bigM * x_j <= minLoad
        for j in range(J):
            self.model.add_constr(
                -bigM + bigM * x[j] - mip.xsum(z[i, j] for i in range(I)) <= minLoad
            )

        load_distribution_weight = self.config.load_distribution_weight
        maximum_coverage_weight = self.config.maximum_coverage_weight
        total_distance_weight = self.config.total_distance_weight

        # objective function: minimize the distance + maximize coverage + minimize difference between max facility and min facility allocation
        # note that minLoad  takes on a negative value, and hence (maxLoad + minLoad) in the milp model objective
        self.model.objective = mip.minimize(
            (maxLoad + minLoad) * load_distribution_weight
            + mip.xsum(-y[i] * maximum_coverage_weight for i in range(I))
            + mip.xsum(
                Dist[i, j] * z[i, j] * total_distance_weight
                for i in range(I)
                for j in range(J)
            )
        )

        self.model.start = [(z[i, j], 1.0) for (i, j) in initialSolution]
        self.model.max_mip_gap = self.config.max_gap

    def optimize(self, max_seconds=200):
        """
        Optimize Maximize Coverage Minimize Cost Problem

        :param max_seconds: The amount of time given to the solver, defaults to 200.
        :type max_seconds: int, optional
        :return: Returns an instance of self consisting of
            the configuration <location_allocation.common.CONFIG>,
            mip model <mip.model.Model> (https://docs.python-mip.com/en/latest/classes.html)
            and points to facility allocations <location_allocation._maximize_coverage_minimize_cost.RESULT>.
        :rtype: :class:`location_allocation._maximize_coverage_minimize_cost.MAXIMIZE_COVERAGE_MINIMIZE_COST`
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
