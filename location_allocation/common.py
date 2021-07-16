class CONFIG:
    def __init__(
        self, algorithm_name, points, facilities, cost_matrix, cost_cutoff, **kwargs
    ):

        # common settings for all algorithms
        self.points = points
        self.facilities = facilities
        self.cost_matrix = cost_matrix
        self.cost_cutoff = cost_cutoff

        # specific settings for algorithms and weights
        for k in kwargs.keys():
            self.__setattr__(k, kwargs[k])

        # optimality gap
        if not hasattr(self, "max_gap"):
            self.max_gap = 0.1

        # custom weight settings
        if algorithm_name == "MAXIMIZE_COVERAGE_MINIMIZE_COST":
            if not hasattr(self, "load_distribution_weight"):
                self.load_distribution_weight = 10
            if not hasattr(self, "maximum_coverage_weight"):
                self.maximum_coverage_weight = 100
            if not hasattr(self, "total_distance_weight"):
                self.total_distance_weight = 1
        elif algorithm_name == "MAXIMIZE_COVERAGE_MINIMIZE_FACILITIES":
            if not hasattr(self, "facility_minimisation_weight"):
                self.facility_minimisation_weight = 10
            if not hasattr(self, "coverage_maximisation_weight"):
                self.coverage_maximisation_weight = 1
