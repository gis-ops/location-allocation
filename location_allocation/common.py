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
