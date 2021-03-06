class ContinuumBanditProblem:
    """
    An abstract class for continuum bandit problems
    """
    def get_domains(self):
        """
        Returns a tuple of ranges of parameters 1 to n,
        each range is a tuple containing two limits.
        """
        pass

    def is_valid_params(self, params):
        """
        Returns true if the input parameters are within the allowed ranges
        """
        ranges = self.get_domains()
        for i in range(len(params)):
            if not ranges[i][0] <= params[i] <= ranges[i][1]:
                return False
        return True

    def reward(self, params):
        """
        Returns the reward given values of the n parameters.
        The values should be inputted as a list of n elements.
        """
        pass

