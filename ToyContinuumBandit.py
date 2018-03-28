from ContinuumBanditProblem import ContinuumBanditProblem
from RandomSurface import RandomSurface


class ToyContinuumBandit(ContinuumBanditProblem):
    """
    A simple wrapper to use RandomSurface ad a ContinuumBanditProblem
    """
    def __init__(self, pos_ranges, values_range=(0.0, 1.0), cov_diag_range=(0.01, 0.02),
                 cov_off_diag_range=(-0.01, 0.01), smooth=True, rocks=30):
        self.random_surface = RandomSurface(pos_ranges, values_range, cov_diag_range,
                                            cov_off_diag_range, smooth)
        self.random_surface.create_random_surface(rocks)

    def get_domains(self):
        return self.random_surface.get_domains()

    def reward(self, params):
        return self.random_surface.get_surface_at(params)

