# diffusion/noise_schedules.py

import torch


def linear_beta_schedule(timesteps):

    beta_start = 0.0001
    beta_end = 0.02

    return torch.linspace(beta_start, beta_end, timesteps)


class DiffusionSchedule:

    def __init__(self, timesteps=1000):

        self.timesteps = timesteps

        self.betas = linear_beta_schedule(timesteps)

        self.alphas = 1. - self.betas

        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def get_alpha_bar(self, t):

        return self.alpha_bars[t]