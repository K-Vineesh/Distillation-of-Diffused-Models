import torch


def linear_beta_schedule(timesteps):
    """
    Linear noise schedule from the original DDPM paper.
    beta goes from beta_start to beta_end over 'timesteps' steps.
    """
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


class DiffusionSchedule:
    """
    Precomputes all schedule quantities needed for training and sampling.

    Key quantities:
        betas:      noise added at each step
        alphas:     1 - beta
        alpha_bars: cumulative product of alphas (used in q_sample)
    """

    def __init__(self, timesteps=1000):
        self.timesteps = timesteps

        self.betas      = linear_beta_schedule(timesteps)
        self.alphas     = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def get_alpha_bar(self, t):
        return self.alpha_bars[t]