# diffusion/sampling.py

import torch


def q_sample(x0, t, schedule, noise=None):

    if noise is None:
        noise = torch.randn_like(x0)

    alpha_bar = schedule.alpha_bars[t].view(-1,1,1,1)

    return (
        torch.sqrt(alpha_bar) * x0 +
        torch.sqrt(1 - alpha_bar) * noise
    )