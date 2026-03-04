import torch


def q_sample(x0, t, schedule, noise=None):
    """
    Forward diffusion: add noise to x0 at timestep t.
    Returns x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise
    """
    if noise is None:
        noise = torch.randn_like(x0)

    alpha_bar = schedule.alpha_bars[t].view(-1, 1, 1, 1).to(x0.device)

    return (
        torch.sqrt(alpha_bar) * x0 +
        torch.sqrt(1 - alpha_bar) * noise
    )


@torch.no_grad()
def predict_eps_cfg(model, x_t, labels, t, w, null_label=10):
    """
    Classifier-free guidance prediction.

    Computes:
        eps_guided = eps_uncond + w * (eps_cond - eps_uncond)

    At w=0: pure unconditional generation (maximum diversity)
    At w=1: equal mix
    At w>1: strong guidance toward class label (high quality, less diversity)

    Args:
        model:      trained CFG UNet
        x_t:        (B, 1, H, W) noisy image at timestep t
        labels:     (B,) class labels
        t:          (B,) timestep indices
        w:          float, guidance strength
        null_label: int, index of the unconditional token (10 for MNIST)
    """
    uncond_labels = torch.full_like(labels, fill_value=null_label)

    eps_uncond = model(x_t, uncond_labels, t)
    eps_cond   = model(x_t, labels, t)

    return eps_uncond + w * (eps_cond - eps_uncond)


@torch.no_grad()
def ddpm_sample(model, schedule, shape, labels, w, device, null_label=10):
    """
    Full reverse diffusion sampling loop using CFG.
    Generates images from pure noise.

    Args:
        model:    trained CFG UNet
        schedule: DiffusionSchedule
        shape:    tuple, e.g. (16, 1, 28, 28) for 16 MNIST images
        labels:   (B,) class labels
        w:        guidance strength
        device:   torch device
    Returns:
        x0: (B, 1, H, W) generated images in [-1, 1]
    """
    x = torch.randn(shape, device=device)
    B = shape[0]

    for i in reversed(range(schedule.timesteps)):
        t_batch = torch.full((B,), i, device=device, dtype=torch.long)

        eps = predict_eps_cfg(model, x, labels, t_batch, w, null_label)

        alpha     = schedule.alphas[i].to(device)
        alpha_bar = schedule.alpha_bars[i].to(device)
        beta      = schedule.betas[i].to(device)

        # Standard DDPM reverse step (Ho et al. 2020)
        if i > 0:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)

        x = (
            1 / torch.sqrt(alpha)
            * (x - (beta / torch.sqrt(1 - alpha_bar)) * eps)
            + torch.sqrt(beta) * noise
        )

    return x