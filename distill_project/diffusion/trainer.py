import os
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .sampling import q_sample


def train_teacher(
    model,
    dataloader,
    schedule,
    optimizer,
    device,
    p_drop=0.1,
    null_label=10,
):
    """
    One epoch of teacher training with classifier-free guidance.

    CFG training trick:
        With probability p_drop, we replace the class label with null_label.
        This forces the model to learn BOTH conditional and unconditional denoising
        from a single network — which is exactly what we need to compute guided
        inference later with: eps_guided = eps_uncond + w*(eps_cond - eps_uncond)

    Args:
        model:      SimpleUNet
        dataloader: MNIST dataloader
        schedule:   DiffusionSchedule
        optimizer:  Adam or similar
        device:     cuda or cpu
        p_drop:     probability of dropping label (default 0.1 = 10%)
        null_label: token id for unconditional (default 10 for MNIST)
    """
    model.train()
    total_loss = 0.0

    for images, labels in tqdm(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        # CFG label dropout: replace with null_label with probability p_drop
        drop_mask = torch.rand(labels.shape, device=device) < p_drop
        labels_cfg = labels.clone()
        labels_cfg[drop_mask] = null_label

        # Sample random timesteps for each image in the batch
        t = torch.randint(0, schedule.timesteps, (images.size(0),), device=device)

        # Forward diffusion: add noise
        noise = torch.randn_like(images)
        x_t = q_sample(images, t, schedule, noise)

        # Predict noise — now passing t to the model
        noise_pred = model(x_t, labels_cfg, t)

        # Standard diffusion loss: MSE between predicted and actual noise
        loss = F.mse_loss(noise_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def save_checkpoint(model, optimizer, epoch, loss, out_dir="checkpoints", name="teacher"):
    """
    Save model + optimizer state. Call after every epoch.
    You don't want to retrain from scratch if something breaks in student phase.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}_epoch_{epoch:03d}.pt")
    torch.save(
        {
            "epoch": epoch,
            "loss": loss,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
        },
        path,
    )
    print(f"  Checkpoint saved: {path}")
    return path


def load_checkpoint(model, optimizer, path, device):
    """
    Load a saved checkpoint back into model and optimizer.
    Use this to resume training or load the teacher for student training.
    """
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optim_state"])
    print(f"  Loaded checkpoint from epoch {ckpt['epoch']} (loss={ckpt['loss']:.4f})")
    return ckpt["epoch"]