import torch
from torch import optim
import torchvision.utils as vutils
import os

from data.mnist_loader import get_mnist_dataloader
from diffusion.model import SimpleUNet
from diffusion.noise_schedules import DiffusionSchedule
from diffusion.trainer import train_teacher, save_checkpoint
from diffusion.sampling import ddpm_sample


# ── Config ────────────────────────────────────────────────────────────────────
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS      = 20
LR          = 1e-4
BATCH_SIZE  = 128
P_DROP      = 0.1   # CFG label dropout probability
NULL_LABEL  = 10    # unconditional token (MNIST has labels 0-9)
TIMESTEPS   = 1000
SAMPLE_EVERY = 5    # save a sample grid every N epochs
# ──────────────────────────────────────────────────────────────────────────────

print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

os.makedirs("samples", exist_ok=True)

# Data
dataloader = get_mnist_dataloader(batch_size=BATCH_SIZE, train=True)

# Model
model     = SimpleUNet(num_classes=10).to(DEVICE)
schedule  = DiffusionSchedule(timesteps=TIMESTEPS)
schedule.betas      = schedule.betas.to(DEVICE)
schedule.alphas     = schedule.alphas.to(DEVICE)
schedule.alpha_bars = schedule.alpha_bars.to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# ── Training loop ─────────────────────────────────────────────────────────────
for epoch in range(EPOCHS):

    loss = train_teacher(
        model,
        dataloader,
        schedule,
        optimizer,
        device=DEVICE,
        p_drop=P_DROP,
        null_label=NULL_LABEL,
    )

    print(f"Epoch {epoch:03d} | Loss: {loss:.4f}")

    # Save checkpoint every epoch
    save_checkpoint(model, optimizer, epoch, loss, out_dir="checkpoints", name="teacher")

    # Periodically generate sample grids at different guidance strengths
    # This lets you visually verify CFG is working before moving to student
    if (epoch + 1) % SAMPLE_EVERY == 0:
        model.eval()
        print(f"  Generating sample grids at epoch {epoch}...")

        # Generate digits 0-9 twice each = 20 samples
        sample_labels = torch.arange(10, device=DEVICE).repeat(2)

        for w in [0.0, 1.0, 3.0, 5.0]:
            samples = ddpm_sample(
                model,
                schedule,
                shape=(20, 1, 28, 28),
                labels=sample_labels,
                w=w,
                device=DEVICE,
                null_label=NULL_LABEL,
            )
            # Rescale from [-1, 1] to [0, 1] for saving
            samples = (samples + 1) / 2
            grid_path = f"samples/epoch_{epoch:03d}_w{w}.png"
            vutils.save_image(samples, grid_path, nrow=10)
            print(f"    Saved: {grid_path}")

        model.train()

print("Training complete.")
print("Checkpoints saved in: ./checkpoints/")
print("Sample grids saved in: ./samples/")