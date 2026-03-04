import torch
from torch import optim

from data.mnist_loader import get_mnist_dataloader
from diffusion.model import SimpleUNet
from diffusion.noise_schedules import DiffusionSchedule
from diffusion.trainer import train_teacher


device = "cuda" if torch.cuda.is_available() else "cpu"

dataloader = get_mnist_dataloader()

model = SimpleUNet().to(device)

schedule = DiffusionSchedule()

optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(10):

    loss = train_teacher(
        model,
        dataloader,
        schedule,
        optimizer,
        device
    )

    print(f"Epoch {epoch}: {loss}")