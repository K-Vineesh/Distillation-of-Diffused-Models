# diffusion/trainer.py

import torch
import torch.nn.functional as F
from tqdm import tqdm


def train_teacher(model, dataloader, schedule, optimizer, device):

    model.train()

    total_loss = 0

    for images, labels in tqdm(dataloader):

        images = images.to(device)
        labels = labels.to(device)

        t = torch.randint(
            0,
            schedule.timesteps,
            (images.size(0),),
            device=device
        )

        noise = torch.randn_like(images)

        from .sampling import q_sample

        x_t = q_sample(images, t, schedule, noise)

        noise_pred = model(x_t, labels)

        loss = F.mse_loss(noise_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)