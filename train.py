import torch
import numpy as np
import wandb
from tqdm.auto import tqdm
from torch.optim import Adam
from src.data.cifar_10 import get_dataloader as cifar_dataloader
from src.model.unet import ConditionalUNet
from src.diffusion_utils import linear_beta_schedule, sample_x_t
from accelerate import Accelerator

if __name__=="__main__":
  device = "cuda" if torch.cuda.is_available() else "cpu"
  print(f"device: {device}")


  BATCH_SIZE = 128
  NUM_TIMESTEPS = 1000
  NUM_CLASSES = 10
  LEARNING_RATE = 2e-4
  NAME = "no_attn"
  SAVE_MODEL_PATH = "./checkpoints"
  SAVE_MODEL = True
  PRINT_FREQ = 1000
  SAVE_FREQ = 100000
  MAX_NUM_STEPS = 800000

  wandb.init(
    project="diffusion-pytorch",
    config={
      "learning_rate": LEARNING_RATE,
      "architecture": "Pixelnet + no attn",
      "dataset": "CIFAR-10",
      "max_num_steps": MAX_NUM_STEPS
    }
  )

  model = ConditionalUNet(use_label=True, num_classes=NUM_CLASSES).to(device)
  optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
  train_dataloader = cifar_dataloader(BATCH_SIZE, train=True)

  accelerator = Accelerator()

  train_dataloader, model, optimizer = accelerator.prepare(
    train_dataloader, model, optimizer
  )

  progress_bar = tqdm(range(MAX_NUM_STEPS))

  step = 0
  losses = []
  while step < MAX_NUM_STEPS:
    for example in train_dataloader:
      if step == MAX_NUM_STEPS:
        break

      input_batch = example[0].to(device)
      label = example[1].to(device)

      optimizer.zero_grad()

      noise = torch.randn_like(input_batch).to(device)

      betas = linear_beta_schedule(NUM_TIMESTEPS)
      t = torch.randint(0, NUM_TIMESTEPS, (BATCH_SIZE,), device=device).long().to(device)
      x_t = sample_x_t(input_batch, noise, t, betas)
      pred_noise = model(x_t, t, label)

      loss = torch.nn.functional.mse_loss(noise, pred_noise)
      losses.append(loss.item())

      accelerator.backward(loss)
      optimizer.step()

      if SAVE_MODEL and step % SAVE_FREQ == 0 or step == 0:
        checkpoint = {
          'step': step,
          'model': model.state_dict(),
          'optimizer': optimizer.state_dict()
        }
        torch.save(checkpoint, f'{SAVE_MODEL_PATH}/{NAME}_{step}_{BATCH_SIZE}_{NUM_TIMESTEPS}_checkpoint.pth')

      if step % PRINT_FREQ == 0:
        mean_loss = np.mean(losses)
        print(f"Step: {step} | Loss: {mean_loss}")
        wandb.log({"loss": mean_loss}, step=step)
        losses = []

      progress_bar.update(1)
      step += 1