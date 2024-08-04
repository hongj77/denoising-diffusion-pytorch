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
  BATCH_SIZE = 128
  NUM_TIMESTEPS = 1000
  NUM_CLASSES = 10
  LEARNING_RATE = 2e-4
  NAME = "no_attn"
  SAVE_MODEL_PATH = "./checkpoints"
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

  accelerator = Accelerator()
  device = accelerator.device
  print(f"device: {device}")

  model = ConditionalUNet(use_label=True, num_classes=NUM_CLASSES).to(device)
  optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
  train_dataloader = cifar_dataloader(BATCH_SIZE, train=True)


  train_dataloader, model, optimizer = accelerator.prepare(
    train_dataloader, model, optimizer
  )

  progress_bar = tqdm(range(MAX_NUM_STEPS))

  step = 0
  losses = []
  while step < MAX_NUM_STEPS:
    for example in train_dataloader:
      input_batch = example[0].to(device)
      label = example[1].to(device)

      optimizer.zero_grad()

      noise = torch.randn_like(input_batch)

      betas = linear_beta_schedule(NUM_TIMESTEPS)
      t = torch.randint(0, NUM_TIMESTEPS, (BATCH_SIZE,)).long().to(device)
      x_t = sample_x_t(input_batch, noise, t, betas)
      pred_noise = model(x_t, t, label)

      loss = torch.nn.functional.mse_loss(noise, pred_noise)
      losses.append(loss.item())

      accelerator.backward(loss)
      optimizer.step()

      mean_loss = np.mean(losses)
      if step % PRINT_FREQ == 0 and accelerator.is_local_main_process:
        print(f"Step: {step} | Loss: {mean_loss}")
        wandb.log({"loss": mean_loss}, step=step)
      losses = []

      progress_bar.update(1)
      step += 1

      if step % SAVE_FREQ == 0 or step == 1:
        output_dir = f'{SAVE_MODEL_PATH}/{NAME}_{step}_{BATCH_SIZE}_{NUM_TIMESTEPS}'
        accelerator.save_state(output_dir)

      if step >= MAX_NUM_STEPS:
        break