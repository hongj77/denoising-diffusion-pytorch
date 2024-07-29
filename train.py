import torch
import numpy as np
from tqdm.auto import tqdm
from torch.optim import Adam
from src.data.cifar_10 import get_dataloader as cifar_dataloader
from src.model.unet import ConditionalUNet
from src.diffusion_utils import linear_beta_schedule, sample_x_t

if __name__=="__main__":
  device = "cuda" if torch.cuda.is_available() else "cpu"
  print(f"device: {device}")

  START_EPOCH = 0
  EPOCHS = 1000
  BATCH_SIZE = 128
  PRINT_EPOCHS = 10
  NUM_TIMESTEPS = 1000
  NUM_CLASSES = 10
  NAME = "test"
  SAVE_MODEL = True
  SAVE_MODEL_EPOCH_FREQUENCY = 10
  SAVE_MODEL_PATH = "./checkpoints"

  model = ConditionalUNet(use_label=True, num_classes=NUM_CLASSES).to(device)
  optimizer = Adam(model.parameters(), lr=1e-3)
  dataloader = cifar_dataloader(BATCH_SIZE, train=True)

  epochs = range(START_EPOCH, EPOCHS)
  for epoch in tqdm(epochs):
    losses = []
    for example in dataloader:
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

      loss.backward()
      optimizer.step()

    # Save on every frequency and last epoch.
    if epoch % SAVE_MODEL_EPOCH_FREQUENCY == 0 or epoch == len(epochs)-1:
      print(f"Epoch: {epoch} | Loss: {np.mean(losses)}")
      if SAVE_MODEL:
        checkpoint = {
          'epoch': epoch,
          'model': model.state_dict(),
          'optimizer': optimizer.state_dict()
        }
        torch.save(checkpoint, f'{SAVE_MODEL_PATH}/{NAME}_{epoch}_{BATCH_SIZE}_{NUM_TIMESTEPS}_checkpoint.pth')