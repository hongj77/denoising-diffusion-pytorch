import torch
import numpy as np
import wandb
from tqdm.auto import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR
from src.data.cifar_10 import get_dataloader as cifar_dataloader
from src.model.unet import ConditionalUNet
from src.diffusion_utils import linear_beta_schedule, sample_x_t, sample_images
from src.data.transform_utils import postprocess
from accelerate import Accelerator
from datetime import datetime



# TODO(hongj77): Write overfitting experiment trainer.
if __name__=="__main__":
  SAVE_MODEL_PATH = "./checkpoints"
  BATCH_SIZE = 128
  LEARNING_RATE = 2e-4
  NUM_TIMESTEPS = 1000
  NUM_CLASSES = 1
  MAX_NUM_STEPS = 800000
  SAVE_FREQ = 50000
  PRINT_FREQ = 1000
  EVAL_FREQ = 5000
  IMAGE_SIZE = 32
  WARMUP_STEPS = 1000
  LOG_WANDB = False
  CHECKPOINT_PATH = ""

  # Experiment variables.
  USE_LABEL = False
  USE_ATTN = False
  ACTIVATION = 'relu'
  USE_PRE_ACTIVATION = True
  USE_ZERO_INIT = True

  accelerator = Accelerator()
  device = accelerator.device
  print(f"device: {device}")

  name = f"classes_{NUM_CLASSES}_lr_{LEARNING_RATE:.4f}_timesteps_{NUM_TIMESTEPS}_warmup_{WARMUP_STEPS}_label_{USE_LABEL}_attn_{USE_ATTN}_act_{ACTIVATION}_preact_{USE_PRE_ACTIVATION}_zero"

  name += "_{}".format(datetime.now().strftime("%Y%m%d_%S"))
  print("Starting: ", name)

  if LOG_WANDB and accelerator.is_local_main_process:
    wandb.init(project="diffusion-pytorch", name=name)

  model = ConditionalUNet(num_classes=NUM_CLASSES, use_label=USE_LABEL, use_attn=USE_ATTN).to(device)
  optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
  scheduler = LinearLR(optimizer, start_factor=(1/WARMUP_STEPS), total_iters=WARMUP_STEPS)
  train_dataloader = cifar_dataloader(BATCH_SIZE, train=True)
  eval_dataloader = cifar_dataloader(BATCH_SIZE, train=False)

  if CHECKPOINT_PATH and accelerator.is_local_main_process:
    print(f"Loading checkpoint {CHECKPOINT_PATH}")
    loaded = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(loaded['model_state_dict'])
    optimizer.load_state_dict(loaded['optimizer_state_dict'])
    scheduler.load_state_dict(loaded['scheduler_state_dict'])
    step = loaded['step']
  else:
    step = 0


  train_dataloader, eval_dataloader, model, optimizer, scheduler = accelerator.prepare(
    train_dataloader, eval_dataloader, model, optimizer, scheduler
  )

  print(f"Starting training from {step}")
  progress_bar = tqdm(range(MAX_NUM_STEPS), initial=step)

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
        if LOG_WANDB:
          wandb.log({"loss": mean_loss}, step=step)
      losses = []

      progress_bar.update(1)
      step += 1

      if step % SAVE_FREQ == 0:
        output_dir = f'{SAVE_MODEL_PATH}/{NAME}_{step}_{BATCH_SIZE}_{NUM_TIMESTEPS}'
        checkpoint = dict(
            model_state_dict = accelerator.unwrap_model(model).state_dict(),
            optimizer_state_dict = accelerator.unwrap_model(optimizer).state_dict(),
            scheduler_state_dict = accelerator.unwrap_model(scheduler).state_dict() ,
            step = step
        )
        accelerator.save(checkpoint, output_dir)
      
      if step % EVAL_FREQ == 0:
        with torch.no_grad():
          eval_losses = []
          for eval_example in tqdm(eval_dataloader):
            input_batch = example[0].to(device)
            label = example[1].to(device)
            noise = torch.randn_like(input_batch)

            betas = linear_beta_schedule(NUM_TIMESTEPS)
            t = torch.randint(0, NUM_TIMESTEPS, (BATCH_SIZE,)).long().to(device)
            x_t = sample_x_t(input_batch, noise, t, betas)
            pred_noise = model(x_t, t, label)

            loss = torch.nn.functional.mse_loss(noise, pred_noise)
            eval_losses.append(loss.item())

          if accelerator.is_local_main_process:
            eval_loss = np.mean(eval_losses)
            print(f"eval_loss: {eval_loss}")
            if LOG_WANDB:
              wandb.log({"eval_loss": eval_loss}, step=step)
              # Visualize one example.
              labels = torch.arange(0, NUM_CLASSES).to(device)
              samples = sample_images(model.eval(), num_steps=NUM_TIMESTEPS, batch_size=NUM_CLASSES, img_size=IMAGE_SIZE, num_channels=3, label=labels, device=device)
              # Log the last step of the denoising process.
              last_step_sample = samples[-1]
              samples = [postprocess(sample) for sample in last_step_sample]
              samples = [wandb.Image(sample, caption=f"Class: {label}") for sample, label in zip(samples, labels)] 
              wandb.log({"samples": samples})
        model.train()

      if step >= MAX_NUM_STEPS:
        break
    scheduler.step()