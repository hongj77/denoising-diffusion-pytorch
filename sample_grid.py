import torch
import argparse
from src.model.unet import ConditionalUNet
from src.diffusion_utils import sample_images
from src.sample_utils import plot_image_grid, save_images_as_png
from tqdm import tqdm
from src.data.transform_utils import postprocess
import os


if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint_path")
  parser.add_argument("--output_dir", default=".", help="Folder where to save generated images.")
  parser.add_argument("--rows", help="Number of rows in image grid.")
  parser.add_argument("--cols", help="Number of columns in image grid.")
  args = parser.parse_args()

  # Training settings.
  NUM_CLASSES = 10
  NUM_TIMESTEPS = 1000
  USE_LABEL=False
  USE_ATTN=True
  DROPOUT=0.1

  device = "cuda" if torch.cuda.is_available() else "cpu"
  print(f"device: {device}")

  model = ConditionalUNet(num_classes=NUM_CLASSES, use_label=USE_LABEL, use_attn=USE_ATTN, p_dropout=DROPOUT).to(device)

  model.load_state_dict(torch.load(args.checkpoint_path)['model_state_dict'], strict=False)

  IMAGE_SIZE = 32
  rows = int(args.rows)
  cols = int(args.cols)
  TOTAL_NUMBER_OF_IMAGES = rows * cols

  label = torch.randint(0, NUM_CLASSES, [TOTAL_NUMBER_OF_IMAGES]).to(device)
  samples = sample_images(model.eval(), num_steps=NUM_TIMESTEPS, batch_size=TOTAL_NUMBER_OF_IMAGES, img_size=IMAGE_SIZE, num_channels=3, label=label, device=device)

  images = []
  for i in range(TOTAL_NUMBER_OF_IMAGES):
      images.append(postprocess(samples[-1][i]))

  filename = os.path.join(args.output_dir, args.checkpoint_path.split('/')[-1]) + ".png"
  plot_image_grid(images, rows, cols, save_path=filename)

