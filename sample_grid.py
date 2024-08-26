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
  parser.add_argument("--num_classes", help="Number of label classes. If provided, one row per label will be redenred instead of from the `rows` argument.")
  parser.add_argument("--labels_list", default="airplane,automobile,bird,cat,deer,dog,frog,horse,ship,truck", help="Label text comma separated values. Must have the same number of items as `num_classes`.", type=str)
  args = parser.parse_args()

  # Training settings.
  NUM_CLASSES = int(args.num_classes)
  NUM_TIMESTEPS = 1000
  USE_LABEL=True
  USE_ATTN=True
  DROPOUT=0.5

  device = "cuda" if torch.cuda.is_available() else "cpu"
  print(f"device: {device}")

  model = ConditionalUNet(num_classes=NUM_CLASSES, use_label=USE_LABEL, use_attn=USE_ATTN, p_dropout=DROPOUT).to(device)

  model.load_state_dict(torch.load(args.checkpoint_path)['model_state_dict'], strict=False)

  IMAGE_SIZE = 32
  cols = int(args.cols)

  if args.num_classes:
    rows = NUM_CLASSES
    labels = torch.arange(0, NUM_CLASSES).to(device)
    labels = labels.repeat_interleave(cols)
    label_for_plot = [str(item) for item in args.labels_list.split(',')]
    if len(label_for_plot) != rows:
      raise Exception(f"Number of labels {len(label_for_plot)} is not equal to the number of `num_classes` {args.num_classes} provided.")
  else:
    rows = int(args.rows)
    labels = torch.randint(0, NUM_CLASSES, [TOTAL_NUMBER_OF_IMAGES]).to(device)
    label_for_plot = None

  TOTAL_NUMBER_OF_IMAGES = rows * cols

  samples = sample_images(model.eval(), num_steps=NUM_TIMESTEPS, batch_size=TOTAL_NUMBER_OF_IMAGES, img_size=IMAGE_SIZE, num_channels=3, label=labels, device=device)

  images = []
  for i in range(TOTAL_NUMBER_OF_IMAGES):
      images.append(postprocess(samples[-1][i]))

  filename = os.path.join(args.output_dir, args.checkpoint_path.split('/')[-1]) + ".png"
  plot_image_grid(images, rows, cols, save_path=filename, labels=label_for_plot, img_size=IMAGE_SIZE)

