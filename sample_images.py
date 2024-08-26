import torch
import argparse
from src.model.unet import ConditionalUNet
from src.diffusion_utils import sample_images
from src.sample_utils import save_images_as_png
from tqdm import tqdm
from src.data.transform_utils import postprocess


if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint_path")
  parser.add_argument("--output_dir", help="Folder where to save generated images.")
  parser.add_argument("--num_images", help="Number of images to generate.")
  parser.add_argument("--batch_size", help="Number of images to generate per inference batch.")
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
  TOTAL_NUMBER_OF_IMAGES = int(args.num_images)
  INFERENCE_SIZE = int(args.batch_size)

  for _ in tqdm(range(TOTAL_NUMBER_OF_IMAGES//INFERENCE_SIZE)):
    label = torch.randint(0, NUM_CLASSES, [INFERENCE_SIZE]).to(device)
    samples = sample_images(model.eval(), num_steps=NUM_TIMESTEPS, batch_size=INFERENCE_SIZE, img_size=IMAGE_SIZE, num_channels=3, label=label, device=device)

    images = []
    for i in range(INFERENCE_SIZE):
      images.append(postprocess(samples[-1][i]))

    save_images_as_png(images, destination_folder=args.output_dir)
