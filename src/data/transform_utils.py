import torchvision
import einops
import numpy as np

def preprocess(example, image_size=32):
  transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(image_size),
    torchvision.transforms.CenterCrop(image_size),
    # Transforms PIL image in range [0,255] to pytorch image in range [0,1].
    # Image dimensions go from HWC to CHW.
    # https://pytorch.org/vision/stable/generated/torchvision.transforms.ToTensor.html
    torchvision.transforms.ToTensor(),
    # Scale to [-1, 1]
    torchvision.transforms.Lambda(lambda x: x*2 - 1)
  ])
  return transform(example)

  
# We also need the reverse transformation so we can go from image to pixels.
def postprocess(example):
  transform = torchvision.transforms.Compose([
      torchvision.transforms.Lambda(lambda x: (x+1)/2),
      # CHW -> HWC
      torchvision.transforms.Lambda(lambda x: einops.rearrange(x, 'c h w -> h w c')),
      torchvision.transforms.Lambda(lambda x: x*255.0),
      torchvision.transforms.Lambda(lambda x: x.numpy().astype(np.uint8)),
      torchvision.transforms.ToPILImage(),
  ])
  return transform(example)