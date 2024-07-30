import torch
from src.model.unet import ConditionalUNet
from src.diffusion_utils import sample_images
from src.data.transform_utils import postprocess

NUM_CLASSES = 10
BATCH_SIZE = 128
EPOCHS = 5000
NUM_TIMESTEPS = 1000
device = "cuda" if torch.cuda.is_available() else "cpu"

model = ConditionalUNet(use_label=True, num_classes=NUM_CLASSES).to(device)

PATH = "/Users/hongjeon/projects/diffusion-pytorch/checkpoints/no_attn_700000_128_1000_checkpoint.pth"
model.load_state_dict(torch.load(PATH)['model'], strict=False)

INFERENCE_SIZE = 1
IMAGE_SIZE = 32
label = torch.randint(0, NUM_CLASSES, [INFERENCE_SIZE]).to(device)
print(label)

samples = sample_images(model.eval(), num_steps=NUM_TIMESTEPS, batch_size=INFERENCE_SIZE, img_size=IMAGE_SIZE, num_channels=3, label=label)

print(f"Trained for {EPOCHS} epochs | batch size {BATCH_SIZE}")
print("last step")
res = postprocess(samples[-1][0]).resize((200,200))
res.save("./samples/sample.png")