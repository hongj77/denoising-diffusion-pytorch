# Denoising Diffusion Pytorch
Implementation of DDPM in Pytorch. Let's see if we get good samples..

## Training
```
accelerate launch train.py
```

## Sample Image Grid
Generate samples and render a grid of images into one image.
```
python sample_grid.py --checkpoint_path=<path> --rows=8 --cols=8
```
![](examples/20240819_8x8.png)

## Save Images to Folder
Save a large number of images to a destination folder for running eval metrics.
```
python sample_images.py --checkpoint_path=<path> --output_dir=<dir> --num_images=50000 --batch_size=2500
```