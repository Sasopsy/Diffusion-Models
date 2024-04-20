# Diffusion Models
My repository to implement different diffusion models as I learn about them.

The code for the different schedulers can be found in [`diffusion.py`](diffusion.py) the different modules and models used are in [`model.py`](model.py). 

## Denoising Diffusion Probabalistic Models

 <p align="center">
  <img src="assets/DDPM/generation.gif" alt="Generation GIF">
  <br>
  Figure 1: DDPM Generation
</p>

The model has been trained on the landscape dataset and can be found in [here](https://www.kaggle.com/datasets/arnaud58/landscape-pictures). 

It was trained for around 350 epochs and its weights can be found in [DDPM_356.pth](weights/DDPM_356.pth). To use it, run the following script
```python
import torch
from model import UNet  # The given model file

weights = torch.load('path/to/weights')
unet = UNet()  # Model to be instantiated using default arguments.
unet.load_state_dict(weights)
```
To generate and save images, run the following script
```python
from diffusion import DDPMScheduler
from utils import save_images

num_images = 32  # Can be any number
scheduler = DDPMScheduler(1000,1e-4,0.02,(64,64))
images = scheduler.sample(unet,num_images,False)

save_images(images,'save/path.png')
```
To generate a denoising animation, run the following script
```python
from utils import generate_sample_video

generate_sample_video(unet,
                      scheduler,
                      'save/path.gif',
                      32,
                      25):
```