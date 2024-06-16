from diffusion import DDPM_Scheduler
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
import imageio

def plot_images(images):
    """
    Displays a grid of images using matplotlib.

    Parameters:
    images (torch.Tensor): A tensor containing multiple images with shape (N, C, H, W), 
                           where N is the number of images, C is the number of channels, 
                           H is the height, and W is the width.
    """
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()

def save_images(images, path, **kwargs):
    """
    Saves a grid of images to a specified path using Pillow.

    Parameters:
    images (torch.Tensor): A tensor of images with shape (N, C, H, W), where N is the number of images.
    path (str): File path where the image will be saved.
    **kwargs: Additional keyword arguments passed to `torchvision.utils.make_grid` for image grid customization.
    """
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    print("here")
    im.save(path)
    
def generate_sample_video(model,
                     diffuser: DDPM_Scheduler,
                     path,
                     num_images = 32,
                     fps = 25,
                     **kwargs):
    """
    Generates a video from a sequence of image grids produced by a diffusion process.

    Parameters:
    model (torch.nn.Module): The neural network model used to predict noise in the diffusion process.
    diffuser (DDPM): The diffusion process object used to generate samples.
    path (str): The file path where the video will be saved.
    num_images (int): Number of images to generate for the video.
    fps (int): Frames per second for the output video.
    **kwargs: Additional keyword arguments passed to `torchvision.utils.make_grid` for image grid customization.
    """
    image,noise_to_images = diffuser.sample(model,num_images,True)
    _,noise_to_images = diffuser.convert(image,noise_to_images)
    grids = []
    for images in noise_to_images:
        images_grid = torchvision.utils.make_grid(images, **kwargs)
        ndarr = images_grid.permute(1, 2, 0).to('cpu').numpy()
        grids.append(Image.fromarray(ndarr))
    with imageio.get_writer(path, fps=fps) as writer:
        for i,grid in enumerate(grids):
            if i%4 == 0:  # Only taking every four frames.
                writer.append_data(grid)