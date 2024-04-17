from tqdm import tqdm
import torch

class DDPM:
    """
    Diffusion class for DDPMs.

    Attributes:
        noise_steps (int): Number of noise levels in the diffusion process.
        beta_start (float): Initial value of the variance of the noise schedule.
        beta_end (float): Final value of the variance of the noise schedule.
        img_size (tuple of int): Dimensions of the images to be generated (height, width).
        device (str): The device on which to perform computations.

    Methods:
        prepare_noise_schedule: Creates a linear schedule of noise levels.
        noise_images: Applies noise to images at a specified timestep.
        sample_timesteps: Randomly selects timesteps for the diffusion process.
        sample: Generates images using the diffusion model.
    """
    def __init__(self, 
                 noise_steps=1000, 
                 beta_start=1e-4, 
                 beta_end=0.02, 
                 img_size=(64,64), 
                 device="cpu"):
        """
        Initializes the Diffusion model with specified parameters and precomputes factors for the diffusion process.
        """
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        """
        Generates a tensor representing a linear noise schedule from beta_start to beta_end over the specified noise steps.
        
        Returns:
            torch.Tensor: A tensor of shape (noise_steps,) containing beta values.
        """
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        """
        Adds noise to images for a given timestep.

        Args:
            x (torch.Tensor): A tensor of images of shape (batch_size, channels, height, width).
            t (torch.Tensor): A tensor of timesteps of shape (batch_size,).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - Noised images of shape (batch_size, channels, height, width).
                - Noise of shape (batch_size, channels, height, width) that was added to the images.
        """
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        """
        Randomly selects timesteps for the diffusion process.

        Args:
            n (int): Number of timesteps to sample.

        Returns:
            torch.Tensor: A tensor of sampled timesteps of shape (n,).
        """
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, 
               model, 
               n,
               record_noise = False):
        """
        Generates images using the diffusion process.

        Args:
            model (torch.nn.Module): The denoising model that predicts noise given noisy images and timesteps.
            n (int): Number of images to generate.

        Returns:
            torch.Tensor: A tensor of generated images of shape (n, 3, height, width), where height and width
                          are specified by img_size. The pixel values are in the range [0, 255] and the data type is uint8.
            list[torch.Tensor]: A list of images being converted from noise to images.
        """
        model.eval()
        if record_noise:
            noise_to_image = []
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size[0], self.img_size[1])).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1/torch.sqrt(alpha) * (x-((1-alpha)/(torch.sqrt(1-alpha_hat)))*predicted_noise) + torch.sqrt(beta)*noise
                if record_noise:
                    x_image = (x.clamp(-1, 1) + 1) / 2
                    x_image = (x_image * 255).type(torch.uint8)
                    noise_to_image.append(x_image)
                    
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        if record_noise:
            noise_to_image.append(x)
            return x,noise_to_image
        else:
            return x