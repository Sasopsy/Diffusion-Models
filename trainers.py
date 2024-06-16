import torch
import torch.nn as nn
from tqdm import tqdm
import os
from model import UNet,VAE
from diffusion import LDM

class DDPMTrainer:
    """
    A trainer class for DDPM.

    Attributes:
        model (torch.nn.Module): The neural network model used for predicting the noise in the diffusion process.
        scheduler (Diffusion): An instance of the Diffusion class which handles the specifics of the noise process.
        dataloader: Iterable, typically a PyTorch DataLoader, that provides batches of images for training.
        device (str): The device on which to perform computations ('cpu' or 'cuda').
        learning_rate (float): The learning rate used for the optimizer.
    
    Methods:
        save: Saves the model's parameters to a specified path.
        train_step: Performs a single training step over the entire dataset.
        train: Runs the training process for a specified number of epochs and saves the model periodically.
    """
    def __init__(self,
                 model,
                 scheduler,
                 dataloader,
                 device,
                 learning_rate,
                 ) -> None:
        """
        Initializes the DiffusionTrainer with a model, diffusion process, dataloader, device, and learning rate.
        """
        self.unet = model
        self.scheduler = scheduler
        self.dataloader = dataloader
        self.device = device
        self.optimizer = torch.optim.AdamW(self.unet.parameters(),learning_rate)
        self.criterion = nn.MSELoss()
    
    def save(self,path):
        """
        Saves the model's state dictionary to the specified path.

        Args:
            path (str): The file path where the model's state dictionary should be saved.
        """
        torch.save(
            self.unet.state_dict(),
            path)
        
    def train_step(self):
        """
        Performs one epoch of training: looping over the dataloader to compute the loss, and update model weights.

        Returns:
            float: The average loss for the epoch.
        """
        self.unet.train()
        total_loss = 0.0
        loop = tqdm(self.dataloader,desc="Training")
        for images in loop:
            images = images.to(self.device)
            t = self.scheduler.sample_timesteps(images.shape[0]).to(self.device)
            x_t, noise = self.scheduler.noise_images(images,t)
            predicted_noise = self.unet(x_t,t)
            loss = self.criterion(noise,predicted_noise)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            loop.set_postfix(loss=loss.item())
            total_loss+=loss.item()
        
        return total_loss/len(self.dataloader)
        
    def train(self,
              epochs,
              save_every,
              root_directory,
              starting_point = 0):
        """
        Executes the training process for a specified number of epochs. Saves the model at intervals.

        Args:
            epochs (int): Total number of epochs to train the model.
            save_every (int): Interval of epochs after which the model is saved.
            root_directory (str): Directory where the model checkpoints will be saved.
            starting_point (int, optional): The starting point to label the saved models. Default is 0.
        """
        for epoch in range(epochs):
            loss = self.train_step()
            print(f"Epoch: {epoch+1} | Loss: {loss}")
            if epoch%save_every == 0:
                self.save(os.path.join(root_directory,f"model_ite_{starting_point+epoch}"))
            
class VAETrainer:
    
    def __init__(self,
                 vae: VAE,
                 dataloader,
                 device,
                 learning_rate,
                 criterion: nn.Module = nn.MSELoss()) -> None:
        self.vae = vae
        self.device = device
        self.optimizer = torch.optim.AdamW(self.vae.parameters(),lr=learning_rate)
        self.dataloader = dataloader
        self.criterion = criterion
        self.vae.to(device)
        
    def save(self,path):
        torch.save(
            self.vae.state_dict(),
            path)
    
    def kld_loss(self,mu,logvar):
        loss = -0.5*torch.sum(1+ logvar - mu**2 - logvar.exp())
        return loss
    
    def train_step(self):
        self.vae.train()
        total_loss = 0.0
        loop = tqdm(self.dataloader,desc="Training")
        for images in loop:
            images = images.to(self.device)
            images_reconstructed, mu, logvar = self.vae(images)
            reconstruction_loss = self.criterion(images_reconstructed,images)
            kld_loss = self.kld_loss(mu,logvar)
            loss = reconstruction_loss+kld_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            loop.set_postfix(loss=loss.item())
            total_loss += loss
        
        return total_loss/len(self.dataloader)
            
    def train(self,
              epochs,
              save_every,
              root_directory,
              starting_point = 0):

        for epoch in range(epochs):
            loss = self.train_step()
            print(f"Epoch: {epoch+1} | Loss: {loss}")
            if epoch%save_every == 0:
                self.save(os.path.join(root_directory,f"model_ite_{starting_point+epoch}"))
                

class LDMTrainer:
    
    def __init__(self,
                 ldm: LDM,
                 dataloader,
                 device,
                 learning_rate):
        self.ldm = ldm
        self.dataloader = dataloader
        self.device = device
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.AdamW(self.ldm.unet.parameters(),lr=learning_rate)
        self.criterion = nn.MSELoss()
        
    def save(self,path):
        self.ldm.save(path)
    
    def train_step(self):
        self.ldm.unet.train()
        total_loss = 0.0
        loop = tqdm(self.dataloader,desc="Training")
        for images in loop:
            images = images.to(self.device)
            latent_vectors = self.ldm.encode(images)
            t = self.ldm.scheduler.sample_timesteps(latent_vectors.shape[0]).to(self.device)
            z_t,noise = self.ldm.scheduler.noise_images(latent_vectors,t)
            predicted_noise = self.ldm.unet(z_t,t)
            loss = self.criterion(noise,predicted_noise)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            loop.set_postfix(loss=loss.item())
            total_loss+=loss.item()
        
        return total_loss/len(self.dataloader)
    
    def train(self,
              epochs,
              save_every,
              root_directory,
              starting_point = 0):

        for epoch in range(epochs):
            loss = self.train_step()
            print(f"Epoch: {epoch+1} | Loss: {loss}")
            if epoch%save_every == 0:
                self.save(os.path.join(root_directory,f"model_ite_{starting_point+epoch}"))
            
            
        
            