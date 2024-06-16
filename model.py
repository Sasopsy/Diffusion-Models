import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import *

class UNet(nn.Module):
    """
    A U-Net architecture with embedded self-attention modules and positional encoding. This network is designed
    for tasks like image segmentation where context and localization are important. The U-Net structure includes
    contracting (downsampling) and expanding (upsampling) paths to capture context and localize, enhanced with
    self-attention for better feature representation.

    Attributes:
        device (str): The device on which to perform computations.
        time_dim (int): Dimensionality of the time embedding.
        inc (nn.Module): Initial double convolution module.
        down1, down2, down3 (Down): Downsampling modules.
        sa1, sa2, sa3, sa4, sa5, sa6 (SelfAttention): Self-attention modules placed at various points in the network.
        bot1, bot2, bot3 (DoubleConv): Additional convolutional blocks in the bottom of the U-Net.
        up1, up2, up3 (Up): Upsampling modules.
        outc (nn.Conv2d): The final convolutional layer to produce output images.
    """
    def __init__(self, 
                 c_in=3, 
                 c_out=3, 
                 time_dim=256, 
                 device="cpu",
                 intermediate_channel = 32):
        """
        Args:
            c_in (int): Number of input channels.
            c_out (int): Number of output channels.
            time_dim (int): Dimensionality for the time embedding.
            device (str): The computing device ('cpu' or 'cuda').
            intermediate_channel (int): 
        """
        super().__init__()
        
        self.device = device
     
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in,intermediate_channel)
        self.down1 = Down(intermediate_channel,2*intermediate_channel,emb_dim=time_dim)
        self.sa1 = SelfAttention(2*intermediate_channel, 4)
        self.down2 = Down(2*intermediate_channel,4*intermediate_channel,emb_dim=time_dim)
        self.sa2 = SelfAttention(4*intermediate_channel,4)
        self.down3 = Down(4*intermediate_channel,4*intermediate_channel,emb_dim=time_dim)
        self.sa3 = SelfAttention(4*intermediate_channel,4)

        self.bot1 = DoubleConv(4*intermediate_channel,8*intermediate_channel)
        self.bot2 = DoubleConv(8*intermediate_channel,8*intermediate_channel)
        self.bot3 = DoubleConv(8*intermediate_channel,4*intermediate_channel)

        self.up1 = Up(8*intermediate_channel,2*intermediate_channel,emb_dim=time_dim)
        self.sa4 = SelfAttention(2*intermediate_channel,4)
        self.up2 = Up(4*intermediate_channel,intermediate_channel,emb_dim=time_dim)
        self.sa5 = SelfAttention(intermediate_channel,4)
        self.up3 = Up(2*intermediate_channel,intermediate_channel,emb_dim=time_dim)
        self.sa6 = SelfAttention(intermediate_channel,4)
        self.outc = nn.Conv2d(intermediate_channel,c_out,kernel_size=1)
        
        self.to(device)

    def pos_encoding(self, t, channels):
        """
        Generate positional encodings using a sinusoidal function, which helps the model to understand the 
        position or order of the inputs. This encoding adds information about the relative or absolute position 
        of the tokens in the sequence.

        Args:
            t (torch.Tensor): Time or positional tensor of shape (batch_size, 1), where each entry is a time 
                            or position value.
            channels (int): The number of channels for the encoding.

        Returns:
            torch.Tensor: The positional encodings of shape (batch_size, channels).
        """
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        """
        Forward pass of the U-Net model, processing the input through the architecture, incorporating downsampling,
        self-attention, and upsampling layers.

        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, c_in, height, width).
            t (torch.Tensor): Input time step. (batch_size)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, c_out, height, width).
        """
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output
    

class VAE(nn.Module):
    
    def __init__(self,
                 latent_dim,
                 downsample_factor,
                 num_heads,
                 device = "cpu"):
        super().__init__()
        self.encoder = Encoder(downsample_factor,latent_dim,num_heads=num_heads)
        self.decoder = Decoder(downsample_factor,latent_dim,num_heads=num_heads)
        self.to(device)
        
    def reparametrize(self,mu,logvar):
        std = torch.exp(0.5*logvar)
        epsilon = torch.randn_like(mu)
        return mu + epsilon*std
    
    def forward(self, x):
        mu,logvar = self.encoder(x)
        z = self.reparametrize(mu,logvar)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mu, logvar
