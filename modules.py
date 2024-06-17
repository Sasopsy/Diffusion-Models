import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    """
    Implements a self-attention module with multi-head attention mechanism and a multilayer perceptron (MLP) block.

    Attributes:
        channels (int): Number of channels in the input.
        num_heads (int): Number of attention heads.
        mha (nn.MultiheadAttention): Multi-head attention mechanism.
        ln (nn.LayerNorm): Layer normalization before attention.
        mlp (nn.Sequential): Multilayer perceptron for processing outputs.

    Args:
        channels (int): Number of input and output channels.
        num_heads (int): Number of heads in multi-head attention.
    """
    def __init__(self,
                 channels: int,
                 num_heads: int) -> None:
        assert channels%num_heads == 0, "channels must be divisible by num_heads"
        super(SelfAttention,self).__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.mha = nn.MultiheadAttention(channels,num_heads)
        self.ln = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels,channels),
            nn.GELU(),
            nn.Linear(channels,channels),
        )
        
    def forward(self, x: torch.Tensor):
        """
        Forward pass for the self-attention module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, channels, height, width).
        """
        # Put resolution in 1'th dimension for generating attention maps.
        _,_,H,W = x.shape
        x = x.view(-1,self.channels,H*W).swapaxes(1,2)
        x = self.ln(x)
        attention_value, _ = self.mha(x,x,x)
        x = x + attention_value
        x = self.mlp(x) + x
        return x.swapaxes(2,1).view(-1,self.channels,H,W)
        

class DoubleConv(nn.Module):
    """
    A module implementing a double convolutional layer optionally with a residual connection.

    Attributes:
        residual (bool): If True, adds the input to the output.
        double_conv (nn.Sequential): Sequential container for the two convolutional layers.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        mid_channels (int, optional): Number of channels after the first convolution.
        residual (bool, optional): Whether to add a residual connection. Default is False.
    """
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 mid_channels: int = None, 
                 residual: bool = False):
        super(DoubleConv,self).__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        """
        Forward pass for the double convolution module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        """
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    """"
    A downsampling module that combines max pooling, double convolution, and an embedded layer.

    Attributes:
        maxpool_conv (nn.Sequential): Max pooling followed by double convolution layers.
        emb_layer (nn.Sequential): Embedded layer for processing additional input vector.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        emb_dim (int, optional): Dimensionality of the embedding input. Default is 256.
    """
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 emb_dim: int =256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        """
        Forward pass for the downsampling module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
            t (torch.Tensor): Embedding tensor of shape (batch_size, emb_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height/2, width/2).
        """
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb

    
class Up(nn.Module):
    """
    An upsampling module that combines bilinear upsampling, double convolution, and an embedded layer.

    Attributes:
        up (nn.Upsample): Bilinear upsampling layer.
        conv (nn.Sequential): Sequential container for convolution operations.
        emb_layer (nn.Sequential): Embedded layer for processing additional input vector.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        emb_dim (int, optional): Dimensionality of the embedding input. Default is 256. 
    """
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 emb_dim: int = 256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        """
        Forward pass for the upsampling module.

        Args:
            x (torch.Tensor): Input tensor from the lower layer of shape (batch_size, in_channels, height, width).
            skip_x (torch.Tensor): Input tensor from the skip connection of shape (batch_size, in_channels, 2*height, 2*width).
            t (torch.Tensor): Embedding tensor of shape (batch_size, emb_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, 2*height, 2*width).
        """
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb
    

def is_power_of_two(n):
    return (n & (n - 1) == 0) and n != 0

class UpVAE(nn.Module):
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_heads: int = 4):
        super().__init__()
        
        self.up = nn.Upsample(scale_factor=2,mode="bilinear",align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            SelfAttention(in_channels, num_heads),
            DoubleConv(in_channels, out_channels, in_channels // 2)
        )
        
    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x
    

class DownVAE(nn.Module):
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_heads: int = 4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            SelfAttention(in_channels, num_heads),
            DoubleConv(in_channels, out_channels)
        )
        
    def forward(self, x):
        return self.conv(x)


class Encoder(nn.Module):
    
    def __init__(self,
                 downsample_factor: int,
                 latent_dim: int,
                 in_channels: int = 3,
                 num_heads: int = 4):
        super().__init__()
        assert is_power_of_two(downsample_factor), "downsample_factor must be a power of 2"
        num_layers = int(torch.log2(torch.tensor(downsample_factor)))
        self.init_conv = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, padding=1)
        self.downsampling_layers = nn.Sequential(*[DownVAE(16 * (2 ** i), 16 * (2 ** (i + 1)), num_heads) for i in range(num_layers)])
        self.conv_mu = nn.Conv2d(in_channels=16 * (2 ** num_layers), out_channels=latent_dim, kernel_size=1)
        self.conv_logvar = nn.Conv2d(in_channels=16 * (2 ** num_layers), out_channels=latent_dim, kernel_size=1)
        
    def forward(self, x):
        x = self.init_conv(x)
        x = self.downsampling_layers(x)
        mu = self.conv_mu(x)
        logvar = self.conv_logvar(x)
        return mu, logvar


class Decoder(nn.Module):
    
    def __init__(self, 
                 upsample_factor: int,
                 latent_dim: int,
                 out_channels: int = 3,
                 num_heads: int = 4):
        super().__init__()
        assert is_power_of_two(upsample_factor), "upsample_factor must be a power of 2"
        num_layers = int(torch.log2(torch.tensor(upsample_factor)))
        self.init_conv = nn.Conv2d(in_channels=latent_dim, out_channels=16 * (2 ** num_layers), kernel_size=1)
        self.upsampling_layers = nn.Sequential(*[UpVAE(16 * (2 ** (i + 1)), 16 * (2 ** i), num_heads) for i in reversed(range(num_layers))])
        self.final_layer = nn.Conv2d(in_channels=16, out_channels=out_channels, kernel_size=3, padding=1)
        
    def forward(self, z):
        z = self.init_conv(z)
        z = self.upsampling_layers(z)
        x_reconstructed = self.final_layer(z)
        return x_reconstructed