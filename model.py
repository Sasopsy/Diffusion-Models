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
                 emb_dim: int =256):
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

    Args:
        c_in (int): Number of input channels.
        c_out (int): Number of output channels.
        time_dim (int): Dimensionality for the time embedding.
        device (str): The computing device ('cpu' or 'cuda').
    """
    def __init__(self, 
                 c_in=3, 
                 c_out=3, 
                 time_dim=256, 
                 device="cpu"):
        super().__init__()
        self.device = device
     
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in,32)
        self.down1 = Down(32,64)
        self.sa1 = SelfAttention(64, 4)
        self.down2 = Down(64,128)
        self.sa2 = SelfAttention(128,4)
        self.down3 = Down(128,128)
        self.sa3 = SelfAttention(128,4)

        self.bot1 = DoubleConv(128, 256)
        self.bot2 = DoubleConv(256,256)
        self.bot3 = DoubleConv(256,128)

        self.up1 = Up(256,64)
        self.sa4 = SelfAttention(64,4)
        self.up2 = Up(128,32)
        self.sa5 = SelfAttention(32,4)
        self.up3 = Up(64,32)
        self.sa6 = SelfAttention(32,4)
        self.outc = nn.Conv2d(32,c_out,kernel_size=1)
        
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