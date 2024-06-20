import torch
import torch.nn as nn
import torchvision.models as models

def fetch_model(model: str):
    """Fetches model to be used for perceptual loss."""
    # Dictionary of models.
    models_ = {'vgg19':(models.VGG19_Weights,models.vgg19),
               'vgg16':(models.VGG16_Weights,models.vgg16),
               'efficientnet':(models.EfficientNet_B0_Weights,models.efficientnet_b0),
               'vgg11':(models.VGG11_Weights,models.vgg11),}
    # Fetch and load the model.
    model_ = models_[model][1](weights=models_[model][0].DEFAULT).features  # Will only return feature extractor layers in our models.
    return model_


class PerceptualLoss(nn.Module):
    """Object that calculates perceptual loss using a pre-trained classification model.
    
    Args:
        selected_layers (tuple[int]): list of the index of layers to be used.
        model (str): model to be used for perceptual loss.
        device (str): device where the compute will be performed.
    """
    def __init__(self, selected_layers: tuple[int],
                 model: str,
                 device: str,
                 reduction: str = 'mean') -> None:
        super(PerceptualLoss,self).__init__()
        self.model = fetch_model(model).eval().to(device)
        self.selected_layers = sorted(selected_layers)
        self.loss = nn.MSELoss(reduction=reduction)
        
        # Freeze the vgg parameters
        for params in self.model.parameters():
            params.requires_grad = False
        
        # List of slices through which our image will pass.
        self.slices = nn.ModuleList([nn.Sequential() for _ in range(len(selected_layers))])
        
        # Keep track of the index of first layer of every slice.
        start_layer = 0
        for i,layer in enumerate(selected_layers):
            for x in range(start_layer,layer+1): 
                self.slices[i].add_module(str(x),self.model[x])
            start_layer = layer+1
                
    def get_reconstruction_features(self,image: torch.Tensor):
        """Returns list of features from different layers of the model.
        Args:
            image (torch.Tensor): image we want to extract features of.
        """
        reconstructed_features = []
        x = image
        
        # Loop over the slices and get different representations from different layers.
        for layers in self.slices:
            x = layers(x)
            reconstructed_features.append(x)
            
        return reconstructed_features
   
    def compute_L_reconstructed(self,
                   reconstruction_features_og: list[torch.Tensor],
                   reconstruction_features_up: list[torch.Tensor]):
        """Computes reconstruction loss between the features.

        Args:
            reconstruction_features_og (list[torch.Tensor]): list of reconstruction features of original picture from different layers of model.
            reconstruction_features_up (list[torch.Tensor]): list of reconstruction features of upscaled picture from different layers of model.
        """
        # Reconstruction loss with the output of the last layer in the list.
        L = 0.0
        for i in range(len(reconstruction_features_og)):
            L += self.loss(reconstruction_features_og[i],reconstruction_features_up[i])
        return L
    
    def forward(self, original_image: torch.Tensor,
                upscaled_image: torch.Tensor):
        """Computes perception loss between original and upscaled image.

        Args:
            original_image (torch.Tensor): Tensor of original image.
            upscaled_image (torch.Tensor): Tensor of upscaled image.

        Returns:
            torch.Tensor: Final loss.
        """
        # Get the reconstructed features.
        reconstruction_features_up = self.get_reconstruction_features(upscaled_image)
        reconstruction_features_og = self.get_reconstruction_features(original_image)
        
        # Compute the reconstruction loss.
        loss = self.compute_L_reconstructed(reconstruction_features_og,
                                                       reconstruction_features_up)
        
        return loss
    
    
class PerceptualMse(PerceptualLoss):
    """Calculates mse loss with a weighted perceptual loss.
    """
    def __init__(self, selected_layers: tuple[int],
                 model: str,
                 device: str, 
                 perceptual_weight: float,
                 reduction: str = 'mean') -> None:
        super(PerceptualMse,self).__init__(selected_layers,model,device,reduction)
        # Define Mse loss
        self.mse_loss = nn.MSELoss(reduction=reduction)
        self.perceptual_weight = perceptual_weight
        
    def forward(self, original_image: torch.Tensor,
                upscaled_image: torch.Tensor):
        """Computes perception loss between original and upscaled image.

        Args:
            original_image (torch.Tensor): Tensor of original image.
            upscaled_image (torch.Tensor): Tensor of upscaled image.

        Returns:
            torch.Tensor: Final loss.
        """
        perceptual_loss = super().forward(original_image,upscaled_image)
        mse_loss = self.mse_loss(original_image,upscaled_image)
        
        total_loss = mse_loss + self.perceptual_weight*perceptual_loss
        
        return total_loss