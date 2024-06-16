from PIL import Image
import torch
import os
from torch.utils.data import Dataset

class Landscape(Dataset):
    
    def __init__(self,
                 root_directory: str,
                 transform = None):
        super(Landscape,self).__init__()
        self.root_directory = root_directory
        self.transform = transform
        self.dataset = os.listdir(self.root_directory)
        if '.DS_Store' in self.dataset: 
            self.dataset.remove('.DS_Store') 
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, image_index: int) -> torch.Tensor:
        self.image_index = image_index
        image_path = os.path.join(self.root_directory,self.dataset[image_index])
        image = Image.open(image_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        return image