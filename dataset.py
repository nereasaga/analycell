import torch
from torch.utils.data import Dataset
from skimage.io import imread
from skimage.transform import resize
import numpy as np
import os

class CellsDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        # obtener lista de archivos
        self.img_files = os.listdir(img_dir)
        self.mask_files = os.listdir(mask_dir)
        
        # ordenar las listas
        self.img_files.sort()
        self.mask_files.sort()
        
        # guardar directorios
        self.img_dir = img_dir
        self.mask_dir = mask_dir

    def __len__(self):
        # retornar número de archivos
        return len(self.img_files)

    def __getitem__(self, idx):
        # construir rutas completas
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
        
        # leer imágenes
        img = imread(img_path, as_gray=True)
        mask = imread(mask_path, as_gray=True)
        
        # redimensionar imagen
        img_resized = resize(img, (256, 256), preserve_range=True)
        # normalizar imagen
        img_normalized = img_resized / 255.0
        
        # redimensionar máscara
        mask_resized = resize(mask, (256, 256), preserve_range=True)
        # convertir máscara a binaria
        mask_binary = mask_resized > 0
        mask_binary = mask_binary.astype(np.float32)
        
        # convertir a tensores
        img_tensor = torch.tensor(img_normalized, dtype=torch.float32)
        img_tensor = img_tensor.unsqueeze(0)  # añadir dimensión de canal
        
        mask_tensor = torch.tensor(mask_binary, dtype=torch.float32)
        mask_tensor = mask_tensor.unsqueeze(0)
        
        return img_tensor, mask_tensor