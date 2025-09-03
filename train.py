import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from glob import glob
from model_junior import UNet
from torchvision.transforms import v2
import numpy as np
from scipy.ndimage import distance_transform_edt

# clase para cargar los datos
class CellDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # cargar imagen y máscara
        img = Image.open(self.image_paths[idx]).convert("L")
        mask = Image.open(self.mask_paths[idx]).convert("L")
        
        # crear diccionario para las transformaciones
        sample = {'image': img, 'mask': mask}

        # aplicar transformaciones si las hay
        if self.transform:
            sample = self.transform(sample)

        # convertir a tensor
        img_tensor = v2.ToDtype(torch.float32, scale=True)(sample['image'])
        mask_transformed = v2.ToDtype(torch.float32, scale=True)(sample['mask'])

        # convertir máscara a mapa de distancia
        mask_np = np.array(mask_transformed.squeeze(0))
        binary_mask = mask_np > 0
        distance_map = distance_transform_edt(binary_mask)

        # normalizar mapa de distancia
        max_dist = np.max(distance_map)
        if max_dist > 0:
            distance_map = distance_map / max_dist

        # convertir a tensor
        mask_tensor = torch.from_numpy(distance_map).float().unsqueeze(0)
        
        return img_tensor, mask_tensor

# configurar rutas
train_image_root = "BBBC005/selected_dataset/images"
train_mask_root = "BBBC005/selected_dataset/grounds"
val_image_root = "BBBC005/validation/images"
val_mask_root = "BBBC005/validation/grounds"

# buscar archivos
train_image_paths = sorted(glob(os.path.join(train_image_root, "*.TIF")))
train_mask_paths = sorted(glob(os.path.join(train_mask_root, "*.TIF")))

val_image_paths = sorted(glob(os.path.join(val_image_root, "*.TIF")))
val_mask_paths = sorted(glob(os.path.join(val_mask_root, "*.TIF")))

print(f"Imágenes de entrenamiento encontradas: {len(train_image_paths)}")
print(f"Máscaras de entrenamiento encontradas: {len(train_mask_paths)}")
print(f"Imágenes de validación encontradas: {len(val_image_paths)}")
print(f"Máscaras de validación encontradas: {len(val_mask_paths)}")

# parámetros de entrenamiento
num_epochs = 100
batch_size = 4
learning_rate = 1e-4
patience = 10

# configurar dispositivo
if torch.cuda.is_available():
    device = "cuda"
    print("Usando GPU (CUDA)")
else:
    device = "cpu"
    print("Usando CPU")

# definir transformaciones de data augmentation
transform = v2.Compose([
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5),
    v2.RandomRotation(degrees=45),
    v2.RandomAffine(degrees=0, scale=(0.8, 1.2)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

# crear datasets
train_dataset = CellDataset(train_image_paths, train_mask_paths, transform=transform)
val_dataset = CellDataset(val_image_paths, val_mask_paths, transform=transform)

# crear dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# crear modelo
model = UNet().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# variables para early stopping
best_val_loss = float('inf')
epochs_no_improve = 0
best_model_state = None

print("Iniciando entrenamiento...")
print("-" * 50)

# bucle de entrenamiento
for epoch in range(num_epochs):
    
    # fase de entrenamiento
    model.train()
    train_loss = 0
    train_batches = 0
    
    for imgs, masks in train_dataloader:
        # mover datos a GPU/CPU
        imgs = imgs.to(device)
        masks = masks.to(device)
        
        # limpiar gradientes
        optimizer.zero_grad()
        
        # forward pass
        outputs = model(imgs)
        
        # calcular pérdida
        loss = criterion(outputs, masks)
        
        # backward pass
        loss.backward()
        optimizer.step()
        
        # acumular pérdida
        train_loss += loss.item()
        train_batches += 1
    
    # calcular pérdida promedio
    avg_train_loss = train_loss / train_batches

    # fase de validación
    model.eval()
    val_loss = 0
    val_batches = 0
    
    with torch.no_grad():
        for imgs, masks in val_dataloader:
            # mover datos a GPU/CPU
            imgs = imgs.to(device)
            masks = masks.to(device)
            
            # forward pass
            outputs = model(imgs)
            
            # verificar dimensiones
            if outputs.shape != masks.shape:
                masks = nn.functional.interpolate(masks, size=outputs.shape[2:], mode='nearest')
            
            # calcular pérdida
            loss = criterion(outputs, masks)
            
            # acumular pérdida
            val_loss += loss.item()
            val_batches += 1
    
    # calcular pérdida promedio
    avg_val_loss = val_loss / val_batches

    # mostrar resultados
    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"  Train Loss: {avg_train_loss:.6f}")
    print(f"  Val Loss:   {avg_val_loss:.6f}")

    # verificar si mejora
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        best_model_state = model.state_dict()
        print("Nueva mejor pérdida de validación")
    else:
        epochs_no_improve += 1
        print(f"  Sin mejora por {epochs_no_improve} epochs")
    
    # early stopping
    if epochs_no_improve >= patience:
        print(f"Deteniendo entrenamiento después de {epoch+1} epochs")
        print("No hubo mejora en validación")
        break
    
    print("-" * 50)

# guardar el mejor modelo
if best_model_state:
    model.load_state_dict(best_model_state)
    torch.save(model.state_dict(), "cell_counter_nuclear.pth")
    print("Mejor modelo guardado como 'cell_counter_nuclear.pth'")
    print(f"Mejor pérdida de validación: {best_val_loss:.6f}")
else:
    print("No se pudo guardar el modelo")

print("Entrenamiento completado")