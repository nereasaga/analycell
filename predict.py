import torch
import numpy as np
import cv2
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.morphology import disk, opening, closing
from skimage.measure import label, regionprops
from scipy import ndimage
from PIL import Image
import io
import base64

# configurar dispositivo para GPU o CPU
if torch.cuda.is_available():
    device = "cuda"
    print("Usando GPU (CUDA)")
else:
    device = "cpu"
    print("Usando CPU")

# modelo global, inicializado vacío
model = None

# función para cargar el modelo solo cuando se necesite
def load_model(weights_path="cell_counter_nuclear.pth"):
    global model
    if model is not None:
        return model

    from model import UNet

    model = UNet()
    model = model.to(device)

    # cargar pesos del modelo
    try:
        model_weights = torch.load(weights_path, map_location=device)
        model.load_state_dict(model_weights)
        model.eval()
        print("Modelo cargado exitosamente")
    except FileNotFoundError:
        print("ERROR: No se encontró el archivo del modelo 'cell_counter_nuclear.pth'.")
        print("Asegúrate de que el archivo esté en el directorio actual")
        model = None

    return model

# función para convertir array numpy a base64
def convert_array_to_base64(image_array):
    if image_array is None:
        return None
 
    if image_array.max() <= 1:
        # si está normalizado, multiplicar por 255
        image_array = image_array * 255
        image_array = image_array.astype(np.uint8)
    else:
        image_array = image_array.astype(np.uint8)
    
    # crear imagen PIL
    if len(image_array.shape) == 2:
        # imagen en escala de grises
        pil_image = Image.fromarray(image_array, mode='L')
    else:
        # imagen en color
        pil_image = Image.fromarray(image_array)
    
    # convertir a base64
    img_buffer = io.BytesIO()
    pil_image.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    img_b64 = base64.b64encode(img_buffer.getvalue())
    img_b64_str = img_b64.decode('utf-8')
    
    return img_b64_str

# función ppal para predecir y contar núcleos con parámetros ajustables
def predict_and_count_adjustable(input_image, min_distance=8, sigma=1.0, exclude_border=2):
    global model
    if model is None:
        model = load_model()
        if model is None:
            return 0, None, None, None

    try:
        # escala de grises
        gray_image = input_image.convert("L")
        img_array = np.array(gray_image)
        
        # crear tensor para el modelo
        img_tensor = torch.from_numpy(img_array)
        img_tensor = img_tensor.float()
        img_tensor = img_tensor.unsqueeze(0)  # añadir batch dimension
        img_tensor = img_tensor.unsqueeze(0)  # añadir channel dimension
        img_tensor = img_tensor.to(device)
        
        # normalizar si es necesario
        if img_tensor.max() > 1:
            img_tensor = img_tensor / 255.0
        
        # hacer predicción con el modelo
        with torch.no_grad():
            prediction = model(img_tensor)
            heatmap = prediction.squeeze()
            heatmap = heatmap.cpu().numpy()
        
        print("Predicción completada, procesando heatmap...")
        
        # PROCESAMIENTO DEL MAPA DE CALOR
        
        # aplicar filtro gaussiano si sigma > 0
        if sigma > 0:
            from scipy.ndimage import gaussian_filter
            heatmap_smooth = gaussian_filter(heatmap, sigma=sigma)
        else:
            heatmap_smooth = heatmap.copy()
        
        # filtrado morfológico para limpiar
        morphology_kernel = disk(3)
        heatmap_clean = opening(heatmap_smooth, morphology_kernel)
        
        # crear máscara binaria con umbralización (usar percentil para obtener threshold auto)
        positive_values = heatmap_clean[heatmap_clean > 0]
        if len(positive_values) > 0:
            threshold_value = np.percentile(positive_values, 70)
        else:
            threshold_value = 0.1
        
        binary_mask = heatmap_clean > threshold_value
        binary_mask = binary_mask.astype(np.uint8)
        
        # filtrar regiones por área y forma
        labeled_regions = label(binary_mask)
        filtered_mask = np.zeros_like(binary_mask)
        
        region_props = regionprops(labeled_regions)
        for region in region_props:
            area = region.area
            if area < 20 or area > 2000:
                continue
                
            # filtrar por circularidad
            perimeter = region.perimeter
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)
                if circularity < 0.3:
                    continue
            
            # filtrar por aspect ratio
            bbox = region.bbox
            minr, minc, maxr, maxc = bbox
            height = maxr - minr
            width = maxc - minc
            if width > 0:
                aspect_ratio = height / width
                if aspect_ratio > 3 or aspect_ratio < 0.33:
                    continue
            
            # si pasa todos los filtros, mantener
            coords = region.coords
            for coord in coords:
                filtered_mask[coord[0], coord[1]] = 1
        
        # suavizar con closing
        closing_kernel = disk(2)
        filtered_mask = closing(filtered_mask, closing_kernel)
        
        # encontrar picos locales
        heatmap_for_peaks = heatmap_clean * filtered_mask
        
        peak_coordinates = peak_local_max(
            heatmap_for_peaks,
            min_distance=min_distance,
            threshold_abs=0.3,
            exclude_border=exclude_border,
            num_peaks_per_label=1
        )
        
        # método de respaldo usando centroides
        if len(peak_coordinates) == 0:
            print("No se encontraron picos, usando centroides...")
            labeled_backup = label(filtered_mask)
            centroid_coords = []
            backup_props = regionprops(labeled_backup)
            for region in backup_props:
                centroid = region.centroid
                centroid_coords.append([int(centroid[0]), int(centroid[1])])
            
            if len(centroid_coords) > 0:
                peak_coordinates = np.array(centroid_coords)
            else:
                peak_coordinates = np.array([]).reshape(0, 2)
        
        # verificación de consistencia
        labeled_final_regions = label(filtered_mask)
        num_regions = len(regionprops(labeled_final_regions))
        
        # si hay gran discrepancia, usar centroides como backup
        if len(peak_coordinates) < num_regions * 0.7:
            print(f"Discrepancia detectada: {len(peak_coordinates)} picos vs {num_regions} regiones")
            print("Usando centroides como método de respaldo...")
            
            backup_centroids = []
            for region in regionprops(labeled_final_regions):
                centroid = region.centroid
                backup_centroids.append([int(centroid[0]), int(centroid[1])])
            
            if len(backup_centroids) > len(peak_coordinates):
                peak_coordinates = np.array(backup_centroids)
                print(f"Usando {len(peak_coordinates)} centroides")
        
        # crear marcadores para watershed
        markers_array = np.zeros(heatmap_clean.shape, dtype=int)
        if len(peak_coordinates) > 0:
            for i, coord in enumerate(peak_coordinates):
                markers_array[coord[0], coord[1]] = i + 1
        
        # aplicar watershed si tenemos marcadores
        if markers_array.max() == 0:
            nuclei_count = 0
            result_image = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
            print("No se encontraron núcleos válidos")
        else:
            # calcular transform de distancia para watershed
            distance_transform = ndimage.distance_transform_edt(filtered_mask)
            
            # aplicar watershed
            watershed_labels = watershed(-distance_transform, markers_array, mask=filtered_mask)
            
            # contar núcleos únicos
            unique_labels = np.unique(watershed_labels)
            nuclei_count = len(unique_labels) - 1  # restar el fondo (label 0)
            
            # crear imagen final con contornos
            result_image = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
            
            # encontrar contornos
            watershed_binary = (watershed_labels > 0).astype(np.uint8)
            contours, hierarchy = cv2.findContours(
                watershed_binary, 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            # dibujar contornos verdes
            cv2.drawContours(result_image, contours, -1, (0, 255, 0), 2)
            
            # marcar centros con puntos
            if len(peak_coordinates) > 0:
                for coord in peak_coordinates:
                    cv2.circle(result_image, (coord[1], coord[0]), 3, (255, 0, 0), -1)
        
        # PREPARAR IMÁGENES PARA RETURN
        
        # imagen con contornos
        result_b64 = convert_array_to_base64(result_image)
        
        # máscara binaria
        mask_b64 = convert_array_to_base64(filtered_mask)
        
        # mapa de calor coloreado
        heatmap_normalized = (heatmap_clean * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
        heatmap_b64 = convert_array_to_base64(heatmap_colored)
        
        print(f"Procesamiento completado. Núcleos detectados: {nuclei_count}")
        return nuclei_count, result_b64, mask_b64, heatmap_b64
    
    except Exception as error:
        print(f"Error durante el procesamiento: {error}")
        return 0, None, None, None

# función de compatibilidad con versiones anteriores
def predict_and_count(input_image):
    
    result = predict_and_count_adjustable(input_image)
    count = result[0]
    result_img_b64 = result[1]
    
    # convertir base64 a array si es necesario
    if result_img_b64:
        img_bytes = base64.b64decode(result_img_b64)
        img_pil = Image.open(io.BytesIO(img_bytes))
        result_array = np.array(img_pil)
        return count, result_array
    else:
        return 0, None
