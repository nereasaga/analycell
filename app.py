from flask import Flask, request, render_template, send_from_directory
import os
import io
import base64
from PIL import Image
from predict import predict_and_count_adjustable

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    num_cells = None
    result_image_data = None
    mask_image_data = None
    heatmap_image_data = None
    error_message = None

    if request.method == "POST":
        # verificar si hay archivo
        if "file" not in request.files:
            error_message = "No se encontró archivo en la solicitud."
        else:
            file = request.files["file"]

            # verificar que se seleccionó un archivo
            if file.filename == "":
                error_message = "No se seleccionó ningún archivo."
            else:
                # revisar extensión del archivo
                filename = file.filename
                filename_lower = filename.lower()
                valid_extensions = ['.tif', '.tiff', '.png', '.jpg', '.jpeg']
                is_valid = False
                for ext in valid_extensions:
                    if filename_lower.endswith(ext):
                        is_valid = True
                        break
                
                if not is_valid:
                    error_message = "Formato de archivo no soportado. Use TIFF, PNG, JPG o JPEG."
                else:
                    try:
                        # leer la imagen
                        img_pillow = Image.open(file.stream)
                        
                        # verificar que se cargó bien
                        if img_pillow is None:
                            error_message = "No se pudo cargar la imagen."
                            return render_template(
                                "index.html",
                                error=error_message,
                            )

                        # obtener parámetros del form
                        min_distance_str = request.form.get("min_distance", "8")
                        sigma_str = request.form.get("sigma", "1.0")
                        
                        # convertir a números
                        min_distance = int(min_distance_str)
                        sigma = float(sigma_str)

                        print("Parámetros recibidos:")
                        print("min_distance =", min_distance)
                        print("sigma =", sigma)
                        print("Procesando imagen:", file.filename)
                        print("tamaño:", img_pillow.size)

                        # llamar a la función de predicción
                        result = predict_and_count_adjustable(
                            img_pillow,
                            min_distance=min_distance,
                            sigma=sigma
                        )
                        
                        # extraer resultados
                        num_cells = result[0]
                        result_image_data = result[1]
                        mask_image_data = result[2]
                        heatmap_image_data = result[3]

                        if num_cells is not None and result_image_data is not None:
                            print("Conteo exitoso:")
                            print("núcleos detectados:", num_cells)
                        else:
                            error_message = "No se pudo realizar el conteo de núcleos. Verifique el modelo y la imagen."
                            print("Error en el conteo.")
                            
                    except Exception as e:
                        # manejar errores
                        error_message = "Error durante el procesamiento: " + str(e)
                        print("Excepción:", error_message)

    # renderizar template
    return render_template(
        "index.html",
        num_cells=num_cells,
        result_image_data=result_image_data,
        mask_image_data=mask_image_data,
        heatmap_image_data=heatmap_image_data,
        error=error_message,
    )

# servir imágenes de validación
@app.route('/BBBC005/validation/images/<path:filename>')
def serve_validation_image(filename):
    return send_from_directory('BBBC005/validation/images', filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))