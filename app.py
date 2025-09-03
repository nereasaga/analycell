# from flask import Flask, request, render_template, send_from_directory
# import os
# import io
# import base64
# from PIL import Image
# from predict import predict_and_count_adjustable

# app = Flask(__name__)

# @app.route("/", methods=["GET", "POST"])
# def index():
#     num_cells = None
#     result_image_data = None
#     mask_image_data = None
#     heatmap_image_data = None
#     error_message = None

#     if request.method == "POST":
#         # verificar si hay archivo
#         if "file" not in request.files:
#             error_message = "No se encontró archivo en la solicitud."
#         else:
#             file = request.files["file"]

#             # verificar que se seleccionó un archivo
#             if file.filename == "":
#                 error_message = "No se seleccionó ningún archivo."
#             else:
#                 # revisar extensión del archivo
#                 filename = file.filename
#                 filename_lower = filename.lower()
#                 valid_extensions = ['.tif', '.tiff', '.png', '.jpg', '.jpeg']
#                 is_valid = False
#                 for ext in valid_extensions:
#                     if filename_lower.endswith(ext):
#                         is_valid = True
#                         break
                
#                 if not is_valid:
#                     error_message = "Formato de archivo no soportado. Use TIFF, PNG, JPG o JPEG."
#                 else:
#                     try:
#                         # leer la imagen
#                         img_pillow = Image.open(file.stream)
                        
#                         # verificar que se cargó bien
#                         if img_pillow is None:
#                             error_message = "No se pudo cargar la imagen."
#                             return render_template(
#                                 "index.html",
#                                 error=error_message,
#                             )

#                         # obtener parámetros del form
#                         min_distance_str = request.form.get("min_distance", "8")
#                         sigma_str = request.form.get("sigma", "1.0")
                        
#                         # convertir a números
#                         min_distance = int(min_distance_str)
#                         sigma = float(sigma_str)

#                         print("Parámetros recibidos:")
#                         print("min_distance =", min_distance)
#                         print("sigma =", sigma)
#                         print("Procesando imagen:", file.filename)
#                         print("tamaño:", img_pillow.size)

#                         # llamar a la función de predicción
#                         result = predict_and_count_adjustable(
#                             img_pillow,
#                             min_distance=min_distance,
#                             sigma=sigma
#                         )
                        
#                         # extraer resultados
#                         num_cells = result[0]
#                         result_image_data = result[1]
#                         mask_image_data = result[2]
#                         heatmap_image_data = result[3]

#                         if num_cells is not None and result_image_data is not None:
#                             print("Conteo exitoso:")
#                             print("núcleos detectados:", num_cells)
#                         else:
#                             error_message = "No se pudo realizar el conteo de núcleos. Verifique el modelo y la imagen."
#                             print("Error en el conteo.")
                            
#                     except Exception as e:
#                         # manejar errores
#                         error_message = "Error durante el procesamiento: " + str(e)
#                         print("Excepción:", error_message)

#     # renderizar template
#     return render_template(
#         "index.html",
#         num_cells=num_cells,
#         result_image_data=result_image_data,
#         mask_image_data=mask_image_data,
#         heatmap_image_data=heatmap_image_data,
#         error=error_message,
#     )

# # servir imágenes de validación
# @app.route('/BBBC005/validation/images/<path:filename>')
# def serve_validation_image(filename):
#     return send_from_directory('BBBC005/validation/images', filename)

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

from flask import Flask, request, render_template, send_from_directory, jsonify
import os
import io
import base64
from PIL import Image
import traceback

app = Flask(__name__)

# Variable global para tracking de imports
import_status = {
    "predict_imported": False,
    "predict_error": None,
    "predict_function": None
}

def safe_import_predict():
    """Importa predict de forma segura y reporta errores detallados"""
    try:
        print("🔄 Intentando importar predict...")
        from predict import predict_and_count_adjustable
        print("✅ predict.py importado exitosamente")
        return predict_and_count_adjustable, None
    except Exception as e:
        error_details = {
            "error": str(e),
            "type": type(e).__name__,
            "traceback": traceback.format_exc()
        }
        print(f"❌ Error importando predict: {error_details}")
        return None, error_details

# Intentar importar al inicio
print("🚀 Iniciando aplicación Flask...")
predict_func, import_error = safe_import_predict()

if predict_func:
    import_status["predict_imported"] = True
    import_status["predict_function"] = predict_func
else:
    import_status["predict_imported"] = False
    import_status["predict_error"] = import_error

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "message": "App is running",
        "predict_available": import_status["predict_imported"],
        "predict_error": import_status["predict_error"]
    })

@app.route('/debug')
def debug():
    """Endpoint para debugging de imports"""
    debug_info = {
        "python_path": os.environ.get('PYTHONPATH', 'Not set'),
        "current_directory": os.getcwd(),
        "files_in_directory": os.listdir('.'),
        "predict_import_status": import_status,
        "torch_available": False,
        "model_file_exists": os.path.exists('cell_counter_nuclear.pth')
    }
    
    # Verificar si torch está disponible
    try:
        import torch
        debug_info["torch_available"] = True
        debug_info["torch_version"] = torch.__version__
        debug_info["cuda_available"] = torch.cuda.is_available()
    except:
        debug_info["torch_available"] = False
    
    return jsonify(debug_info)

@app.route('/predict', methods=['POST'])
def predict_route():
    if not import_status["predict_imported"]:
        return jsonify({
            "error": "Prediction not available",
            "details": import_status["predict_error"]
        }), 500
    
    try:
        # Verificar si se subió un archivo
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Convertir archivo a PIL Image
        image = Image.open(file.stream)
        
        # Obtener parámetros opcionales
        min_distance = int(request.form.get('min_distance', 8))
        sigma = float(request.form.get('sigma', 1.0))
        exclude_border = int(request.form.get('exclude_border', 2))
        
        # Llamar función de predicción
        count, result_img, mask_img, heatmap_img = import_status["predict_function"](
            image, min_distance, sigma, exclude_border
        )
        
        response = {
            "count": count,
            "result_image": result_img,
            "mask_image": mask_img,
            "heatmap_image": heatmap_img
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🌟 Iniciando servidor en puerto {port}")
    print(f"📊 Status de predict: {import_status['predict_imported']}")
    app.run(host='0.0.0.0', port=port, debug=False)