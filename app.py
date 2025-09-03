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
import traceback
import sys

# Configurar logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Variable global para tracking de imports
import_status = {
    "predict_imported": False,
    "predict_error": None,
    "predict_function": None,
    "pil_available": False,
    "torch_available": False,
    "cv2_available": False
}

def check_dependencies():
    """Verificar todas las dependencias una por una"""
    results = {}
    
    # Verificar PIL
    try:
        from PIL import Image
        results["PIL"] = {"status": "OK", "version": getattr(Image, "__version__", "unknown")}
    except Exception as e:
        results["PIL"] = {"status": "ERROR", "error": str(e)}
    
    # Verificar torch
    try:
        import torch
        results["torch"] = {
            "status": "OK", 
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available()
        }
    except Exception as e:
        results["torch"] = {"status": "ERROR", "error": str(e)}
    
    # Verificar cv2
    try:
        import cv2
        results["cv2"] = {"status": "OK", "version": cv2.__version__}
    except Exception as e:
        results["cv2"] = {"status": "ERROR", "error": str(e)}
    
    # Verificar numpy
    try:
        import numpy as np
        results["numpy"] = {"status": "OK", "version": np.__version__}
    except Exception as e:
        results["numpy"] = {"status": "ERROR", "error": str(e)}
    
    # Verificar scikit-image
    try:
        import skimage
        results["skimage"] = {"status": "OK", "version": skimage.__version__}
    except Exception as e:
        results["skimage"] = {"status": "ERROR", "error": str(e)}
    
    return results

def safe_import_predict():
    """Importa predict de forma segura y reporta errores detallados"""
    try:
        logger.info("🔄 Intentando importar predict...")
        from predict import predict_and_count_adjustable
        logger.info("✅ predict.py importado exitosamente")
        return predict_and_count_adjustable, None
    except Exception as e:
        error_details = {
            "error": str(e),
            "type": type(e).__name__,
            "traceback": traceback.format_exc()
        }
        logger.error(f"❌ Error importando predict: {error_details}")
        return None, error_details

# Inicialización con manejo de errores
try:
    logger.info("🚀 Iniciando aplicación Flask...")
    
    # Verificar dependencias básicas primero
    deps = check_dependencies()
    logger.info(f"📦 Estado de dependencias: {deps}")
    
    # Verificar PIL específicamente (lo necesitamos para la ruta básica)
    from PIL import Image
    import_status["pil_available"] = True
    logger.info("✅ PIL disponible")
    
    # Intentar importar predict
    predict_func, import_error = safe_import_predict()
    
    if predict_func:
        import_status["predict_imported"] = True
        import_status["predict_function"] = predict_func
        logger.info("✅ Predict function loaded successfully")
    else:
        import_status["predict_imported"] = False
        import_status["predict_error"] = import_error
        logger.warning("⚠️ Predict function not available")
        
except Exception as e:
    logger.error(f"💥 Error crítico durante la inicialización: {e}")
    logger.error(traceback.format_exc())

@app.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error en index route: {e}")
        return f"Error loading index page: {str(e)}", 500

@app.route('/health')
def health():
    try:
        return jsonify({
            "status": "healthy",
            "message": "App is running",
            "predict_available": import_status.get("predict_imported", False),
            "pil_available": import_status.get("pil_available", False)
        })
    except Exception as e:
        logger.error(f"Error en health route: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/deps')
def check_deps():
    """Endpoint para verificar dependencias"""
    try:
        deps = check_dependencies()
        return jsonify(deps)
    except Exception as e:
        logger.error(f"Error en deps route: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/debug')
def debug():
    """Endpoint para debugging completo"""
    try:
        # Crear info básica primero
        debug_info = {
            "status": "✅ App funcionando correctamente",
            "python_version": sys.version.split()[0],  # Solo la versión, no todo
            "python_path": os.environ.get('PYTHONPATH', 'Not set'),
            "current_directory": os.getcwd(),
            "predict_imported": import_status.get("predict_imported", False),
            "model_file_exists": os.path.exists('cell_counter_nuclear.pth'),
            "key_files_present": {
                "app.py": os.path.exists('app.py'),
                "predict.py": os.path.exists('predict.py'),
                "model.py": os.path.exists('model.py'),
                "templates/index.html": os.path.exists('templates/index.html')
            }
        }
        
        # Agregar error de predict solo si existe y es serializable
        if import_status.get("predict_error"):
            error = import_status["predict_error"]
            debug_info["predict_error"] = {
                "type": error.get("type", "Unknown"),
                "message": error.get("error", "Unknown error")
                # Omitir traceback para evitar problemas de serialización
            }
        
        return jsonify(debug_info)
    except Exception as e:
        logger.error(f"Error en debug route: {e}")
        return jsonify({
            "error": str(e), 
            "message": "Debug endpoint failed but app is working"
        }), 500

@app.route('/predict', methods=['POST'])
def predict_route():
    try:
        if not import_status.get("predict_imported", False):
            return jsonify({
                "error": "Prediction not available",
                "details": import_status.get("predict_error")
            }), 503  # Service Unavailable
        
        # Verificar si se subió un archivo
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Convertir archivo a PIL Image
        from PIL import Image
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
        logger.error(f"Error en predict route: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Error 500: {error}")
    return jsonify({"error": "Internal server error", "details": str(error)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"🌟 Iniciando servidor en puerto {port}")
    logger.info(f"📊 Status de predict: {import_status.get('predict_imported', False)}")
    app.run(host='0.0.0.0', port=port, debug=False)