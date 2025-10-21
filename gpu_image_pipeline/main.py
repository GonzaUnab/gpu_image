import time
import cv2
import numpy as np
import os

# 📥 Cargar imagen
image_path = os.path.join("images", "input", "sample.jpg")
image_cpu = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if image_cpu is None:
    raise FileNotFoundError(f"No se encontró la imagen en {image_path}")

# 🧪 Importar pipeline y filtros CPU
from image_pipeline import ImagePipeline
from filters.convolution import sobel_filter_cpu
from filters.nonlinear import median_filter_cpu
from filters.morphological import dilation_filter_cpu

# 🧠 Crear pipeline CPU
pipeline_cpu = ImagePipeline([
    sobel_filter_cpu,
    median_filter_cpu,
    dilation_filter_cpu
])

# ⏱️ Ejecutar pipeline CPU
start_cpu = time.time()
result_cpu = pipeline_cpu.run(image_cpu)
end_cpu = time.time()

# 📊 Mostrar tiempo de ejecución
print(f"Tiempo de ejecución CPU: {end_cpu - start_cpu:.4f} segundos")

# 💾 Guardar resultados
os.makedirs(os.path.join("images", "output"), exist_ok=True)
cv2.imwrite("images/output/result_cpu.jpg", result_cpu)
cv2.imwrite("images/output/comparacion.jpg", np.hstack([image_cpu, result_cpu]))

# 👁️ Mostrar resultados
cv2.imshow("Original", image_cpu)
cv2.imshow("Procesado CPU", result_cpu)
cv2.waitKey(0)
cv2.destroyAllWindows()