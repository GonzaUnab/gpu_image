# gpu_image_pipeline/pipeline.py

import numpy as np

import time

class ImagePipeline:
    def __init__(self, filters):
        self.filters = filters

    def run(self, image):
        result = image
        for filter_func in self.filters:
            time.sleep(0.001)  # Simulación de sincronización
            result = filter_func(result)
        return result