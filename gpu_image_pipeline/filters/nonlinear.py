import cv2
import numpy as np
from numba import cuda

def median_filter_cpu(image: np.ndarray) -> np.ndarray:
    return cv2.medianBlur(image, 3)

@cuda.jit
def median_kernel(input_img, output_img, width, height):
    x, y = cuda.grid(2)
    if x >= 1 and x < width - 1 and y >= 1 and y < height - 1:
        window = []
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                window.append(input_img[y + dy, x + dx])
        window.sort()
        output_img[y, x] = window[4]  # valor mediano

def median_filter_gpu(image: np.ndarray) -> np.ndarray:
    height, width = image.shape
    input_gpu = cuda.to_device(image.astype(np.uint8))
    output_gpu = cuda.device_array_like(image)

    threadsperblock = (16, 16)
    blockspergrid = ((width + 15) // 16, (height + 15) // 16)

    median_kernel[blockspergrid, threadsperblock](input_gpu, output_gpu, width, height)
    cuda.synchronize()

    return output_gpu.copy_to_host()