import cv2
import numpy as np
from numba import cuda

def dilation_filter_cpu(image: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)

@cuda.jit
def dilation_kernel(input_img, output_img, width, height):
    x, y = cuda.grid(2)
    if x >= 1 and x < width - 1 and y >= 1 and y < height - 1:
        max_val = 0
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                val = input_img[y + dy, x + dx]
                if val > max_val:
                    max_val = val
        output_img[y, x] = max_val

def dilation_filter_gpu(image: np.ndarray) -> np.ndarray:
    height, width = image.shape
    input_gpu = cuda.to_device(image.astype(np.uint8))
    output_gpu = cuda.device_array_like(image)

    threadsperblock = (16, 16)
    blockspergrid = ((width + 15) // 16, (height + 15) // 16)

    dilation_kernel[blockspergrid, threadsperblock](input_gpu, output_gpu, width, height)
    cuda.synchronize()

    return output_gpu.copy_to_host()