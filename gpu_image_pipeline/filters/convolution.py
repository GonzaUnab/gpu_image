import cv2
import numpy as np
from numba import cuda

def sobel_filter_cpu(image: np.ndarray) -> np.ndarray:
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobel_x**2 + sobel_y**2)
    return np.uint8(np.clip(sobel, 0, 255))

@cuda.jit
def sobel_kernel(input_img, output_img, width, height):
    x, y = cuda.grid(2)
    if x >= 1 and x < width - 1 and y >= 1 and y < height - 1:
        Gx = (-1 * input_img[y - 1, x - 1] + 1 * input_img[y - 1, x + 1] +
              -2 * input_img[y, x - 1]     + 2 * input_img[y, x + 1] +
              -1 * input_img[y + 1, x - 1] + 1 * input_img[y + 1, x + 1])

        Gy = (-1 * input_img[y - 1, x - 1] + -2 * input_img[y - 1, x] + -1 * input_img[y - 1, x + 1] +
               1 * input_img[y + 1, x - 1] +  2 * input_img[y + 1, x] +  1 * input_img[y + 1, x + 1])

        magnitude = np.sqrt(Gx**2 + Gy**2)
        output_img[y, x] = min(255, magnitude)

def sobel_filter_gpu(image: np.ndarray) -> np.ndarray:
    height, width = image.shape
    input_gpu = cuda.to_device(image.astype(np.float32))
    output_gpu = cuda.device_array_like(image)

    threadsperblock = (16, 16)
    blockspergrid_x = (width + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (height + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    sobel_kernel[blockspergrid, threadsperblock](input_gpu, output_gpu, width, height)
    cuda.synchronize()

    return output_gpu.copy_to_host().astype(np.uint8)