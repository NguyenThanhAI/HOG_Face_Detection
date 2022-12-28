import os
from typing import List, Optional
import numpy as np

import cv2


base_dim: int = 50

scale = np.array([1, 1.5], dtype=np.float32)

width = base_dim / np.sqrt(scale)
height = base_dim * np.sqrt(scale)

print(height, width)

image_size = (500, 500)
stride_height = 10
stride_width = 10

y = np.array(list(range(int(np.ceil(np.max(height))), int(np.floor(image_size[0] - np.max(height))), stride_height)))

x = np.array(list(range(int(np.ceil(np.max(width))), int(np.floor(image_size[1] - np.max(width))), stride_width)))

centers = np.meshgrid(y, x)

print(centers)