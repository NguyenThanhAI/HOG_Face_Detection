import os
from typing import Dict, List, Optional

from tqdm import tqdm

import numpy as np
from skimage import color
import cv2


base_dim: int = 50
scales = [0.5, 1., 2., 3., 4.]
ratios = [1, 1.2]
offset = 0.5
stride = 0.2

#width = base_dim * scales[:, np.newaxis] / np.sqrt(ratios[np.newaxis, :])
#height = base_dim * scales[:, np.newaxis] * np.sqrt(ratios[np.newaxis, :])
#
#print(height, width)

#image_size = (500, 500)
#stride_height = 10
#stride_width = 10
#
#y = np.array(list(range(int(np.ceil(np.max(height))), int(np.floor(image_size[0] - np.max(height))), stride_height)))
#
#x = np.array(list(range(int(np.ceil(np.max(width))), int(np.floor(image_size[1] - np.max(width))), stride_width)))
#
#centers = np.meshgrid(y, x)
#
#print(centers)

image_size = (500, 500)

anchors: Dict[float, Dict[float, np.ndarray]] = {}

for scale in scales:
    anchors[scale] = {}
    for ratio in ratios:
        print("Scale: {}, ratio: {}".format(scale, ratio))
        height = base_dim * scale * np.sqrt(ratio)
        width = base_dim * scale / np.sqrt(ratio)
        #print(height, width)
        stride_height = int(stride * height)
        stride_width = int(stride * width)
        print(stride_height, stride_width)
        y = np.array(list(range(int(np.ceil(height/2)), int(np.floor(image_size[0] - height/2)), stride_height))) + offset
        x = np.array(list(range(int(np.ceil(width/2)), int(np.floor(image_size[1] - width/2)), stride_width))) + offset
        #print(y, x)
        centers = np.meshgrid(y, x)
        top = (centers[0] - height/2).reshape(-1)
        bottom = (centers[0] + height/2).reshape(-1)
        left = (centers[1] - width/2).reshape(-1)
        right = (centers[1] + width/2).reshape(-1)
        coords = np.stack([top, left, bottom, right], axis=0)
        coords = coords.transpose()
        print(coords)
        anchors[scale][ratio] = coords
        
image = np.ones(shape=[image_size[0], image_size[1], 3], dtype=np.uint8) * 255
#for i, rect in enumerate(anchors[1][1.2]):
#    cv2.rectangle(image, (int(rect[1]), int(rect[0])), (int(rect[3]), int(rect[2])), color=(0, 0, 255), thickness=1)
#    cv2.putText(image, text=str(i+1), org=(int(rect[1])+10, int(rect[0])+10), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.3, color=(0, 255, 0))
#cv2.imshow("Anh", image)
#cv2.waitKey(0)

gray_image = color.rgb2gray(image)
for scale in anchors.keys():
    for ratio in anchors[scale].keys():
        print("Scale: {}, ratio: {}".format(scale, ratio))
        crop_images = []
        for rect in tqdm(anchors[scale][ratio]):
            crop_images.append(gray_image[int(rect[0]):int(rect[2]), int(rect[1]):int(rect[3])])

        crop_images = np.stack(crop_images, axis=0)
        print(crop_images.shape)