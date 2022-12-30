import os
from typing import Dict, List, Optional
import joblib
from joblib import Parallel, delayed
from tqdm import tqdm

import numpy as np
from skimage import io, color, transform, feature
import cv2

image_path = r"C:\Users\Thanh_Tuyet\Downloads\practical-face-detection\practical-face-detection\images\img1.jpg"

base_dim: int = 50
scales = [0.25, 0.5, 0.75, 1., 1.25, 1.5, 1.75, 2.]
ratios = [1, 1.2]
offset = 0.5
stride = 0.1

threshold = 0.9

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
image = io.imread(image_path)
result_image = image.copy()
#image_size = (500, 500)
image_size = tuple(list(image.shape[:2]))

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
        
#image = np.ones(shape=[image_size[0], image_size[1], 3], dtype=np.uint8) * 255
for i, rect in enumerate(anchors[0.75][1]):
    cv2.rectangle(image, (int(rect[1]), int(rect[0])), (int(rect[3]), int(rect[2])), color=(0, 0, 255), thickness=1)
    cv2.putText(image, text=str(i+1), org=(int(rect[1])+10, int(rect[0])+10), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.3, color=(0, 255, 0))
cv2.imshow("Anh", image[:, :, ::-1])
cv2.waitKey(0)
dict_crop_images = {}
gray_image = color.rgb2gray(image)
for scale in anchors.keys():
    dict_crop_images[scale] = {}
    for ratio in anchors[scale].keys():
        print("Scale: {}, ratio: {}".format(scale, ratio))
        crop_images = []
        for rect in tqdm(anchors[scale][ratio]):
            crop_images.append(gray_image[int(rect[0]):int(rect[2]), int(rect[1]):int(rect[3])])

        crop_images = np.stack(crop_images, axis=0)
        dict_crop_images[scale][ratio] = crop_images
        print(crop_images.shape)

hog_features = {}     
        
for scale in dict_crop_images.keys():
    hog_features[scale] = {}
    for ratio in dict_crop_images[scale].keys():
        print("Scale: {}, ratio: {}".format(scale, ratio))
        features = []
        crop_images = dict_crop_images[scale][ratio]
        for img in tqdm(crop_images):
            resized_img = transform.resize(img, output_shape=(24, 24))
            resized_img = (resized_img - np.mean(resized_img)) / np.std(resized_img)
            feat = feature.hog(image=resized_img, orientations=9, cells_per_block=(2, 2), block_norm="L2")
            features.append(feat)
        features = np.stack(features, axis=0)
        print(features.shape)
        hog_features[scale][ratio] = features
        

model = joblib.load("svm_model.pkl")

bboxes = []
probs = {}
chosen_probs = []
for scale in hog_features.keys():
    probs[scale] = {}
    for ratio in hog_features[scale].keys():
        print("Scale: {}, ratio: {}".format(scale, ratio))
        features = hog_features[scale][ratio]
        predict_prob = model.predict_proba(features)[:, 1]
        #print(predict_prob.shape)
        probs[scale][ratio] = predict_prob
        
        pos_anchor = np.where(predict_prob > threshold)[0]
        '''predict_prob = model.predict(features)
        pos_anchor = np.where(predict_prob == 1)[0]'''
        if pos_anchor.shape[0] > 0:
            pos_boxes = anchors[scale][ratio][pos_anchor]
            pos_probs = predict_prob[pos_anchor]
            bboxes.append(pos_boxes)
            chosen_probs.append(pos_probs)
            print(pos_boxes.shape)
            assert pos_boxes.shape[0] == pos_probs.shape[0]
            
bboxes = np.concatenate(bboxes, axis=0)
chosen_probs = np.concatenate(chosen_probs, axis=0)
print(bboxes, chosen_probs)

for rect in bboxes:
    cv2.rectangle(result_image, (int(rect[1]), int(rect[0])), (int(rect[3]), int(rect[2])), color=(0, 0, 255), thickness=1)

cv2.imshow("Anh", result_image[:, :, ::-1])
cv2.waitKey(0)