import os
import numpy as np
from scipy import io


import cv2

positive_path = r"C:\Users\Thanh_Tuyet\Downloads\practical-face-detection\practical-face-detection\images\possamples.mat"

negative_path = r"C:\Users\Thanh_Tuyet\Downloads\practical-face-detection\practical-face-detection\images\negsamples.mat"

pos_mat = io.loadmat(file_name=positive_path)["possamples"]
neg_mat = io.loadmat(file_name=negative_path)["negsamples"]

pos_mat = np.transpose(pos_mat, axes=(2, 0, 1))
neg_mat = np.transpose(neg_mat, axes=(2, 0, 1))

print(pos_mat.shape, neg_mat.shape)

cv2.imshow("Anh", pos_mat[1])
cv2.waitKey(0)

hog = cv2.HOGDescriptor()
