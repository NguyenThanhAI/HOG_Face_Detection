import os
from typing import Dict, List, Optional
import joblib
from joblib import Parallel, delayed
from tqdm import tqdm

import numpy as np
from skimage import io, color, transform, feature
from skimage.transform import pyramid_gaussian
import cv2
import imutils
from nms import non_max_suppression

def sliding_window(image, window_size, step_size):
    for y in range(0, image.shape[0], step_size[1]):
        for x in range(0, image.shape[1], step_size[0]):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])


if __name__ == "__main__":
    #image_path = r"images\img1.jpg"
    image_path = r"images\167877cf2d05f45bad14.jpg"
    model_path = "svm_model.pkl"
    model = joblib.load(model_path)
    (win_w, win_h) = (24, 24)
    window_size = (win_w, win_h)
    downscale = 1.5
    step_size = (5, 5)
    orientations = 9
    pixels_per_cell = (8, 8)
    cells_per_block = (2, 2)
    threshold = 1.0
    nmsthreshold = 0.3
    #img = io.imread(image_path)
    img = cv2.imread(image_path)
    img = imutils.resize(img, width=600)
    scale = 0
    detections = []
    for resized in pyramid_gaussian(img, downscale=1.5):
        print(resized.shape)
        for (x, y, window) in sliding_window(resized, window_size=window_size, step_size=step_size):
            if window.shape[0] != win_h or window.shape[1] !=win_w or window.shape[2] != 3:
                continue
            window=color.rgb2gray(window)
            window = (window - np.mean(window)) / (np.std(window) + 0.001)
            fds = feature.hog(window, orientations, pixels_per_cell, cells_per_block, block_norm='L2')
            fds = fds.reshape(1, -1)
            pred = model.predict(fds)
            
            if pred == 1:
                if model.decision_function(fds) > threshold: 
                    #print("Detection:: Location -> ({}, {})".format(x, y))
                    #print("Scale ->  {} | Confidence Score {} \n".format(scale,model.decision_function(fds)))
                    detections.append((int(x * (downscale**scale)), int(y * (downscale**scale)), model.decision_function(fds),
                                       int(window_size[0]*(downscale**scale)), # create a list of all the predictions found
                                       int(window_size[1]*(downscale**scale))))
                    
        scale += 1
        
    clone = resized.copy()
    #for (x_tl, y_tl, _, w, h) in detections:
    #    cv2.rectangle(img, (x_tl, y_tl), (x_tl + w, y_tl + h), (0, 0, 255), thickness = 2)
    rects = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in detections])
    sc = [score[0] for (x, y, score, w, h) in detections]
    print("detection confidence score: ", sc)
    sc = np.array(sc)
    pick = non_max_suppression(rects, probs=sc, overlapThresh=nmsthreshold)
    
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(img, (xA, yA), (xB, yB), (0,255,0), 2)
    cv2.imshow("Result after NMS", img)
    cv2.waitKey(0)
    
    cv2.imwrite(os.path.join("results", "result_" + os.path.basename(image_path)), img)