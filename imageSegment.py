# -*- coding: utf-8 -*-
"""
imageSegment.py

YOUR WORKING FUNCTION

"""
import cv2
import numpy as np
import matplotlib.pyplot as plt


input_dir = 'dataset/test'
output_dir = 'dataset/output'
gt_dir = 'dataset/groundtruth'

# you are allowed to import other Python packages above
##########################
def segmentImage(img):
    # Inputs
    # img: Input image, a 3D numpy array of row*col*3 in BGR format
    #
    # Output
    # outImg: segmentation image
    #
    #########################################################################
    # ADD YOUR CODE BELOW THIS LINE
    img = cv2.GaussianBlur(img, (5, 5), 0)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    vectorized_img = img.reshape((-1, 3))
    vectorized_img_float = np.float32(vectorized_img)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(vectorized_img_float, 3, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

    outImg = generate_segmentation_mask(img, label, center)
    
   
    # END OF YOUR CODE
    #########################################################################
    return outImg

def assign_cluster_indices(center):
    rb_diff = center[:, 0] - center[:, 2]
    sorted_indices = np.argsort(rb_diff)
    return sorted_indices[0], sorted_indices[1], sorted_indices[-1]  # Background, Rotten, Fruit

def generate_segmentation_mask(img, label, center):
    background_cluster_idx, rotten_cluster_idx, fruit_cluster_idx = assign_cluster_indices(center)
    height, width = img.shape[:2]
    label = label.reshape(height, width)

    mask = np.zeros((height, width), dtype=np.uint8)
    mask[label == background_cluster_idx] = 0  # Background
    mask[label == rotten_cluster_idx] = 1      # Rotten
    mask[label == fruit_cluster_idx] = 2       # Fruit


    kernel = np.ones((7,7), np.uint8)
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel,iterations=2)

    kernel = np.ones((1,7), np.uint8)
    mask = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

    return mask
