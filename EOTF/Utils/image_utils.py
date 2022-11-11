import cv2
import numpy as np


def total_variation(frame):
    gx = cv2.Sobel(frame, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(frame, cv2.CV_64F, 0, 1)
    gmag = gx**2+gy**2
    return np.sum(gmag)


def moving_variance(frame, kernel_size):
    #kernel = np.ones((conv1_kernel_size, conv1_kernel_size)) / (conv1_kernel_size*conv1_kernel_size)
    EX = cv2.blur(frame, kernel_size)
    EX2 = cv2.blur(np.square(frame), kernel_size)
    result = np.square(EX) - EX2
    return result
