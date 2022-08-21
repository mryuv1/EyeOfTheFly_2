import cv2
import numpy as np


def total_variation(frame):
    gx = cv2.Sobel(frame, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(frame, cv2.CV_64F, 0, 1)
    gmag = gx**2+gy**2
    return np.sum(gmag)