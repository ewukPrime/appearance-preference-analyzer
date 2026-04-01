import mediapipe as mp
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision as mpv
import numpy as np
import cv2


class ImagePreprocessor:
    def __init__(self, path_image):
        image_bgr = cv2.imread(path_image)
        
        


    def without_background(self):
        image = self.image_rgb


