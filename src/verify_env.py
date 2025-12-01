# src/verify_env.py
import sys
import tensorflow as tf
import numpy as np
from deepface import DeepFace
import cv2
print("python:", sys.version.splitlines()[0])
print("tf:", tf.__version__)
print("keras import ok:", hasattr(tf, 'keras'))
print("numpy:", np.__version__)
print("deepface:", DeepFace.__version__)
print("opencv:", cv2.__version__)
