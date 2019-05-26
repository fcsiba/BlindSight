#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 23:33:41 2019

@author: rehan
"""

from PIL import Image
import pytesseract
import argparse
import cv2
import os

image = cv2.imread('text.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.threshold(gray, 0, 255,
cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

 
# write the grayscale image to disk as a temporary file so we can
# apply OCR to it
filename = "{}.png".format(os.getpid())
cv2.imwrite(filename, gray)


try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"/usr/local/Cellar/tesseract/4.0.0_1/bin/tesseract"

os.system('convert text.jpg -resize 400% -type Grayscale input.tif')   



print(pytesseract.image_to_string(Image.open('input.tif')))
