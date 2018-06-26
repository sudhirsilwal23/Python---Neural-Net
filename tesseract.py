# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 16:12:42 2018

@author: Sudee
"""

# import the necessary packages
from PIL import Image
import sys
import pyocr
import pyocr.builders
#import pytesseract
import cv2
import os
 


# load the example image and convert it to grayscale
image = cv2.imread("hindi_data_set/numerals/Test/hindi1.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
# check to see if we should apply thresholding to preprocess the
# image

gray = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
 
# make a check to see if median blurring should be done to remove
# noise

gray = cv2.medianBlur(gray, 3)
 
# write the grayscale image to disk as a temporary file so we can
# apply OCR to it
#filename = "{}.jpg".format(os.getpid())
#print(filename)
cv2.imwrite('gray.jpg', gray)
#filename='gray.jpg'

# load the image as a PIL/Pillow image, apply OCR, and then delete
# the temporary file

 
tools = pyocr.get_available_tools()
if len(tools) == 0:
    print("No OCR tool found")
    sys.exit(1)
tool = tools[0]
print("Will use tool '%s'" % (tool.get_name()))
 
langs = tool.get_available_languages()
lang = langs[0]
 
txt = tool.image_to_string(
    Image.open('gray.jpg'),
    lang=lang,
    builder=pyocr.builders.TextBuilder()
)
#text = pytesseract.image_to_string(image.open(filename))
#os.remove(filename)
#print(text)

# show the output images
cv2.imshow("Image", image)
cv2.imshow("Output", gray)
cv2.waitKey(0)