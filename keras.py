# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 17:32:38 2018

@author: Sudee
"""
#import numpy as np
###import tensorflow as tf
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print(train_images.shape)
