# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 10:14:07 2017

@author: ali.khalili
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import gridspec

class ImageTransform(object):
  
  def __init__(self, images, mtx, dist):
               
    """
    Construct an ImageTransform object for manipulating images.
    scaled_dim: scaled dimension of images after pre-processing
    """
    
    self._num_examples = images.shape[0]
    self._images = images
    self._mtx = mtx     # camera matrix
    self._dist = dist   # distortion matrix
    
    
    #
    self._gray = np.zeros_like(self._images)
    

  @property
  def images(self):
    return self._images

  @property
  def mtx(self):
    return self._mtx

  @property
  def dist(self):
    return self._dist
    
  @property
  def gray(self):
    return self._abs_x_sobel
    
  
  def color_to_gray(self):
    """
    Changes the color to grayscale (i.e. turns images to only one channel)
    """
    self._gray = np.asarray([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in self._images])
    pass

  
  def undistort(self) :
    """
    Undistorts the images based on the camera and distortion matrices
    """
    self._images = np.asarray([cv2.undistort(img, self._mtx, self._dist, None, self._mtx) for img in self._images])
    pass


  def abs_sobel_thresh(self, orient='x', sobel_kernel=3, thresh=(0, 255)):
    """
    Calculates directional gradient, and applies threshold. returns a binary image
    img: grayscaled image
    orient: either 'x' or 'y', and determines the direction of sobel operator
    sobel_kernel: the size of sobel_kernel
    thresh: threshold for binary image creation
    """
    
    # calculating the scaled absolute value of the sobel gradients based on the direction specified.
    if orient == 'x':
      abs_x_sobel = np.absolute(np.asarray([cv2.Sobel(img, cv2.CV_64F, 1, 0) for img in self._gray]))
      abs_scaled_sobel = np.uint8(255*abs_x_sobel/np.max(abs_x_sobel))
    if orient == 'y':
      abs_y_sobel = np.absolute(np.asarray([cv2.Sobel(img, cv2.CV_64F, 0, 1) for img in self._gray]))
      abs_scaled_sobel = np.uint8(255*abs_y_sobel/np.max(abs_y_sobel))
      
    # forming the binary image based on thresholds
    sbinary = np.zeros_like(abs_scaled_sobel)
    sbinary[(abs_scaled_sobel >= thresh[0]) & (abs_scaled_sobel <= thresh[1])] = 1
    
    # returning the binary image    
    return sbinary

  
  def mag_thresh(self, sobel_kernel=3, mag_thresh=(0, 255)):
    '''
    Calculates the magnitude of sobel gradient, and applies threshold. returns a binary image
    img: original image
    sobel_kernel: the size of sobel_kernel
    mag_thresh: threshold for binary image creation
    '''

    # calculating the scaled magnitude of the sobel gradients
    sobelx = np.asarray([cv2.Sobel(img, cv2.CV_64F, 1, 0) for img in self._gray])
    sobely = np.asarray([cv2.Sobel(img, cv2.CV_64F, 0, 1) for img in self._gray])
    sobelg = ((sobelx**2+sobely**2)**0.5)
    abs_sobel = np.absolute(sobelg)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    
    # forming the binary image based on thresholds
    sbinary = np.zeros_like(scaled_sobel)
    sbinary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    
    # returning the binary image
    return sbinary
 

  def dir_threshold(self, sobel_kernel=3, thresh=(0, np.pi/2)):
    '''
    Calculates the direction of sobel gradient, and applies threshold. returns a binary image
    img: original image
    sobel_kernel: the size of sobel_kernel
    mag_thresh: threshold for binary image creation
    '''

    # calculating the scaled magnitude of the sobel directions
    sobelx = np.asarray([cv2.Sobel(img, cv2.CV_64F, 1, 0) for img in self._gray])
    sobely = np.asarray([cv2.Sobel(img, cv2.CV_64F, 0, 1) for img in self._gray])
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    dir_sobel = np.arctan2(abs_sobely, abs_sobelx)
    
    # forming the binary image based on thresholds
    sbinary = np.zeros_like(dir_sobel)
    sbinary[(dir_sobel >= thresh[0]) & (dir_sobel <= thresh[1])] = 1
    
    # returning the binary image
    return sbinary
