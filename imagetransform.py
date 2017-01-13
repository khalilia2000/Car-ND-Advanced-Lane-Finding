# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 10:14:07 2017

@author: ali.khalili
"""

import numpy as np
import cv2

class ImageTransform(object):
  
    def __init__(self, images, camera_matrix, dist_matrix, warp_matrix, colorspec='BGR'):
                 
        '''
        Construct an ImageTransform object for manipulating images.
        scaled_dim: scaled dimension of images after pre-processing
        '''
        # initializing variables
        self._num_examples = images.shape[0]
        self._images = images
        self._mtx = camera_matrix   # camera matrix
        self._dist = dist_matrix    # distortion matrix
        self._warp = warp_matrix    # warp matrix
        self._colorspec = colorspec # color space of the images that are passed
        self._blurred = None
        self._undistorted = None
        self._gray = None
        self._HLS = None
        self._RGB = None
        self._R = None
        self._G = None
        self._B = None
        self._H = None
        self._L = None
        self._S = None
      
  
    @property
    def images(self):
        return self._images
  
    @property
    def mtx(self):
        return self._mtx
        
    @property
    def warp(self):
        return self._warp
  
    @property
    def dist(self):
        return self._dist
      
    @property
    def gray(self):
        return self._gray
      
    @property
    def undistorted(self):
        return self._undistorted
      
    @property
    def HLS(self):
        return self._HLS
       
    @property
    def RGB(self):
        return self._RGB

    @property
    def blurred(self):
        return self._blurred

    
    def to_blur(self, kernel_size=3):
        '''
        Applies a Gaussian Noise kernel
        '''
        self._blurred = np.asarray([cv2.GaussianBlur(img, (kernel_size, kernel_size), 0) for img in self._images])


   
    def to_undistort(self) :
        '''
        Undistorts the images based on the camera and distortion matrices
        '''
        # works on images that were subject to gaussian blurring
        if self._blurred is None:
            self.to_blur()
        #
        self._undistorted = np.asarray([cv2.undistort(img, self._mtx, self._dist, None, self._mtx) for img in self._images])
     
   
   
    def to_gray(self):
        """
        Changes the color to grayscale (i.e. turns images to only one channel)
        """
        # checking to see if undistorted images are available. makes undistorted images otherwise
        if self._undistorted is None:
            self.to_undistort()
        # converting grayscale    
        color_change_spec = 'cv2.COLOR_'+self._colorspec+'2GRAY'
        self._gray = np.asarray([cv2.cvtColor(img, eval(color_change_spec)) for img in self._undistorted])
  
  
    
    def to_HLS(self):
        '''
        Changes the color to HLS
        '''
        # checking to see if undistorted images are available. makes undistorted images otherwise
        if self._undistorted is None:
            self.to_undistort()
        # converting grayscale  
        if self._colorspec != 'HLS':
            color_change_spec = 'cv2.COLOR_'+self._colorspec+'2HLS'
            self._HLS = np.asarray([cv2.cvtColor(img, eval(color_change_spec)) for img in self._undistorted])
        else:
            self._HLS = np.copy(self._undistorted)
        # extracting channels
        self._H = np.asarray([img[:,:,0] for img in self._HLS])
        self._L = np.asarray([img[:,:,1] for img in self._HLS])
        self._S = np.asarray([img[:,:,2] for img in self._HLS])
        
  
    
    def to_RGB(self):
        '''
        Changes the color to RGB
        '''
        # checking to see if undistorted images are available. makes undistorted images otherwise
        if self._undistorted is None:
            self.to_undistort()
        # converting grayscale  
        if self._colorspec != 'RGB':
            color_change_spec = 'cv2.COLOR_'+self._colorspec+'2RGB'
            self._RGB = np.asarray([cv2.cvtColor(img, eval(color_change_spec)) for img in self._undistorted])
        else:
            self._RGB = np.copy(self._undistorted)
        # extracting channels
        self._R = np.asarray([img[:,:,0] for img in self._RGB])
        self._G = np.asarray([img[:,:,1] for img in self._RGB])
        self._B = np.asarray([img[:,:,2] for img in self._RGB])
        
  
  
    def get_dir_sobel_thresh(self, orient='x', sobel_kernel=3, thresh=(0, 255), warp=True):
        '''
        Calculates directional gradient, and applies threshold. returns a binary image
        img: grayscaled image
        orient: either 'x' or 'y', and determines the direction of sobel operator
        sobel_kernel: the size of sobel_kernel
        thresh: threshold for binary image creation
        '''
        # only works on gray images
        if self._gray is None:
            self.to_gray()
        # calculating the scaled absolute value of the sobel gradients based on the direction specified.
        sbinary = []
        for img in self._gray:
            if orient == 'x':
                abs_x_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0))
                abs_scaled_sobel = np.uint8(255*abs_x_sobel/np.max(abs_x_sobel))
            if orient == 'y':
                abs_y_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1))
                abs_scaled_sobel = np.uint8(255*abs_y_sobel/np.max(abs_y_sobel))
            # forming the binary image based on thresholds
            sbinary.append(np.zeros_like(abs_scaled_sobel))
            sbinary[-1][(abs_scaled_sobel >= thresh[0]) & (abs_scaled_sobel <= thresh[1])] = 1
        sbinary = np.asarray(sbinary)
        # warp perspective if needed
        if warp and self._warp != None:
            return np.asarray([cv2.WarpPerspective(img, self._warp, (img.shape[1],img.shape[0]), flags=cv2.INTER_LINEAR) for img in sbinary])
        else:    
            return sbinary
  
    
    
    def get_mag_sobel_thresh(self, sobel_kernel=3, thresh=(0, 255), warp=True):
        '''
        Calculates the magnitude of sobel gradient, and applies threshold. returns a binary image
        img: original image
        sobel_kernel: the size of sobel_kernel
        mag_thresh: threshold for binary image creation
        '''
        # only works on gray images
        if self._gray is None:
            self.to_gray()  
        # calculating the scaled magnitude of the sobel gradients
        sbinary = []
        for img in self._gray:
            sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
            sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1)
            sobelg = ((sobelx**2+sobely**2)**0.5)
            abs_sobel = np.absolute(sobelg)
            scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
            # forming the binary image based on thresholds
            sbinary.append(np.zeros_like(scaled_sobel))
            sbinary[-1][(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
        sbinary = np.asarray(sbinary)
        # warp perspective if needed
        if warp and self._warp != None:
            return np.asarray([cv2.WarpPerspective(img, self._warp, (img.shape[1],img.shape[0]), flags=cv2.INTER_LINEAR) for img in sbinary])
        else:    
            return sbinary
   
  
  
    def get_angle_sobel_thresh(self, sobel_kernel=3, thresh=(0, np.pi/2), warp=True):
        '''
        Calculates the direction of sobel gradient, and applies threshold. returns a binary image
        img: original image
        sobel_kernel: the size of sobel_kernel
        mag_thresh: threshold for binary image creation
        '''
        # only works on gray images
        if self._gray is None:
            self.to_gray()
        # calculating the scaled magnitude of the sobel directions
        sbinary = []
        for img in self._gray:
            sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
            sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1)
            abs_sobelx = np.absolute(sobelx)
            abs_sobely = np.absolute(sobely)
            dir_sobel = np.arctan2(abs_sobely, abs_sobelx)
            # forming the binary image based on thresholds
            sbinary.append(np.zeros_like(dir_sobel))
            sbinary[-1][(dir_sobel >= thresh[0]) & (dir_sobel <= thresh[1])] = 1
        sbinary = np.asarray(sbinary)
        # warp perspective if needed
        if warp and self._warp != None:
            return np.asarray([cv2.WarpPerspective(img, self._warp, (img.shape[1],img.shape[0]), flags=cv2.INTER_LINEAR) for img in sbinary])
        else:    
            return sbinary



    def get_R_thresh(self, thresh=(0,255), warp=True):
        '''
        returns a binary image calculated based on the threshold given on the R channel
        '''
        # only works on RGB images
        if self._RGB is None:
            self.to_RGB()
        # forming the binary image based on thresholds
        sbinary = []
        for img in self._R:
            sbinary.append(np.zeros_like(img))
            sbinary[-1][(img >= thresh[0]) & (img <= thresh[1])] = 1
        sbinary = np.asarray(sbinary)
        # warp perspective if needed
        if warp and self._warp != None:
            return np.asarray([cv2.WarpPerspective(img, self._warp, (img.shape[1],img.shape[0]), flags=cv2.INTER_LINEAR) for img in sbinary])
        else:    
            return sbinary
        
        
        
    def get_B_thresh(self, thresh=(0,255), warp=True):
        '''
        returns a binary image calculated based on the threshold given on the B channel
        '''
        # only works on RGB images
        if self._RGB is None:
            self.to_RGB()
        # forming the binary image based on thresholds
        sbinary = []
        for img in self._B:
            sbinary.append(np.zeros_like(img))
            sbinary[-1][(img >= thresh[0]) & (img <= thresh[1])] = 1
        sbinary = np.asarray(sbinary)
        # warp perspective if needed
        if warp and self._warp != None:
            return np.asarray([cv2.WarpPerspective(img, self._warp, (img.shape[1],img.shape[0]), flags=cv2.INTER_LINEAR) for img in sbinary])
        else:    
            return sbinary       
           
        
        
    def get_G_thresh(self, thresh=(0,255), warp=True):
        '''
        returns a binary image calculated based on the threshold given on the B channel
        '''
        # only works on RGB images
        if self._RGB is None:
            self.to_RGB()
        # forming the binary image based on thresholds
        sbinary = []
        for img in self._G:
            sbinary.append(np.zeros_like(img))
            sbinary[-1][(img >= thresh[0]) & (img <= thresh[1])] = 1
        sbinary = np.asarray(sbinary)
        # warp perspective if needed
        if warp and self._warp != None:
            return np.asarray([cv2.WarpPerspective(img, self._warp, (img.shape[1],img.shape[0]), flags=cv2.INTER_LINEAR) for img in sbinary])
        else:    
            return sbinary         
                   
        
        
    def get_H_thresh(self, thresh=(0,255), warp=True):
        '''
        returns a binary image calculated based on the threshold given on the B channel
        '''
        # only works on HLS images
        if self._HLS is None:
            self.to_HLS()
        # forming the binary image based on thresholds
        sbinary = []
        for img in self._H:
            sbinary.append(np.zeros_like(img))
            sbinary[-1][(img >= thresh[0]) & (img <= thresh[1])] = 1
        sbinary = np.asarray(sbinary)
        # warp perspective if needed
        if warp and self._warp != None:
            return np.asarray([cv2.WarpPerspective(img, self._warp, (img.shape[1],img.shape[0]), flags=cv2.INTER_LINEAR) for img in sbinary])
        else:    
            return sbinary          
                           
        
        
    def get_L_thresh(self, thresh=(0,255), warp=True):
        '''
        returns a binary image calculated based on the threshold given on the B channel
        '''
        # only works on HLS images
        if self._HLS is None:
            self.to_HLS()
        # forming the binary image based on thresholds
        sbinary = []
        for img in self._L:
            sbinary.append(np.zeros_like(img))
            sbinary[-1][(img >= thresh[0]) & (img <= thresh[1])] = 1
        sbinary = np.asarray(sbinary)
        # warp perspective if needed
        if warp and self._warp != None:
            return np.asarray([cv2.WarpPerspective(img, self._warp, (img.shape[1],img.shape[0]), flags=cv2.INTER_LINEAR) for img in sbinary])
        else:    
            return sbinary  
                   
        
        
    def get_S_thresh(self, thresh=(0,255), warp=True):
        '''
        returns a binary image calculated based on the threshold given on the B channel
        '''
        # only works on HLS images
        if self._HLS is None:
            self.to_HLS()
        # forming the binary image based on thresholds
        sbinary = []
        for img in self._S:
            sbinary.append(np.zeros_like(img))
            sbinary[-1][(img >= thresh[0]) & (img <= thresh[1])] = 1
        sbinary = np.asarray(sbinary)
        # warp perspective if needed
        if warp and self._warp != None:
            return np.asarray([cv2.WarpPerspective(img, self._warp, (img.shape[1],img.shape[0]), flags=cv2.INTER_LINEAR) for img in sbinary])
        else:    
            return sbinary                    
                    
        
        
    def get_gray_thresh(self, thresh=(0,255), warp=True):
        '''
        returns a binary image calculated based on the threshold given on the B channel
        '''
        # only works on HLS images
        if self._gray is None:
            self.to_gray()
        # forming the binary image based on thresholds
        sbinary = []
        for img in self._gray:
            sbinary.append(np.zeros_like(img))
            sbinary[-1][(img >= thresh[0]) & (img <= thresh[1])] = 1
        sbinary = np.asarray(sbinary)
        # warp perspective if needed
        if warp and self._warp != None:
            return np.asarray([cv2.WarpPerspective(img, self._warp, (img.shape[1],img.shape[0]), flags=cv2.INTER_LINEAR) for img in sbinary])
        else:    
            return sbinary                           
        