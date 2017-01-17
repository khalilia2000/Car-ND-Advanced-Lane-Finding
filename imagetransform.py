# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 10:14:07 2017

@author: ali.khalili
"""

import cv2
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np


class ImageTransform(object):
  
    def __init__(self, images, labels, camera_matrix, dist_matrix, warp_matrix, colorspec='BGR'):
                 
        '''
        Construct an ImageTransform object for manipulating images.
        scaled_dim: scaled dimension of images after pre-processing
        '''
        # initializing variables
        self._num_examples = images.shape[0]
        self._original_images = images
        self._labels = labels
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
        self._processed_images = None
        self._birds_eye_original = None
        self._birds_eye_processed = None
        # viewport points
        # x: is in percentage of the width
        # y: is in percentage of the image height
        # P0 to P3 form the source viewport
        self._p_0 = np.float32([[0.10,1.0]])
        self._p_1 = np.float32([[0.445,0.62]])
        self._p_2 = np.float32([[0.555,0.62]])
        self._p_3 = np.float32([[0.90,1.0]])
        # q0 to q3 form the destination viewport
        self._q_0 = np.float32([[0.25,1.0]])
        self._q_1 = np.float32([[0.25,0.2]])
        self._q_2 = np.float32([[0.75,0.2]])
        self._q_3 = np.float32([[0.75,1.0]])
        #
        # apply gaussian blura and camera undistortion, and convert to RGB
        self.to_blur()
        self.to_undistort()
        self.to_RGB()
      
  
  
    @property
    def original_images(self):
        return self._original_images
    
    @original_images.setter
    def original_images(self, value):
        self._original_images = value
        self._num_examples = self._original_images.shape[0]
  
    @property
    def labels(self):
        return self._labels
        
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

    @property
    def processed_images(self):
        return self._processed_images
    
    @processed_images.setter
    def processed_images(self, value):
        self._processed_images = value
       
    @property
    def birds_eye_original(self):
        return self._birds_eye_original

    @property
    def birds_eye_processed(self):
        return self._birds_eye_processed


    def draw_lines(self, lines, color=[255, 0, 0], thickness=2, original=True, processed=False):
        """
        This function shows all of the specified lines on the photo
        color: color in original color space
        original: if True lines will be drawn on undistorted images formed from originals
        processed: if True lines will be drawn on processed images also
        """
        if original:
            for line in lines:
                for p1,p2 in line:
                    for img in self._undistorted:
                        img_size = (img.shape[1],img.shape[0])
                        p1_x = int(p1[0][0]*img_size[0])
                        p1_y = int(p1[0][1]*img_size[1])
                        p2_x = int(p2[0][0]*img_size[0])
                        p2_y = int(p2[0][1]*img_size[1])
                        cv2.line(img, (p1_x,p1_y), (p2_x,p2_y), color, thickness)
            # apply gaussian blura and camera undistortion, and convert to RGB
            self.to_RGB()
            
        if processed:
            for line in lines:
                for p1,p2 in line:
                    for img in self._processed_images:
                        img_size = (img.shape[1],img.shape[0])
                        p1_x = int(p1[0][0]*img_size[0])
                        p1_y = int(p1[0][1]*img_size[1])
                        p2_x = int(p2[0][0]*img_size[0])
                        p2_y = int(p2[0][1]*img_size[1])
                        cv2.line(img, (p1_x,p1_y), (p2_x,p2_y), color, thickness)
   

    def draw_viewport(self, original=True, processed=False, src_viewport=True, dst_viewport=False):
        '''
        plots the viewports on the images:
        original: if operation to be performed on the original_images
        processed: if operation to be performed on the processed images
        src_viewport: if source viewport to be drawn
        dst_viewport: if destination viewport to be drawn (for warp perspective)
        '''
        if src_viewport:
            self.draw_lines([[[self._p_0, self._p_1]]],original=original,processed=processed)
            self.draw_lines([[[self._p_1, self._p_2]]],original=original,processed=processed)
            self.draw_lines([[[self._p_2, self._p_3]]],original=original,processed=processed)
            self.draw_lines([[[self._p_3, self._p_0]]],original=original,processed=processed)
        if dst_viewport:
            self.draw_lines([[[self._q_0, self._q_1]]],original=original,processed=processed)
            self.draw_lines([[[self._q_1, self._q_2]]],original=original,processed=processed)
            self.draw_lines([[[self._q_2, self._q_3]]],original=original,processed=processed)
            self.draw_lines([[[self._q_3, self._q_0]]],original=original,processed=processed)


    def weighted_img(self, img, initial_img, α=0.8, β=1., λ=0.):
        """
        `img` is the processed binary image with lines drawn on it.
        
        `initial_img` should be the image before any processing.
        
        The result image is computed as follows:
        
        initial_img * α + img * β + λ
        NOTE: initial_img and img must be the same shape!
        """
        return cv2.addWeighted(initial_img, α, img, β, λ)    



    def to_birds_eye(self, original=True, processed=True):
        """
        to be implemented.
        """
        
        if original:
            # making sure RGB image has been formed
            self.to_RGB()
            # warping perspectives
            beo_list = []
            for img in self._RGB:
                img_size = (img.shape[1], img.shape[0])
                src_viewport = np.float32([[self._p_0[0]],[self._p_1[0]],
                                           [self._p_2[0]],[self._p_3[0]]]) * np.float32(img_size)
                dst_viewport = np.float32([[self._q_0[0]],[self._q_1[0]],
                                           [self._q_2[0]],[self._q_3[0]]]) * np.float32(img_size)
                M = cv2.getPerspectiveTransform(src_viewport, dst_viewport)
                beo_list.append(cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR))
            self._birds_eye_original = np.asarray(beo_list)
        if processed:
            # warping perspectives
            bep_list = []
            for img in self._processed_images:
                img_size = (img.shape[1], img.shape[0])
                src_viewport = np.float32([[self._p_0[0]],[self._p_1[0]],
                                           [self._p_2[0]],[self._p_3[0]]]) * np.float32(img_size)
                dst_viewport = np.float32([[self._q_0[0]],[self._q_1[0]],
                                           [self._q_2[0]],[self._q_3[0]]]) * np.float32(img_size)
                M = cv2.getPerspectiveTransform(src_viewport, dst_viewport)
                bep_list.append(cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR))
            self._birds_eye_processed = np.asarray(bep_list)

    
    def to_blur(self, kernel_size=3):
        '''
        Applies a Gaussian Noise kernel
        '''
        self._blurred = np.asarray([cv2.GaussianBlur(img, (kernel_size, kernel_size), 0) for img in self._original_images])


   
    def to_undistort(self) :
        '''
        Undistorts the images based on the camera and distortion matrices
        '''
        #
        self._undistorted = np.asarray([cv2.undistort(img, self._mtx, self._dist, None, self._mtx) for img in self._original_images])
     
   
   
    def to_gray(self):
        """
        Changes the color to grayscale (i.e. turns images to only one channel)
        """
        # should be applied on undistorted images
        # converting grayscale    
        color_change_spec = 'cv2.COLOR_'+self._colorspec+'2GRAY'
        self._gray = np.asarray([cv2.cvtColor(img, eval(color_change_spec)) for img in self._undistorted])
  
  
    
    def to_HLS(self):
        '''
        Changes the color to HLS
        '''
        # should be applied on undistorted images
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
        # should be applied on undistorted images
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
        
  
  
    def get_dir_sobel_thresh(self, orient='x', sobel_kernel=2, thresh=(0, 255), warp=True):
        '''
        Calculates directional gradient, and applies threshold. returns a binary image
        img: grayscaled image
        orient: either 'x' or 'y', and determines the direction of sobel operator
        sobel_kernel: the size of sobel_kernel
        thresh: threshold for binary image creation
        '''
        # only works on gray images
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
        # only works on grayscale images
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
        
        
    def _plot_image(self, ax_list, grid_fig, grid_index, img, label="", cmap=None):
        '''
        helper function that plots one image in the grid space and passess the appended axis list
        ax_list: list of axes pertaining to the grid space
        grid_fig: grid space object
        grid_index: index in the grid to plot onto
        img: image to be plotted
        label: label to be shown above the image
        cmap: cmap
        '''
        ax_list.append(plt.subplot(grid_fig[grid_index]))
        
        #if img.shape[2] == 1:
        #    img = img.reshape(img.shape[0],img.shape[1])
                  
        ax_list[-1].imshow(img, cmap=cmap)
        ax_list[-1].axis('off')
        ax_list[-1].set_aspect('equal')
        y_lim = ax_list[-1].get_ylim()
        if label:
            ax_list[-1].text(0,int(-1*y_lim[0]*0.05),label)
        #
        return ax_list 
     


    def plot_comparison(self, birds_eye=False):
        """
        plotting 2 columns of images with assoicated labels
        the first column will be the self._original_images
        the second column will be the self._processed_images
        """
        # changing images to RGB
        self.to_RGB()
        
        # initializing variables   
        if not birds_eye:
            org_images = self._RGB
            rev_images = self._processed_images
        else:
            self.to_birds_eye()
            org_images = self._birds_eye_original
            rev_images = self._birds_eye_processed
        org_labels = self._labels
        rev_labels = None
        n_rows = self._num_examples
        n_cols = 2
        
        # creating an array of images to plot
        imgs_to_plot = []
        for img1, img2 in zip(org_images, rev_images):
            imgs_to_plot.append(img1)
            imgs_to_plot.append(img2)
        
        # creating labels for all images
        labels = np.empty(shape=(org_images.shape[0],2), dtype=np.dtype((str, 255)))
        if org_labels is not None:
            for i in range(org_images.shape[0]):
                labels[i, 0] = org_labels[i]
        if rev_labels is not None:
            for i in range(rev_images.shape[0]):
                labels[i, 1] = rev_labels[i]
        
        # creating color maps for all images
        cmaps = ['gray' for i in range(len(org_images)+len(rev_images))]
        cmaps = np.asarray(cmaps)
        
        # creating the grid space
        hspace = 0.2    # distance between images vertically
        wspace = 0.01   # distance between images horizontally
        g_fig = gridspec.GridSpec(n_rows,n_cols) 
        g_fig.update(wspace=wspace, hspace=hspace)
        
        # setting up the figure
        size_factor = 4.5
        aspect_ratio = 1.777
        fig_w_size = n_cols*size_factor*aspect_ratio+(n_cols-1)*wspace
        fig_h_size = n_rows*size_factor+(n_rows-1)*hspace
        plt.figure(figsize=(fig_w_size,fig_h_size))
        
        # plotting the images
        ax_list = []
        for i in range(n_rows*n_cols):
            ax_list = self._plot_image(ax_list, g_fig, i, imgs_to_plot[i], labels.ravel()[i], cmap=cmaps.ravel()[i]) 