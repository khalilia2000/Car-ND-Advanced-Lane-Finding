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
  
    def __init__(self, images, labels, camera_matrix, dist_matrix, colorspec='BGR'):
                 
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
        self._p_1 = np.float32([[0.43,0.65]])
        self._p_2 = np.float32([[0.57,0.65]])
        self._p_3 = np.float32([[0.90,1.0]])
        # q0 to q3 form the destination viewport
        self._q_0 = np.float32([[0.20,1.0]])
        self._q_1 = np.float32([[0.20,0.3]])
        self._q_2 = np.float32([[0.80,0.3]])
        self._q_3 = np.float32([[0.80,1.0]])
        #
        # apply gaussian blura and camera undistortion, and convert to RGB
        self.to_blur()
        self.to_undistort()
        self.to_RGB()
        
        # Define conversions in x and y from pixels space to meters
        self._ym_per_pix = 30/720 # meters per pixel in y dimension
        self._xm_per_pix = 3.7/700 # meteres per pixel in x dimension
      
  
  
    @property
    def original_images(self):
        return self._original_images
    
    @original_images.setter
    def original_images(self, value):
        self._original_images = value
        self._num_examples = self._original_images.shape[0]
  
    
    def set_labels(self, value):
        self._labels = value        
    
    @property
    def labels(self):
        return self._labels
        
    @property
    def mtx(self):
        return self._mtx
        
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
        self._undistorted = np.asarray([cv2.undistort(img, self._mtx, self._dist, None, self._mtx) for img in self._blurred])
     
   
   
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
        
  

    def get_canny(self, thresh=(53,55)):
        # only works on gray images
        self.to_gray()
        # calculate canney edge transform
        sbinary = []
        for img in self._gray:
            sbinary.append(cv2.Canny(img, thresh[0], thresh[1]))
            sbinary[-1][sbinary[-1]>0] = 1
        sbinary = np.asarray(sbinary)
        return sbinary
  
    
    
    
    def get_dir_sobel_thresh(self, orient='x', sobel_kernel=2, thresh=(0, 255)):
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
        return sbinary
  
    
    
    def get_mag_sobel_thresh(self, sobel_kernel=3, thresh=(0, 255)):
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
        return sbinary
   
  
  
    def get_angle_sobel_thresh(self, sobel_kernel=3, thresh=(0, np.pi/2), blur_kernel=9, blur_thresh=0.65):
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
            # apply gaussian blur to reduce             
            sbinary[-1] = cv2.GaussianBlur(sbinary[-1], (blur_kernel, blur_kernel), 0)
            sbinary[-1][sbinary[-1]>blur_thresh] = 1
            sbinary[-1][sbinary[-1]<=blur_thresh] = 0
        sbinary = np.asarray(sbinary)
        return sbinary



    def get_R_thresh(self, thresh=(0,255)):
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
        return sbinary
        
        
        
    def get_B_thresh(self, thresh=(0,255)):
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
        return sbinary       
           
        
        
    def get_G_thresh(self, thresh=(0,255)):
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
        return sbinary         
                   
        
        
    def get_H_thresh(self, thresh=(0,255)):
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
        return sbinary          
                           
        
        
    def get_L_thresh(self, thresh=(0,255)):
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
        return sbinary  
                   
        
        
    def get_S_thresh(self, thresh=(0,255)):
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
        return sbinary                    
                    
        
        
    def get_gray_thresh(self, thresh=(0,255)):
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



    def process_images(self, pass_grade_grad=0.35, pass_grade_color=0.59, blur_kernel=3, blur_thresh=135): 
        """
        processes images and creates binary processed images
        pass_grade: passing grade for pixel values give a scale of [0,1]
        """
    
        # Assign weights for the voting process for each binary contribution
        # of the gradient-baed inputs
        weight_arr_grad = [1.5, #canny 
                           1.0, #ast    
                           1.0, #mast   
                           1.5, #dst_x  
                           1.0] #dst_y  
        
        # Assign weights for the voting process for each binary contribution
        # of the color-baed inputs                               
        weight_arr_color = [1.2, #r channel
                            1.7, #g channel
                            0.8, #b channel
                            1.7, #h channel
                            1.0, #s channel
                            1.7, #l channel
                            1.0] #gray
  
        
        # Combine all gradient-based binary images together - weighted sum
        img_binary_grad = []
        img_binary_grad.append(weight_arr_grad[0]  * self.get_canny(thresh=(145,150)))
        img_binary_grad.append(weight_arr_grad[1]  * self.get_angle_sobel_thresh(sobel_kernel=3, thresh=(0.8, 1.2), blur_kernel=3, blur_thresh=0.8))
        img_binary_grad.append(weight_arr_grad[2]  * self.get_mag_sobel_thresh(thresh=(30,100)))
        img_binary_grad.append(weight_arr_grad[3]  * self.get_dir_sobel_thresh(orient='x',thresh=(35,110)))
        img_binary_grad.append(weight_arr_grad[4]  * self.get_dir_sobel_thresh(orient='y',thresh=(25,110)))
        img_binary_grad = np.asarray(img_binary_grad)
        img_binary_grad_sum = img_binary_grad.sum(axis=0)
        
        # Combine all color-based binary images together - weighted sum
        img_binary_color = []
        img_binary_color.append(weight_arr_color[0]  * self.get_R_thresh(thresh=(200,255)))
        img_binary_color.append(weight_arr_color[1]  * self.get_G_thresh(thresh=(160,255)))
        img_binary_color.append(weight_arr_color[2]  * self.get_B_thresh(thresh=(90,255)))
        img_binary_color.append(weight_arr_color[3]  * self.get_H_thresh(thresh=(20,50)))
        img_binary_color.append(weight_arr_color[4]  * self.get_S_thresh(thresh=(80,255)))
        img_binary_color.append(weight_arr_color[5]  * self.get_L_thresh(thresh=(130,255)))
        img_binary_color.append(weight_arr_color[6] * self.get_gray_thresh(thresh=(180,255)))
        img_binary_color = np.asarray(img_binary_color)
        img_binary_color_sum = img_binary_color.sum(axis=0)
        
        # creating a procssed binary image based on gradations   
        img_post_grad = []    
        for img_grad in img_binary_grad_sum: 
            max_pix_grad = sum(weight_arr_grad)
            img_post_grad.append(np.zeros_like(img_grad.astype('uint8')))    
            img_post_grad[-1][(img_grad/max_pix_grad >= pass_grade_grad)] = 255
            # apply gaussian blur to reduce isolated pixels           
            img_post_grad[-1] = cv2.GaussianBlur(img_post_grad[-1], (blur_kernel, blur_kernel), 0)
            img_post_grad[-1][img_post_grad[-1]>blur_thresh] = 255
            img_post_grad[-1][img_post_grad[-1]<=blur_thresh] = 0
        img_post_grad = np.asarray(img_post_grad)
        
        # creating a procssed binary image based on colors
        img_post_color = []    
        for img_color in img_binary_color_sum: 
            max_pix_color = sum(weight_arr_color)
            img_post_color.append(np.zeros_like(img_color.astype('uint8')))    
            img_post_color[-1][(img_color/max_pix_color >= pass_grade_color)] = 255
            # apply gaussian blur to reduce isolated pixels           
            img_post_color[-1] = cv2.GaussianBlur(img_post_color[-1], (blur_kernel, blur_kernel), 0)
            img_post_color[-1][img_post_color[-1]>blur_thresh] = 255
            img_post_color[-1][img_post_color[-1]<=blur_thresh] = 0
        img_post_color = np.asarray(img_post_color)
               
        # saving processed images to the iamge transform object
        img_result_list = []
        for img1, img2 in zip(img_post_grad, img_post_color):
            img_result_list.append(np.zeros_like(img1))
            img_result_list[-1][(img1>0) | (img2>0)]=255
        img_result_list = np.asarray(img_result_list)
            
        self.processed_images = np.copy(img_result_list)
        
        
        
    def detect_lanes(self, prev_left_pos=None, prev_right_pos=None, verbose=False):
        """
        detects the lane lines in the img and returns xxx
        prev_left_pos, and prev_right_pos are the previous position of the left and right lanes respectively.
        verbose mode is only used for debugging
        """
        
        results_list = []
        for img in self._birds_eye_processed:
            
            if verbose:
                fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(14,7.87))
                axes[0].imshow(img, cmap='gray')            
            
            # initializing local variables including copy of the image
            res_img = np.zeros_like(img)    # revised image with lane lines plotted on it
            color_res_img = np.dstack((res_img, res_img, res_img)) # colored resulting image 
            min_x_margin = 80                  # margin around the centre to look for lane line pixels 
            max_x_margin = 160
            x_margin_left = min_x_margin
            x_margin_right = min_x_margin
            margin_multiplier = 1.2
            num_horizontal_bands = 10       # number of hoirzontal bands used to trace the lane lines
            y_grid = np.linspace(0, img.shape[0], num_horizontal_bands, dtype='uint16') # horizontal band coordiantes
            y_grid = y_grid[::-1]           # reversing the order
            img_width = img.shape[1]        # width of the image
            img_height = img.shape[0]       # height of the image
            left_lane_yvals = []            # keeping track of all points for left lane
            left_lane_xvals = []            # keeping track of all points for left lane
            right_lane_yvals = []           # keeping track of all points for right lane
            right_lane_xvals = []           # keeping track of all points for right lane
            
            # finding left lane - initial attempt 
            if prev_left_pos is None:
                bottom_left_hist = np.sum(img[img_height//2:,:img_width//2],axis=0)
                bottom_left_hist_list = bottom_left_hist.tolist()
                left_max_index_list = [i for i, x in enumerate(bottom_left_hist_list) if x==bottom_left_hist.max()]
                left_pos = left_max_index_list[len(left_max_index_list)//2]
            else:
                left_pos = img.shape[1] // 2 - abs(int(prev_left_pos / self._xm_per_pix))
            left_pos_delta = 0
            
            # finding right lane - initial attempt
            if prev_right_pos is None:
                bottom_right_hist = np.sum(img[img_height//2:,img_width//2:],axis=0)
                bottom_right_hist_list = bottom_right_hist.tolist()
                right_max_index_list = [i for i, x in enumerate(bottom_right_hist_list) if x==bottom_right_hist.max()]
                right_pos = right_max_index_list[len(right_max_index_list)//2]+img_width//2
            else:
                right_pos = img.shape[1] // 2 + abs(int(prev_right_pos / self._xm_per_pix))
            right_pos_delta = 0
            
            # Trace the lane lines from bottom of the image upward and detect lane pixels
            for i in range(len(y_grid)-1):
                # assigning the rows        
                from_y = y_grid[i+1]
                to_y = y_grid[i]
                
                # updating the left box for searching for lane points
                box_left = img[from_y:to_y,max(left_pos-x_margin_left,0):min(left_pos+x_margin_left,img_width)]
                res_img[from_y:to_y,max(left_pos-x_margin_left,0):min(left_pos+x_margin_left,img_width)] = box_left
                box_left_hist = box_left.sum(axis=0)
                
                # adding coordinates of non-zero elements to the lists
                left_lane_yvals += (np.nonzero(box_left)[0]+from_y).tolist()
                left_lane_xvals += (np.nonzero(box_left)[1]+max(left_pos-x_margin_left,0)).tolist()
                
                # for visualization and debuggin purposes
                if verbose:
                    p1_x = max(left_pos-x_margin_left,0)
                    p1_y = from_y
                    p2_x = left_pos+x_margin_left
                    p2_y = to_y
                    cv2.rectangle(res_img, (p1_x,p1_y), (p2_x,p2_y), [255,0,0],2)
                
                # moving the position of the left box based on what was found
                if len(box_left_hist)!=0 and box_left_hist.max() >= 0.25*abs(to_y-from_y)*255:
                    box_left_hist_list = box_left_hist.tolist()
                    left_max_index_list = [i for i, x in enumerate(box_left_hist_list) if x==box_left_hist.max()]
                    left_pos_delta = left_max_index_list[len(left_max_index_list)//2]+max(left_pos-x_margin_left,0)-left_pos
                    left_pos = left_pos+left_pos_delta
                    x_margin_left = max(int(x_margin_left/margin_multiplier),min_x_margin)
                else:
                    left_pos = left_pos+left_pos_delta
                    x_margin_left = min(int(x_margin_left*margin_multiplier),max_x_margin)
                left_pos = min(left_pos,img_width)
                left_pos = max(left_pos, 0)
                    
                # updating the left box for searching for lane points
                box_right = img[from_y:to_y,max(right_pos-x_margin_right,0):min(right_pos+x_margin_right,img_width)]
                res_img[from_y:to_y,max(right_pos-x_margin_right,0):min(right_pos+x_margin_right,img_width)] = box_right
                box_right_hist = box_right.sum(axis=0)
                
                # adding coordinates of non-zero elements to the lists
                right_lane_yvals += (np.nonzero(box_right)[0]+from_y).tolist()
                right_lane_xvals += (np.nonzero(box_right)[1]+max(right_pos-x_margin_right,0)).tolist()
                
                # for visualization and debuggin purposes
                if verbose:
                    p1_x = max(right_pos-x_margin_right,0)
                    p1_y = from_y
                    p2_x = right_pos+x_margin_right
                    p2_y = to_y
                    cv2.rectangle(res_img, (p1_x,p1_y), (p2_x,p2_y), [255,0,0],2)
                
                # moving the position of the right box based on what was found
                if len(box_right_hist)!=0 and box_right_hist.max() >= 0.25*abs(to_y-from_y)*255:
                    box_right_hist_list = box_right_hist.tolist()
                    right_max_index_list = [i for i, x in enumerate(box_right_hist_list) if x==box_right_hist.max()]
                    right_pos_delta = right_max_index_list[len(right_max_index_list)//2]+max(right_pos-x_margin_right,0)-right_pos
                    right_pos = right_pos+right_pos_delta
                    x_margin_right = max(int(x_margin_right/margin_multiplier),min_x_margin)
                else:
                    right_pos = right_pos+right_pos_delta 
                    x_margin_right = min(int(x_margin_right*margin_multiplier),max_x_margin)
                right_pos = min(right_pos,img_width)
                right_pos = max(right_pos, 0) 
            
            # converting lane point x and y's to np array objects
            left_lane_yvals = np.asarray(left_lane_yvals)
            left_lane_xvals = np.asarray(left_lane_xvals)
            right_lane_yvals = np.asarray(right_lane_yvals)
            right_lane_xvals = np.asarray(right_lane_xvals)
            
            # calculating best fit polylines - degree=2    
            if len(left_lane_yvals)!=0:
                left_fit = np.polyfit(left_lane_yvals, left_lane_xvals, 2)
                left_fit_cr = np.polyfit(left_lane_yvals*self._ym_per_pix, left_lane_xvals*self._xm_per_pix, 2)
            else:
                left_fit = np.array([0,0,0], dtype='float')
                left_fit_cr = np.array([0,0,0], dtype='float')
            if len(right_lane_yvals)!=0:
                right_fit = np.polyfit(right_lane_yvals, right_lane_xvals, 2)
                right_fit_cr = np.polyfit(right_lane_yvals*self._ym_per_pix, right_lane_xvals*self._xm_per_pix, 2)
            else:
                right_fit = np.array([0,0,0], dtype='float')
                right_fit_cr = np.array([0,0,0], dtype='float')
            
            # calculating the fitted line X values for all y values in the image
            y_points = np.asarray(range(img_height), dtype=np.float32)
            left_fit_x = left_fit[0]*y_points**2 + left_fit[1]*y_points + left_fit[2]
            right_fit_x = right_fit[0]*y_points**2 + right_fit[1]*y_points + right_fit[2]    
            
            # Calculate curvature radii for left and right lane and the average thereof
            y_eval = img_height
            left_curverad = ((1 + (2*left_fit_cr[0]*y_eval + left_fit_cr[1])**2)**1.5) \
                                     /np.absolute(2*left_fit_cr[0])
            
            right_curverad = ((1 + (2*right_fit_cr[0]*y_eval + right_fit_cr[1])**2)**1.5) \
                                            /np.absolute(2*right_fit_cr[0])   
            
            # Calculate off-center location of the car assuming the centre of the image is the centre of teh car
            center_position = img_width//2
            left_lane_pos = center_position - left_fit_x[-1]
            right_lane_pos = center_position - right_fit_x[-1]
            off_center = np.mean((left_lane_pos, right_lane_pos))
            
            # converting ot m            
            left_lane_pos *= self._xm_per_pix
            right_lane_pos *= self._xm_per_pix
            off_center *= self._xm_per_pix
            
            # Recast the x and y points into usable format for cv2.fillPoly()
            pts_left = np.array([np.transpose(np.vstack([left_fit_x,y_points]))])
            pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fit_x,y_points])))])
            pts = np.hstack((pts_left, pts_right))

            # Draw the lane onto the warped blank image
            cv2.fillPoly(color_res_img, np.int_(pts), (0,255,0))   
            
            results = []
            results.append({})  # left lane resutls
            results.append({})  # right lane rseults
            # left lane
            results[0]['yvals'] = left_lane_yvals
            results[0]['xvals'] = left_lane_xvals
            results[0]['poly_fit'] = left_fit
            results[0]['fitted_xvals'] = (center_position-left_fit_x)*self._xm_per_pix
            results[0]['curve_rad'] = left_curverad
            results[0]['base_pos'] = left_lane_pos
            # right lane
            results[1]['yvals'] = right_lane_yvals
            results[1]['xvals'] = right_lane_xvals
            results[1]['poly_fit'] = right_fit
            results[1]['fitted_xvals'] = (center_position-right_fit_x)*self._xm_per_pix
            results[1]['curve_rad'] = right_curverad
            results[1]['base_pos'] = right_lane_pos
            # appending the results
            results_list.append(results)
                        
            # plotting tracing boxes and fitted lines on the raw processed image    
            if verbose: 
                axes[1].imshow(res_img, cmap='gray')    
                axes[1].plot(left_fit_x, y_points)
                axes[1].plot(right_fit_x, y_points)
                axes[1].set_ylim(img.shape[0],0)
                axes[1].set_xlim(0,img.shape[1])
            
            # plotting filled poly image as the final check.
            if verbose:
                axes[2].imshow(color_res_img)
            
        
        return results_list



    def plot_fitted_poly(self, fitted_poly_list, labels=None):
        
        self._processed_images = []     
        self._labels = labels
        
        for idx, img in enumerate(self._RGB):                        
            
            res_img = np.zeros_like(img[:,:,0])    # revised image with lane lines plotted on it
            color_res_img = np.dstack((res_img, res_img, res_img)) # colored resulting image 
            img_height = img.shape[0]       # height of the image
            img_width = img.shape[1]
            
            # calculating the y_points in vertical direction
            y_points = np.asarray(range(img_height), dtype=np.float32)
            
            left_fit = fitted_poly_list[idx][0]
            right_fit = fitted_poly_list[idx][1]
            left_fit_x = left_fit[0]*y_points**2 + left_fit[1]*y_points + left_fit[2]
            right_fit_x = right_fit[0]*y_points**2 + right_fit[1]*y_points + right_fit[2]   
            
            # Recast the x and y points into usable format for cv2.fillPoly()
            pts_left = np.array([np.transpose(np.vstack([left_fit_x,y_points]))])
            pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fit_x,y_points])))])
            pts = np.hstack((pts_left, pts_right))
            
            # Draw the lane onto the warped blank image
            cv2.fillPoly(color_res_img, np.int_(pts), (0,255,0))
            
            # warp perspective from birds_eye_view to the original perspective
            img_size = (img_width, img_height)
            src_viewport = np.float32([[self._p_0[0]],[self._p_1[0]],
                                       [self._p_2[0]],[self._p_3[0]]]) * np.float32(img_size)
            dst_viewport = np.float32([[self._q_0[0]],[self._q_1[0]],
                                       [self._q_2[0]],[self._q_3[0]]]) * np.float32(img_size)
            M_inv = cv2.getPerspectiveTransform(dst_viewport, src_viewport)
            color_res_img = cv2.warpPerspective(color_res_img, M_inv, img_size, flags=cv2.INTER_LINEAR)
            
            # writing labels on images
            if labels is not None:
                cv2.putText(color_res_img, labels[idx], (50,50), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (255,255,255))
            
            self._processed_images.append(cv2.addWeighted(img, 1, color_res_img, 0.3, 0))
        
        self._processed_images = np.asarray(self._processed_images)
            
            
        
        