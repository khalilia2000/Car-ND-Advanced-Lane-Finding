# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from matplotlib import gridspec
from imagetransform import ImageTransform

from_computer_1 = False
if from_computer_1:
    # when working from computer 1
    cal_dir = "C:/Udacity Courses/Car-ND-Udacity/P4-Advanced-Lane-Lines/camera_cal/"
    tst_dir = "C:/Udacity Courses/Car-ND-Udacity/P4-Advanced-Lane-Lines/test_images/"
else:
    # when working from computer 2
    cal_dir = "C:/Users/ali.khalili/Desktop/Car-ND/CarND-P4-Advanced-Lane-Lines/camera_cal/"
    tst_dir = "C:/Users/ali.khalili/Desktop/Car-ND/CarND-P4-Advanced-Lane-Lines/test_images/"


def calibrate_camera_from_path(cal_path, nx, ny, save_with_corners=False, save_undistort=False):
    '''
    returns camera matrix and distortion matrix based on all the chessboard pictures located in the path
    cal_path: is the path to the calibration files. All calibration files are assumed to have 
              'calibration*.jpg' format
    nx: number of chessboard corners in x direction // 9 in our example
    ny: number of chessboard corners in y direction // 6 in our example
    save_with_corners:  if True and all coreners are found then a copy of the calibration images with 
                        marked corners will be saved as corners_found*.jpg
    save_undistorted: if True a copy of the undistorted image will be saved as undistort*.jpg
    '''
    
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((nx*ny,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)
    
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.
    
    # Make a list of calibration images
    images = glob.glob(cal_path+'calibration*.jpg')
                
    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    
        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
      
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
            
            if save_with_corners:
                findex_str = images[idx][images[idx].find('calibration')+11:images[idx].find('.jpg')]
                write_name = 'corners_found'+findex_str+'.jpg'
                cv2.imwrite(cal_path+write_name, img)
    
    
    img_size = (img.shape[1],img.shape[0])
    
    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)  

    # saves the undistorted calibration files if the flag is True    
    if save_undistort:  
        for idx, fname in enumerate(images):
            img = cv2.imread(fname)
            dst = cv2.undistort(img, mtx, dist, None, mtx)
            findex_str = images[idx][images[idx].find('calibration')+11:images[idx].find('.jpg')]
            write_name = 'undistort'+findex_str+'.jpg'
            cv2.imwrite(cal_path+write_name,dst)
        
    return mtx, dist
  
  

def plot_image(ax_list, grid_fig, grid_index, img, label="", cmap=None):
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


  
def plot_grid(images, labels, cmaps):
    '''
    plots a random grid of images to verify
    labels: array of labels with the same rows and columns of the grid
    cmap: cmap
    '''
    #
    n_rows = labels.shape[0]
    n_cols = labels.shape[1]
    # creating the grid space
    hspace = 0.2    # distance between images vertically
    wspace = 0.01   # distance between images horizontally
    g_fig = gridspec.GridSpec(n_rows,n_cols) 
    g_fig.update(wspace=wspace, hspace=hspace)
    
    # setting up the figure
    size_factor = 3.0
    aspect_ratio = 1.777
    fig_w_size = n_cols*size_factor*aspect_ratio+(n_cols-1)*wspace
    fig_h_size = n_rows*size_factor+(n_rows-1)*hspace
    plt.figure(figsize=(fig_w_size,fig_h_size))
    
    ax_list = []
    for i in range(n_rows*n_cols):
        ax_list = plot_image(ax_list, g_fig, i, images[i], labels.ravel()[i], cmap=cmaps.ravel()[i])    
 


def load_test_images():
    # calibrate camera
    cam_matrix, dist_matrix = calibrate_camera_from_path(cal_dir, 9, 6)    
    # load all test images
    f_names = glob.glob(tst_dir+'*.jpg')
    images = []
    for idx, fname in enumerate(f_names):
        img = cv2.imread(fname)
        images.append(img)
    images = np.asarray(images)
    # create image transform object
    img_trns = ImageTransform(images, cam_matrix, dist_matrix, None)
    
    return img_trns
    

def process_images(images=None):
    if not images:    
        img_trans_obj = load_test_images()
    img_trans_obj.to_RGB()
    
    img_pre = img_trans_obj.RGB
    
    # binary image containing directional sobel with thresholds    
    img_ast = img_trans_obj.get_angle_sobel_thresh(thresh=(0.5,1.2))
    # binary image containing magnitude of the gradient    
    img_mst = img_trans_obj.get_mag_sobel_thresh(thresh=(30,110))
    # binary image containing abs magnitude of the x sobel gradient
    img_dstx = img_trans_obj.get_dir_sobel_thresh(orient='x',thresh=(20,110))
    # binary image containing abs magnitude of the x sobel gradient
    img_dsty = img_trans_obj.get_dir_sobel_thresh(orient='y',thresh=(30,110))
    # binary image containing the R channel  
    img_R = img_trans_obj.get_R_thresh(thresh=(200,255))
    # binary image containing the G channel  
    img_G = img_trans_obj.get_G_thresh(thresh=(180,255))
    # binary image containing the H channel  
    img_H = img_trans_obj.get_H_thresh(thresh=(15,100))
    # binary image containing the S channel  
    img_S = img_trans_obj.get_S_thresh(thresh=(80,225))
    # binary image containing the S channel  
    img_gr = img_trans_obj.get_gray_thresh(thresh=(180,225))
    
    # creating the final revised images 
    img_post = []
    w_R = 1.0
    w_G = 1.0
    w_H = 1.0
    w_S = 1.5
    w_gr = 0.5
    for img_R, img_G, img_H, img_S, img_gr in zip(img_R, img_G, img_H, img_S, img_gr):
        img_post.append(np.zeros_like(img_R))
        img_tmp = img_R*w_R+img_G*w_G+img_H*w_H+img_S*w_S+img_gr*w_gr
        img_post[-1][img_tmp>=3.0] = 1 
    img_post = np.asarray(img_post)
    
#    img_post = []
#    for img1, img2, img3 in zip(img_ast, img_mst, img_dstx, img_dsty):
#        img_post.append(np.zeros_like(img1))
#        img_post[-1][((img2 == 1) | (img3 == 1) | (img4 == 1)) & (img1 == 1)] = 1 
#    img_post = np.asarray(img_post)
    

    
    # combining original and processed images for plotting
    i_to_plot = []
    for img1, img2 in zip(img_pre, img_post):
        i_to_plot.append(img1)
        i_to_plot.append(img2)
    
    labels = np.empty(shape=(img_pre.shape[0],2), dtype=str)
    cmaps = ['gray' for i in range(len(img_pre)+len(img_post))]
    cmaps = np.asarray(cmaps)
    plot_grid(i_to_plot, labels, cmaps)
    
    
    

def main():
    pass
    

if __name__=='__main__':
    main()   
