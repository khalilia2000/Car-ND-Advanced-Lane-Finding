# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

cal_dir = "C:/Udacity Courses/Car-ND-Udacity/P4-Advanced-Lane-Lines/camera_cal/"


def calibrate_camera_from_path(cal_path, nx, ny, save_with_corners=True, save_undistort=True):
    """
    returns camera matrix and distortion matrix based on all the chessboard pictures located in the path
    cal_path: is the path to the calibration files. All calibration files are assumed to have 
              'calibration*.jpg' format
    nx: number of chessboard corners in x direction // 9 in our example
    ny: number of chessboard corners in y direction // 6 in our example
    save_with_corners:  if True and all coreners are found then a copy of the calibration images with 
                        marked corners will be saved as corners_found*.jpg
    save_undistorted: if True a copy of the undistorted image will be saved as undistort*.jpg
    """
    
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
   

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    '''
    Calculates directional gradient, and applies threshold. returns a binary image
    img: original image
    orient: either 'x' or 'y', and determines the direction of sobel operator
    sobel_kernel: the size of sobel_kernel
    thresh: threshold for binary image creation
    '''
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        sobelg = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    else:
        sobelg = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    abs_sobel = np.absolute(sobelg)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    
    sbinary = np.zeros_like(scaled_sobel)
    sbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    
    binary_output = sbinary
    
    return binary_output
    
    

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    '''
    Calculates the magnitude of sobel gradient, and applies threshold. returns a binary image
    img: original image
    sobel_kernel: the size of sobel_kernel
    mag_thresh: threshold for binary image creation
    '''

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    sobelg = ((sobelx**2+sobely**2)**0.5)
    abs_sobel = np.absolute(sobelg)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    
    sbinary = np.zeros_like(scaled_sobel)
    sbinary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    
    binary_output = sbinary
    
    #binary_output = np.copy(img) # Remove this line
    return binary_output
    
    

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    '''
    Calculates the direction of sobel gradient, and applies threshold. returns a binary image
    img: original image
    sobel_kernel: the size of sobel_kernel
    mag_thresh: threshold for binary image creation
    '''

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    sobelg = ((sobelx**2+sobely**2)**0.5)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    abs_sobelg = np.absolute(sobelg)
    
    dir_sobel = np.arctan2(abs_sobely, abs_sobelx)
    scaled_sobel = np.uint8(255*abs_sobelg/np.max(abs_sobelg))
    
    sbinary = np.zeros_like(scaled_sobel)
    sbinary[(dir_sobel >= thresh[0]) & (dir_sobel <= thresh[1])] = 1
    
    binary_output = sbinary
    
    #binary_output = np.copy(img) # Remove this line
    return binary_output



def main():
  pass


if __name__=='__main__':
    main()   
