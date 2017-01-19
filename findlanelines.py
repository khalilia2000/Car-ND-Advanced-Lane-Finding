# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import cv2
import glob

import matplotlib.pyplot as plt
import time

from imagetransform import ImageTransform
from lanelines import LaneLine

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
    img_trns = ImageTransform(images, f_names, cam_matrix, dist_matrix, None)
    
    return img_trns



def main():
    
    # loading test images, if images are not provided    
    img_trans_obj = load_test_images()
    img_trans_obj.process_images()
    
    # drawing viewports 
    #img_trans_obj.draw_viewport(original=True, processed=True, src_viewport=True, dst_viewport=True)
    
    img_trans_obj.to_birds_eye(original=True, processed=True)
    results = img_trans_obj.detect_lanes()

    poly_fits = []
    labels = []    
    for result in results:
        poly_fits.append([result[0]['poly_fit'],result[1]['poly_fit']])
        base_pos_offset = np.mean((result[0]['base_pos'],result[1]['base_pos']))
        curvature_rad = np.mean((result[0]['curve_rad'],result[1]['curve_rad']))
        if base_pos_offset>0:
            label_text = 'Radius of Curvature = {:.1f} m - Vehicle is {:.2f} m {} of center'.format(curvature_rad, abs(base_pos_offset), 'right')
        elif base_pos_offset<0:
            label_text = 'Radius of Curvature = {:.1f} m - Vehicle is {:.2f} m {} of center'.format(curvature_rad, abs(base_pos_offset), 'left')
        else:
            label_text = 'Radius of Curvature = {:.1f} m - Vehicle is at center'.format(curvature_rad)
        labels.append(label_text)
        
    img_trans_obj.plot_fitted_poly(poly_fits, labels)
    #img_trans_obj.draw_viewport(True, True, True, True)
    img_trans_obj.plot_comparison(birds_eye=False)
    
    


if __name__=='__main__':
    main()   
