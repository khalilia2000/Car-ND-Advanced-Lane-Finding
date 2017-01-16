# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

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
    
    

def process_images(images=None, pass_grade=6):
    
    # loading test images, if images are not provided    
    if not images:    
        img_trans_obj = load_test_images()

    # weights for the voting process for each binary contribution
    weight_arr = [1.0, #ast
                  1.0, #mast
                  1.0, #dst_x 
                  1.0, #dst_y
                  1.0, #r channel
                  1.0, #g channel
                  1.0, #b channel
                  1.0, #h channel
                  1.0, #s channel
                  1.0, #l channel
                  1.0] #gray
    
    # adding all binary images together
    img_binary = []
    img_binary.append(weight_arr[0]  * img_trans_obj.get_angle_sobel_thresh(thresh=(0.7,1.2)))
    img_binary.append(weight_arr[1]  * img_trans_obj.get_mag_sobel_thresh(thresh=(30,110)))
    img_binary.append(weight_arr[2]  * img_trans_obj.get_dir_sobel_thresh(orient='x',thresh=(20,110)))
    img_binary.append(weight_arr[3]  * img_trans_obj.get_dir_sobel_thresh(orient='y',thresh=(30,110)))
    img_binary.append(weight_arr[4]  * img_trans_obj.get_R_thresh(thresh=(115,255)))
    img_binary.append(weight_arr[5]  * img_trans_obj.get_G_thresh(thresh=(115,255)))
    img_binary.append(weight_arr[6]  * img_trans_obj.get_B_thresh(thresh=(115,255)))
    img_binary.append(weight_arr[7]  * img_trans_obj.get_H_thresh(thresh=(15,100)))
    img_binary.append(weight_arr[8]  * img_trans_obj.get_S_thresh(thresh=(80,225)))
    img_binary.append(weight_arr[9]  * img_trans_obj.get_L_thresh(thresh=(110,225)))
    img_binary.append(weight_arr[10] * img_trans_obj.get_gray_thresh(thresh=(180,225)))
    img_binary = np.asarray(img_binary)
    img_binary_sum = img_binary.sum(axis=0)
    
    # creating a procssed binary image    
    img_post = []    
    for img in img_binary_sum:  
        img_post.append(np.zeros_like(img.astype('uint8')))    
        img_post[-1][img >= pass_grade] = 255
    img_post = np.asarray(img_post)
           
    # saving processed images to the iamge transform object
    img_trans_obj.processed_images = np.copy(img_post)
    
    # drawing viewports 
    img_trans_obj.draw_viewport(original=True, processed=True, src_viewport=True, dst_viewport=True)
    
    
    return img_trans_obj



def main():
    img_trans_obj = process_images()
    img_trans_obj.plot_comparison(birds_eye=False)
    

if __name__=='__main__':
    main()   
