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
    
    

def process_images(images=None, pass_grade=0.57):
    """
    processes images
    images: images to be processed. if None then test images will be used
    pass_grade: passing grade for pixel values give a scale of [0,1]
    """
    # loading test images, if images are not provided    
    if not images:    
        img_trans_obj = load_test_images()

    # weights for the voting process for each binary contribution
    weight_arr = [0.0, #ast             
                  1.5, #mast            
                  1.5, #dst_x           
                  0.5, #dst_y           
                  1.3, #r channel       
                  1.3, #g channel       
                  0.5, #b channel       
                  0.6, #h channel
                  0.9, #s channel
                  1.0, #l channel
                  0.9] #gray
    
    # adding all binary images together
    img_binary = []
    img_binary.append(weight_arr[0]  * img_trans_obj.get_angle_sobel_thresh(thresh=(0.8,1.2)))
    img_binary.append(weight_arr[1]  * img_trans_obj.get_mag_sobel_thresh(thresh=(30,100)))
    img_binary.append(weight_arr[2]  * img_trans_obj.get_dir_sobel_thresh(orient='x',thresh=(20,110)))
    img_binary.append(weight_arr[3]  * img_trans_obj.get_dir_sobel_thresh(orient='y',thresh=(30,110)))
    img_binary.append(weight_arr[4]  * img_trans_obj.get_R_thresh(thresh=(185,255)))
    img_binary.append(weight_arr[5]  * img_trans_obj.get_G_thresh(thresh=(130,255)))
    img_binary.append(weight_arr[6]  * img_trans_obj.get_B_thresh(thresh=(90,255)))
    img_binary.append(weight_arr[7]  * img_trans_obj.get_H_thresh(thresh=(15,120)))
    img_binary.append(weight_arr[8]  * img_trans_obj.get_S_thresh(thresh=(90,255)))
    img_binary.append(weight_arr[9]  * img_trans_obj.get_L_thresh(thresh=(110,255)))
    img_binary.append(weight_arr[10] * img_trans_obj.get_gray_thresh(thresh=(180,255)))
    img_binary = np.asarray(img_binary)
    img_binary_sum = img_binary.sum(axis=0)
    
    # creating a procssed binary image    
    img_post = []    
    for img in img_binary_sum: 
        max_pix = img.max()
        img_post.append(np.zeros_like(img.astype('uint8')))    
        img_post[-1][img/max_pix >= pass_grade] = 255
    img_post = np.asarray(img_post)
           
    # saving processed images to the iamge transform object
    img_trans_obj.processed_images = np.copy(img_post)
    
    return img_trans_obj



def detect_lanes(img):
    
    #plt.imshow(img, cmap='gray')
    
    res_img = np.zeros_like(img)
    
    x_margin = 100 # margin around the centre to look for lane line pixels    
    y_grid = np.linspace(0, img.shape[0], 9, dtype='uint16')
    print(img.shape, y_grid)
    # finding left lane
    bottom_left_hist = np.sum(img[img.shape[0]//2:,:img.shape[1]//2],axis=0)
    bottom_left_hist_list = bottom_left_hist.tolist()
    left_max_index_list = [i for i, x in enumerate(bottom_left_hist_list) if x==bottom_left_hist.max()]
    left_pos = left_max_index_list[len(left_max_index_list)//2]
    # finding right lane
    bottom_right_hist = np.sum(img[img.shape[0]//2:,img.shape[1]//2:],axis=0)
    right_pos = bottom_right_hist.tolist().index(bottom_right_hist.max())+int(img.shape[1]/2)
    for i in range(len(y_grid)-1):
        from_y = y_grid[i]
        to_y = y_grid[i+1]
        print(i, from_y, to_y)
        #
        box_left = img[from_y:to_y,left_pos-x_margin:left_pos+x_margin]
        res_img[from_y:to_y,left_pos-x_margin:left_pos+x_margin] = box_left
        box_left_hist = box_left.sum(axis=0)
        plt.plot(box_left_hist, label=str(i))
        if box_left_hist.max() >= 0.5*abs(to_y-from_y):
            print('left_pos=',left_pos, ' - found')
            box_left_hist_list = box_left_hist.tolist()
            left_max_index_list = [i for i, x in enumerate(box_left_hist_list) if x==box_left_hist.max()]
            left_pos = left_max_index_list[len(left_max_index_list)//2]+left_pos-x_margin
        else:
            print('left_pos=',left_pos, ' - not found')
        #
        print('right_pos=',right_pos)
        box_right = img[from_y:to_y,right_pos-x_margin:right_pos+x_margin]
        res_img[from_y:to_y,right_pos-x_margin:right_pos+x_margin] = box_right
        box_right_hist = box_right.sum(axis=0)
        #print(box_right_hist)
        if box_right_hist.max() >= 0.5*abs(to_y-from_y):
            right_pos = box_right_hist.tolist().index(box_right_hist.max())+right_pos-x_margin
        
        plt.legend()
    #plt.imshow(res_img, cmap='gray')



def main():
    img_trans_obj = process_images()
    # drawing viewports 
    #img_trans_obj.draw_viewport(original=True, processed=True, src_viewport=True, dst_viewport=True)
    #img_trans_obj.plot_comparison(birds_eye=True)
    img_trans_obj.to_birds_eye(original=False, processed=True)
    detect_lanes(img_trans_obj.birds_eye_processed[4])
    

if __name__=='__main__':
    main()   
