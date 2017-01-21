# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import cv2
import glob

from moviepy.editor import VideoFileClip

from imagetransform import ImageTransform
from lanelines import LaneLine

# define global variables
left_lane = LaneLine()
right_lane = LaneLine()
cam_matrix = None
dist_matrix = None

from_computer_1 = False
if from_computer_1:
    # when working from computer 1
    cal_dir = "C:/Udacity Courses/Car-ND-Udacity/P4-Advanced-Lane-Lines/camera_cal/"
    tst_dir = "C:/Udacity Courses/Car-ND-Udacity/P4-Advanced-Lane-Lines/test_images/"
    work_dir = "C:/Udacity Courses/Car-ND-Udacity/P4-Advanced-Lane-Lines/"
else:
    # when working from computer 2
    cal_dir = "C:/Users/ali.khalili/Desktop/Car-ND/CarND-P4-Advanced-Lane-Lines/camera_cal/"
    tst_dir = "C:/Users/ali.khalili/Desktop/Car-ND/CarND-P4-Advanced-Lane-Lines/test_images/"
    work_dir = "C:/Users/ali.khalili/Desktop/Car-ND/CarND-P4-Advanced-Lane-Lines/"


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
    
    # load all test images
    f_names = glob.glob(tst_dir+'*.jpg')
    images = []
    for idx, fname in enumerate(f_names):
        img = cv2.imread(fname)
        images.append(img)
    images = np.asarray(images)
    
    return images



def replace_frame(frame_img):
    
    global left_lane
    global right_lane
    global cam_matrix
    global dist_matrix        
    
    img_trans_obj = ImageTransform(np.asarray([frame_img]), "", cam_matrix, dist_matrix, colorspec='RGB')
    img_trans_obj.process_images()
    img_trans_obj.to_birds_eye(original=True, processed=True)
    result = img_trans_obj.detect_lanes(verbose=False)
    
    # automatic check to see if lane lines are in fact detected:
    curve_ratio = max(result[0][0]['curve_rad'],result[0][1]['curve_rad']) / min(result[0][0]['curve_rad'],result[0][1]['curve_rad'])
    detected = (curve_ratio<=3.3)    
    
    # check to see that lanes are separated by the right amount of distance
    dist = result[0][0]['fitted_xvals']-result[0][1]['fitted_xvals']
    detected = detected and (dist.max()<=4.1) and (dist.min()>=1.1)
    
    # initialize left and right lane line objects
    left_lane.add_results(result[0][0], detected)
    right_lane.add_results(result[0][1], detected)
    
    poly_fit_list = [left_lane.poly_fit_average, right_lane.poly_fit_average]
    
    if left_lane.base_pos_average is not None and right_lane.base_pos_average is not None:
        # calculating the average base position and curvature radius
        base_pos_offset = np.mean((left_lane.base_pos_average, right_lane.base_pos_average))
        curvature_rad = np.mean((left_lane.curve_rad_average, right_lane.curve_rad_average))
        
        # set the appropriate text to be printed on the frame
        if base_pos_offset>0:
            label_text = 'Radius of Curvature = {:.1f} m - Vehicle is {:.2f} m {} of center'.format(curvature_rad, abs(base_pos_offset), 'right')
        elif base_pos_offset<0:
            label_text = 'Radius of Curvature = {:.1f} m - Vehicle is {:.2f} m {} of center'.format(curvature_rad, abs(base_pos_offset), 'left')
        else:
            label_text = 'Radius of Curvature = {:.1f} m - Vehicle is at center'.format(curvature_rad)
        
        img_trans_obj.plot_fitted_poly([poly_fit_list], [label_text])
    
        # return obj._processed_images_0
        return img_trans_obj.processed_images[0]
    else:
        return img_trans_obj.RGB[0]



def process_movie(fname):
    '''
    load movie + replace frames with processed images + save movie
    '''
    movie_clip = VideoFileClip(work_dir+fname)
    processed_clip = movie_clip.fl_image(replace_frame)
    processed_clip.write_videofile(work_dir+'AK_'+fname, audio=False, verbose=True, threads=6)
    return


def load_and_process_test_images():
    '''
    load and process all test images
    '''
    global cam_matrix
    global dist_matrix
    
    # loading test images, if images are not provided   
    images = load_test_images()
    img_trans_obj = ImageTransform(images, "", cam_matrix, dist_matrix, colorspec='BGR')
    img_trans_obj.process_images()
    
    img_trans_obj.to_birds_eye(original=True, processed=True)
    results = img_trans_obj.detect_lanes(verbose=True) 
    
    poly_fits = []
    labels = []    
    for result in results:
        
        # automatic check to see if lane lines are in fact detected:
        # check to see that curvatures for left and right lanes are not far apart
        curve_ratio = max(result[0]['curve_rad'],result[1]['curve_rad']) / min(result[0]['curve_rad'],result[1]['curve_rad'])
        detected = (curve_ratio<=3.3)    
        # check to see that lanes are separated by the right amount of distance
        dist = result[0]['fitted_xvals']-result[1]['fitted_xvals']
        detected = detected and (dist.max()<=4.1) and (dist.min()>=1.1)      
        print(curve_ratio, dist.max(), dist.min(), detected)
                
        poly_fits.append([result[0]['poly_fit'],result[1]['poly_fit']])
        base_pos_offset = np.mean((result[0]['base_pos'],result[1]['base_pos']))
        curvature_rad = np.mean((result[0]['curve_rad'],result[1]['curve_rad']))
        label_text = 'Radius of Curvature = {:.1f} m - Vehicle is at center'.format(curvature_rad)
        if base_pos_offset>0:
            label_text = 'Radius of Curvature = {:.1f} m - Vehicle is {:.2f} m {} of center'.format(curvature_rad, abs(base_pos_offset), 'right')
        elif base_pos_offset<0:
            label_text = 'Radius of Curvature = {:.1f} m - Vehicle is {:.2f} m {} of center'.format(curvature_rad, abs(base_pos_offset), 'left')
        label_text += ' - detected = ' + str(detected)
        labels.append(label_text)
        
    img_trans_obj.plot_fitted_poly(poly_fits, labels)
    img_trans_obj.set_labels(labels)
    img_trans_obj.draw_viewport(True, False, True, True)
    img_trans_obj.plot_comparison(birds_eye=False)
    
    return    
        
    

def main():
    
    global left_lane
    global right_lane
    global cam_matrix
    global dist_matrix
    # calibrate camera
    cam_matrix, dist_matrix = calibrate_camera_from_path(cal_dir, 9, 6)  
    
    process_movie('challenge_video.mp4')
    #load_and_process_test_images()
    
    return
    
        


if __name__=='__main__':
    main()   
