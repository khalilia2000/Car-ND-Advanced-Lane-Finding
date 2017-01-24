# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 16:31:14 2017

@author: Ali.Khalili
"""

import numpy as np

# Define a class to receive the characteristics of each line detection
class LaneLine(object):
    
    
    def __init__(self):
        
        # variables keepting track of lane detection        
        self._detected = False              # was the line detected in the last iteration?
        self._num_undetected = 0            # number of consecutive undetected iterations
        self._num_delta_iters = None        # number of the iterations between the last two successful detections
        
        # fitted x values of the polynomial fits        
        self._fitted_xvals_list = []        # x values of the last n fits of the line
        self._fitted_xvals_average = None   # average x values of the fitted line over the last n iterations
        
        # polynomial fit coefficients        
        self._poly_fit_list = []            # polynomial coefficients for the last n fits of the line
        self._poly_fit_average = None       # polynomial coefficients averaged over the last n-1 fits of the line
        self._poly_fit_current = None       # polynomial coefficients for the most recent fit
        self._poly_fit_diffs = np.array([0,0,0], dtype='float')  # difference of values between average values between consecutive iterations
        
        # radius of curvature of the line in some units
        self._curve_rad_list = []           # last n curvature radii of successful detections of lane
        self._curve_rad_average = None      # average curvature radius of last n-1 detections of lane
        self._curve_rad_current = None      # most recent curvature radius of lane that is detected
        self._curve_rad_diff = 0            # difference of values between average values between consecutive iterations
        # distance in meters of vehicle center from the line
        self._base_pos_list = []            # last n position values of successful detections of lane
        self._base_pos_average = None       # average position of last n-1 detections of lane
        self._base_pos_current = None       # most recent position of lane that is detected
        self._base_pos_diff = 0             # difference of values between average values between consecutive iterations
        
        
        self._xvals = None                  #x values for detected line pixels
        self._yvals = None                  #y values for detected line pixels
        
        
        self._num_iter = 4                  #number of interations to track
        
        
    
    @property
    def detected(self):
        return self._detected
    
    
    @detected.setter
    def detected(self, value):
        self._detected = value
    
    
    @property
    def poly_fit_average(self):
        return self._poly_fit_average
        
    
    @property
    def poly_fit_current(self):
        return self._poly_fit_current

    
    @property
    def base_pos_average(self):
        return self._base_pos_average
    
    
    @property
    def base_pos_current(self):
        return self._base_pos_current
    
    
    @property
    def curve_rad_average(self):
        return self._curve_rad_average
    
    
    @property
    def curve_rad_current(self):
        return self._curve_rad_current
        
            
    def add_results(self, result, detected):
        
        # check that the result makes sense
        
        if detected is not None:
            self._detected = detected
            # check the position of the lane from the previous best location
            if self.get_best_pos() is not None:
                self._detected = self._detected and abs(result['base_pos']-self.get_best_pos())<=0.5
            
            # check to see that curvature is not far apart from previous curvatures
            if self.get_best_curve_rad() is not None:
                self._detected = self._detected and abs(1/result['curve_rad']-1/self.get_best_curve_rad())<=0.0006
            
            # check to see that poly_fit_0 and poly_fit_1 are not far apart from previous detection
            if self.get_best_poly_fit() is not None:
                self._detected = self._detected and abs(result['poly_fit'][0]-self.get_best_poly_fit()[0])<=0.0006
                self._detected = self._detected and abs(result['poly_fit'][1]-self.get_best_poly_fit()[1])<=0.5
        else:
            self._detected = False
            
        # add the results to the object if good quality is established
        if self._detected:
            
            # reset num_undetected
            self._num_delta_iters = self._num_undetected + 1
            self._num_undetected = 0            
            
            # update xfitted values
            self._fitted_xvals_list.append(result['fitted_xvals'])
            if len(self._fitted_xvals_list)>self._num_iter:                
                self._fitted_xvals_list.pop(0)
            if len(self._fitted_xvals_list) >= 2:
                self._fitted_xvals_average = np.mean(self._fitted_xvals_list[1:], axis=0) # averages n-1 elements taht are most recent
                        
            # update poly fits
            self._poly_fit_current = result['poly_fit']
            self._poly_fit_list.append(self._poly_fit_current) 
            if len(self._poly_fit_list)>self._num_iter:
                self._poly_fit_list.pop(0)                
            if len(self._poly_fit_list) >= 2:
                self._poly_fit_average = np.mean(self._poly_fit_list[1:], axis=0) # averages n-1 elements taht are most recent
                self._poly_fit_diffs = np.mean(self._poly_fit_list[1:], axis=0)-np.mean(self._poly_fit_list[:-1], axis=0)
            
            # update radius of curvature
            self._curve_rad_current = result['curve_rad']
            self._curve_rad_list.append(self._curve_rad_current)
            if len(self._curve_rad_list)>=self._num_iter:
                self._curve_rad_list.pop(0)
            if len(self._curve_rad_list) >= 2:
                self._curve_rad_average = np.mean(self._curve_rad_list[1:], axis=0)
                self._curve_rad_diff = np.mean(self._curve_rad_list[1:], axis=0)-np.mean(self._curve_rad_list[:-1], axis=0)
            
            # update base position of lane lines
            self._base_pos_current = result['base_pos']
            self._base_pos_list.append(self._base_pos_current)
            if len(self._base_pos_list)>=self._num_iter:
                self._base_pos_list.pop(0)
            if len(self._base_pos_list) >= 2:
                self._base_pos_average = np.mean(self._base_pos_list[1:], axis=0)
                self._base_pos_diff = np.mean(self._base_pos_list[1:], axis=0)-np.mean(self._base_pos_list[:-1], axis=0)
            
            # set all x and y pixel points
            self._xvals = result['xvals']
            self._yvals = result['yvals']
    
        else:
            
            self._num_undetected += 1
            # if number of undetected iterations reaches an upper limit, resets the data
            if self._num_undetected > 2*self._num_iter:
                self.__init__()
    

    def get_best_pos(self):        
        best_pos = None
        if self._detected:
            if self._base_pos_average is not None:
                best_pos = self._base_pos_average
            else:
                best_pos = self._base_pos_current
        else:
            if (self._base_pos_average is not None) and (self._base_pos_diff is not None):
                # extrapolate the position based on the direction / speed of the moving car
                best_pos = self._base_pos_average + self._base_pos_diff * self._num_undetected / self._num_delta_iters
        return best_pos
        
        
    def get_best_curve_rad(self):        
        best_curve = None
        if self._detected:
            if self._curve_rad_average is not None:
                best_curve = self._curve_rad_average
            else:
                best_curve = self._curve_rad_current
        else:
            if (self._curve_rad_average is not None) and (self._curve_rad_diff is not None):
                # extrapolate the curve_rad based on the direction / speed of the moving car
                best_curve = self._curve_rad_average + self._curve_rad_diff * self._num_undetected / self._num_delta_iters
        return best_curve   
        
        
    def get_best_poly_fit(self):        
        best_poly = None
        if self._detected:
            if self._poly_fit_average is not None:
                best_poly = self._poly_fit_average
            else: 
                best_poly = self._poly_fit_current
        else:
            if (self._poly_fit_average is not None):  # note that poly_fit_diffs will never be None
                # extrapolate the poly_fit based on the direction / speed of the moving car
                best_poly = self._poly_fit_average + self._poly_fit_diffs * self._num_undetected / self._num_delta_iters  
        return best_poly