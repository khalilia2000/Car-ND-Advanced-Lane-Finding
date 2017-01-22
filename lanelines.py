# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 16:31:14 2017

@author: Ali.Khalili
"""

import numpy as np

# Define a class to receive the characteristics of each line detection
class LaneLine(object):
    
    
    def __init__(self):
        
        # was the line detected in the last iteration?
        self._detected = True  
        # number of consecutive undetected iterations
        self._num_undetected = 0
        # number of the iterations between the last two successful detections
        self._num_delta_iters = None
        
        # x values of the last n fits of the line
        self._fitted_xvals_list = [] 
        #average x values of the fitted line over the last n iterations
        self._fitted_xvals_average = None     
        
        #polynomial coefficients for the last n fits of the line
        self._poly_fit_list = []  
        #polynomial coefficients averaged over the last n fits of the line
        self._poly_fit_average = None  
        #polynomial coefficients for the most recent fit
        self._poly_fit_current = None  
        #difference in fit coefficients between last and new fits
        self._poly_fit_diffs = np.array([0,0,0], dtype='float') 
        
        #radius of curvature of the line in some units
        self._curve_rad_list = [] 
        self._curve_rad_average = None
        self._curve_rad_current = None
        self._curve_rad_diff = 0
        #distance in meters of vehicle center from the line
        self._base_pos_list = []
        self._base_pos_average = None 
        self._base_pos_current = None 
        self._base_pos_diff = 0
        
        #x values for detected line pixels
        self._xvals = None  
        #y values for detected line pixels
        self._yvals = None
        
        #number of interations to track
        self._num_iter = 3
        
        #keep track of ratios
        self._ratio1_list = []
        self._ratio2_list = []
        
    
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
        
        self._detected = detected
        
        # set lane detection flag       
        # check to see that curvature is not far apart from previous curvatures
        ratio1 = 0
        if self._curve_rad_average is not None:
            ratio1 = max(result['curve_rad'],self.get_best_curve_rad()) / min(result['curve_rad'],self.get_best_curve_rad())
            self._detected = self._detected and (ratio1 <= 2.2)
        self._ratio1_list.append(ratio1)
        
        # check to see that lanes are roughply parallel to the previous find
        ratio2 = 0
        if self._base_pos_average is not None:
            ratio2 = max(result['base_pos'],self.get_best_pos()) / min(result['base_pos'],self.get_best_pos())
            self._detected = self._detected and (ratio2.max()<=1.4)
        self._ratio2_list.append(ratio1)
        
        #self._detected = detected
            
        # add the results to the object if good quality is established
        if self._detected:
            
            # reset num_undetected
            self._num_delta_iters = self._num_undetected + 1
            self._num_undetected = 0            
            
            # update xfitted values
            self._fitted_xvals_list.append(result['fitted_xvals'])
            if len(self._fitted_xvals_list)>self._num_iter:                
                self._fitted_xvals_list.pop(0)
            self._fitted_xvals_average = np.mean(self._fitted_xvals_list, axis=0)
                        
            # update poly fits
            self._poly_fit_current = result['poly_fit']
            self._poly_fit_list.append(self._poly_fit_current) 
            if len(self._poly_fit_list)>self._num_iter:
                self._poly_fit_list.pop(0)                
            self._poly_fit_average = np.mean(self._poly_fit_list, axis=0)
            if len(self._poly_fit_list) >= 2:
                self._poly_fit_diffs = self._poly_fit_list[-1]-self._poly_fit_list[-2]
            
            # update radius of curvature
            self._curve_rad_current = result['curve_rad']
            self._curve_rad_list.append(self._curve_rad_current)
            if len(self._curve_rad_list)>=self._num_iter:
                self._curve_rad_list.pop(0)
            self._curve_rad_average = np.mean(self._curve_rad_list, axis=0)
            if len(self._curve_rad_list) >= 2:
                self._curve_rad_diff = self._curve_rad_list[-1]-self._curve_rad_list[-2]
            
            # update base position of lane lines
            self._base_pos_current = result['base_pos']
            self._base_pos_list.append(self._base_pos_current)
            if len(self._base_pos_list)>=self._num_iter:
                self._base_pos_list.pop(0)
            self._base_pos_average = np.mean(self._base_pos_list, axis=0)
            if len(self._base_pos_list) >= 2:
                self._base_pos_diff = self._base_pos_list[-1]-self._base_pos_list[-2]
            
            # set all x and y pixel points
            self._xvals = result['xvals']
            self._yvals = result['yvals']
    
        else:
            
            self._num_undetected += 1
            # if number of undetected iterations reaches an upper limit, resets the data
            if self._num_undetected > self._num_iter*2:
                self.__init__
    

    def get_best_pos(self):        
        best_pos = None
        if self._detected:
            best_pos = self._base_pos_current
        elif self._base_pos_average is not None:
            best_pos = self._base_pos_average
        return best_pos
        
        
    def get_best_curve_rad(self):        
        best_curve = None
        if self._detected:
            best_curve = self._curve_rad_current
        elif self._curve_rad_average is not None:
            best_curve = self._curve_rad_average
        return best_curve   
        
        
    def get_best_poly_fit(self):        
        best_poly = None
        if self.poly_fit_average is not None:
            best_poly = self.poly_fit_average
        elif self._detected:
            best_poly = self.poly_fit_current
        return best_poly