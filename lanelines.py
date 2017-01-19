# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 16:31:14 2017

@author: Ali.Khalili
"""

import numpy as np

# Define a class to receive the characteristics of each line detection
class LaneLine():
    
    
    def __init__(self):
        
        # was the line detected in the last iteration?
        self._detected = True  
        # number of consecutive undetected iterations
        self._num_undetected = 0
        
        # x values of the last n fits of the line
        self._fitted_xvals_list = [] 
        #average x values of the fitted line over the last n iterations
        self._fitted_xvals_average = None     
        
        #polynomial coefficients for the last n fits of the line
        self._poly_fit_list = None  
        #polynomial coefficients averaged over the last n fits of the line
        self._poly_fit_average = None  
        #polynomial coefficients for the most recent fit
        self._poly_fit_current = None  
        #difference in fit coefficients between last and new fits
        self._poly_fit_diffs = np.array([0,0,0], dtype='float') 
        
        #radius of curvature of the line in some units
        self._radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self._line_base_pos = None 
        
        #x values for detected line pixels
        self._xvals = None  
        #y values for detected line pixels
        self._yvals = None
        
        #number of interations to track
        self._num_iter = 29
        
    
    @property
    def detected(self):
        return self._detected
    
    
    @detected.setter
    def detected(self, value):
        self._detected = value
    
    
    def add_results(self, result):

        # set lane detection flag       
        # check to see that curvatures for left and right lanes are not far apart
        if self._radius_of_curvature is not None:
            ratio = max(result['curve_rad'],self._radius_of_curvature) / min(result['curve_rad'],self._radius_of_curvature)
            self._detected = self._detected and (ratio <= 1.3)
        
        # check to see that lanes are roughply parallel to the previous find
        if len(self._fitted_xvals_list) > 0:
            ratio = np.max((result['fitted_xvals'],
                            self.fitted_xvals_list[-1]),axis=0) / np.min((result['fitted_xvals'],
                                                                            self.fitted_xvals_list[-1]),axis=0)
            self._detected = self._detected and (ratio.min()>=0.5) and (ratio.max()<=2.0)
        
        # add the results to the object if good quality is established
        if self._detected:
            
            # reset num_undetected
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
            self._poly_fit_diffs = self._poly_fit_list[-1]-self._poly_fit_list[-2]
            
            # update radius of curvature and base position of lane lines
            self._radius_of_curvature = result['curve_rad']
            self._line_base_pos = result['base_pos']
            
            # set all x and y pixel points
            self._xvals = result['xvals']
            self._yvals = result['yvals']
    
        else:
            
            self._num_undetected += 1
            if self._num_undetected > self._num_iter*2:
                self.__init__
    
    
    def get_best_lane(self, result):
        return self._poly_fit_average, self._radius_of_curvature, self._line_base_pos
        
        