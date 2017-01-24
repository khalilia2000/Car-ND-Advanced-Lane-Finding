# Car-ND-Advanced-Lane-Finding

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./camera_cal/calibration2.jpg "Calibration Image"
[image2]: ./camera_cal/corners_found2.jpg "Calibration Image with Chessboard Corners Drawn"
[image3]: ./camera_cal/undistort2.jpg "Calibration Image after Undistortion"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here the rubric points are considered individually and descriptions are provided on how I addressed each point in my implementation.  
---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.   

I did use the template provided in the course notes and modified it. You're reading the README.md!
###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the function calibrate_camera_from_path (lines #41 through #100 of the file named `findlanelines.py`).  

This function gets a path as input along with number of chess board corners in x and y directions and returns the camera and distortion matrices. It also has additionl kwargs (i.e. dave_with_corners and save_undistort), that can be used to save the transformed images.
Here is how the function works:  
1- "opbject points" are prepared, which will be the (x, y, z) coordinates of the chessboard corners in the world.Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image. Thus, `objp` is just a replicated array of coordinates. `objpoints` and `imgpoints` are the initialized as empty lists: `objpoints` will be appended with a copy of `objp` every time the program successfully detects all chessboard corners in a test image. `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection;  
2- program uses the glob library to read all files fitting `calibration*.jpg` naming conventions;  
3- for each image that is read, `cv2.findChessboardCorners()` is called to find the chessboard corners;  
4- if save_with_corners kwarg is True, then file is saved with chessboard corners drawn on it;  
5- function `cv2.calibrateCamera()` is then called with 'objpoints' and 'imgpoints' and the camera and distortion matrices are calcualted;  
6- if save_undistort kwarg is True, then file is undistorted using `cv2.undistort()` function and then saved;  
7- the function returns the camera matrix and distortion matrix at the end.  

Examples of the original image, original image with chessboard corners drawn on it, and undistorted image are shown below:  

[1]: <img src="./camera_cal/calibration2.jpg" alt="Calibration Image" height =144 width=256>
[2]: <img src="./camera_cal/corners_found2.jpg" alt="Same Image with Chessboard Corners Drawn" height =144 width=256>
[3]: <img src="./camera_cal/undistort2.jpg" alt="Same Image after Undistortion" height =144 width=256>

| Source        | Destination   | test |
|:-------------:|:-------------:|:----:| 
| [1] | [2] | [3] |


###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
A classe named `ImageTransform` is created for the sole purpose of manipulating images. This class (which is contained in file `imagetransform.py`) gets an array of images, an array of labels as well as camera and distorsion matrices obtianed prviously plus the colorspec of the images. This class has many methods to manipulate images. One of the methods is named `to_undistort()`, which is called when the class is created in `__init__()` method, and calls `cv2.undistort()` method to undistort the images. Gaussian blur (using `cv2.GaussianBlur()`) is also applied to the images in the `__init__()` method prior to undistorting the images. The images are also converted to RGB colorspec after undistorting in the `__init__()`. The follwoing images show a random frame that is in original format plus the same frame after blurring / undisotrting:

![alt text][image4]


####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])

```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
