# **Advanced Lane Finding** 

## James DiDonato
## March 2018



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

[image1]: ./ReportPics/calibration3.jpg "Chess Board Corners"
[image2]: ./ReportPics/calibration6.jpg " Chess Board Corners 2"
[image3]: ./ReportPics/ImageDistortion.png "Image Distortion Examples"
[image4]: ./ReportPics/HLV2.jpg "Hue Saturation Lightness"
[image5]: ./ReportPics/ColorThreshold.png "Color Threshold Examples"
[image6]: ./ReportPics/GradientThreshold.png "Gradient Threshold"
[image7]: ./ReportPics/CombinedThreshold.jpg "Combined Threshold"
[image8]: ./ReportPics/Perspective.png "Illustrating Perspective Transform"
[image9]: ./ReportPics/PerspectiveTransform.jpg "Illustrating Perspective Transform"
[image10]: ./ReportPics/LaneLines1.png "a"
[image11]: ./ReportPics/LaneLines2.png "a"
[image12]: ./ReportPics/.jpg "a"
[image13]: ./ReportPics/.jpg "a"
[image14]: ./ReportPics/.jpg "a"
[image15]: ./ReportPics/.jpg "a"




## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

Welcome to my report!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The  code for this step is located in the first code cell of the jupyter notebook. In order to correct for distortion I had to calibrate the camera using the calibration sample images provided. To do so, I loop through each image example and find the chess board corners and map them to a 9 x 6 grid. For each image, the pixel locations of the corners represent the image points that need to be mapped to the object point grid. The object points comprise a 9x6 grid and do not change for any image.

I should point out that only 17 of the 20 images contained all corners.
Below are some sample images illustrating the image points: 

![alt text][image1]

![alt text][image2]

Once the object points and image points were calculated, camera is calibrated using the following line of code:

```
ret, mtx, dist, rvecs, tvecs =cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
```

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Distortion correction is the first step in the processing pipeline. 

The images are undistorted using the cv2.undistort() function with the mtx and dist matricies computed from the calibration above. Images in the pipeline are un-distorted using this piece of code:

```
undistorted = cv2.undistort(test_img, mtx,dist,None,mtx)
```

Here are a few test images illustrating un-distortion:

![alt text][image3]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image. I began with color.

#### Color Threshold

I started by plotting the test images along their Hue, Saturation, and Lightness axes in the color_threshold(img) function:

![alt text][image4]

It is clear that the saturation does the best job at illuminating the lane lines. After some testing, I chose a thresholding window with the following piece of code:
```
# Thresholding Saturation (OK)   
s_thresh_min = 150
s_thresh_max = 220
s_binary = np.zeros_like(s_channel)
s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1 
```

Using a similar approach, I chose the red channel from the RGB image to extract lane lines.
```
# Thresholding Red (OK)
r_thresh_min = 227
r_thresh_max = 255
r_binary = np.zeros_like(s_channel)
r_binary[(r_channel >= r_thresh_min) & (r_channel <= r_thresh_max)] = 1
```

After thresholding on both the red and saturation color channels, I combined the two techniques into one image as follows :

```
# Combine Saturation & Red Thresholding
combined_binary = np.zeros_like(r_binary)
combined_binary[(r_binary == 1) | (s_binary == 1)] = 1 
```

Here are some examples of the color thresholding applied on test images:

![alt text][image5]

#### Gradient Threshold:

I applied  gradient thresholding in the gradient_threhsold(img) function. I started by converting the RGB image to  grayscale and then using cv2.Sobel() with a kernel of 9 on both the x and y axes. 
```
sobel_kernel = 9 
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, sobel_kernel)
sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,sobel_kernel) 
```
Next I computing the gradient angle and thresholded the output around the vertical to extract lane lines. This yielded poor results and struggled to differentiate lane lines from general road noise ( I left the code commented inside the gradient_threshold(img) function ). I had more success using the magnitude of the gradient and thresholding, shown in the code below:

```
# Thresholding gradient magnitude (OK)
grad_mag = np.sqrt(sobelx**2 + sobely**2)
scale_factor = np.max(grad_mag)/255 
grad_mag = (grad_mag/scale_factor).astype(np.uint8) 
mag_thresh_min = 75
mag_thresh_max = 225
mag_binary = np.zeros_like(grad_mag)
mag_binary[(grad_mag >= mag_thresh_min) & (grad_mag <= mag_thresh_max)] = 1  
```

Here are some sample images showing the gradient threhsold. In general, the gradient threshold performed better than the color threhsold in picking out lane lanes in the distance, where the color contrast was minimal.

![alt text][image6]

#### Combined Threshold

Finally, to complete the binary image transformation, I combined the gradient and color thresholds with an OR operation. This is completed in the Binary_Threshold(img) function:
```
# Combine Gradient + Color Thresholding
def Binary_Threshold(img):    
	gradient_binary = gradient_threshold(img)    
	color_binary = color_threshold(img)       
	combined_binary = np.zeros_like(gradient_binary)
	combined_binary[(gradient_binary == 1) | (color_binary == 1)] = 1
	return combined_binary 
```

Here are some sample images showing the results of the binary threshold.  I have isolated the gradient from the color threshold to highlight their differences:

![alt text][image7]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is included in the jupyter notebook under the "Perform Perspective Transform" Heading, the 6th code cell. The function WarpImage(img,src,dst) manages the transformation using the cv2.warpPerspective() function.
```
def WarpImage(img,src,dst):
	M = cv2.getPerspectiveTransform(src,dst)    
	return cv2.warpPerspective(img, M, (img.shape[1] , img.shape[0]), flags=cv2.INTER_LINEAR)
```

The function takes an image (img), along with the source (src) & destination (dst) arrays that are used to outline the transformation boundary. I chose to calibrate the source and destination arrays on a straight lined image as I would be able to easily gauge the effectivness of the window sizes. 

.
Starting with the easier destination array, I chose a rectangle that was offset by 25% of the image width and spanned the entire image height. This yielded the following pixel values :

```
# Define transformation to area (dst):
offset = 0.25 *img_width
dst = np.float32([[offset,img_height],[offset,0],[img_width-offset,0],[img_width - offset,img_height]])

# Destination Rectangle Output:
dst =  [[320. 720.]
 [320.   0.]
 [960.   0.]
 [960. 720.]]
 ```
With the source (src) array, the goal was to tune it to encapsulate both left and right lane lines evenly, while spanning sufficiently 'back' into the image. This was measured by displaying a trapezoid in red overtop of the undistorted test image. In order to accomplish this, I setup a series of parameters, numbers between 0 and 1 representing the trapezoid area. These are described in comments beside their definition in the code snippet below. The OffC value is used to shift the trapezoid to the center of the image, given that the driver was not directly in the center of the lane line for the straight_lines1.jpg test image.

```
# Define transformation from area (src):
OffC = 20 # How many pixels camera is off center
bottom_offset = 0.98 # Offset from  bottom of image trapezoid bottom
bottom_width = 0.72 # Width of trapezoid in the foreground
top_width = 0.1# Width of trapezoid in the distance
depth = 0.375 # How far to project into the distance

src = np.float32([[(img_width/2.0)*(1 - bottom_width)+OffC ,img_height*bottom_offset],
                  [(img_width/2.0)*(1 - top_width)+OffC,img_height*(1-depth)], 
                  [(img_width/2.0)*(1 + top_width)-OffC,img_height*(1-depth)],
                  [(img_width/2.0)*(1 + bottom_width) , img_height*bottom_offset] 
                 ])

# Source Trapezoid Output:
src =  [[ 199.2  705.6]
 [ 596.   450. ]
 [ 684.   450. ]
 [1100.8  705.6]]
 ```
 The source and destination arrays are fed into the WarpImage() function in the pipeline.

I verified that my perspective transform was working as expected by drawing the `src` points onto a test image and warped it to verify that the lines appear parallel in the warped image:

![alt text][image8]

The perspective transform was applied on other images and shown to be somewhat sufficient (read ahead to further sections to see how the lack of smoothess in test6.jpg was managed).

![alt text][image9]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Notebook code cells 7, 8, & 9 contain the remainder of the image pipeline. Code Cell 7 contains a Tracker class that is used to keep track of previous frames. This is only used when processing images consecutively as part of a video, and not while developing for a single image at a time. The Tracker class is explained in more detail in the discussion at the end of the report. Code cell 8 contains the code for questions 4, 5, and 6.

I chose to use the convolution method from the lectures for selecting lane lines.  Window width & height are set to 45 & 80 respectively, slicing the image into 9 horizontal slats whereby convolutions will be performed on each layer (starting from 2) to determine the most likely position of the lane line, aka where the most amount of white pixels are. The bottom 3/4 of the image is convoluted with a row of 1's to determine the pixel coordinate (in x) that corresponds to lane line position for both left (left half of image) and right (right half of image) lanes. Also, I should point out that the left 75 pixels are ignored due to noise found on the left shoulder that is picked up in a color threshold. This is completed in the following steps (lines 44-47 in FitLaneLine()):

```
l_sum = np.sum(image[int(3*img_height/4):,L_offset:int(img_width/2)], axis=0)    
l_center = np.argmax(np.convolve(window,l_sum))-window_width/2 + L_offset
r_sum = np.sum(image[int(3*img_height/4):,int(img_width/2):img_width], axis=0)
r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(image.shape[1]/2)
```
After the starting coordinates of left and right lane lines are understood, a for loop iterates over the remaining layers and applies the same convolution technique that was used on the bottom layer. A convolution threshold of 50 used to disregard any new layers that do not have a sufficient number of pixels to constitute a lane marking.

After looping over each layer, we are left with two 1x9 arrays whereby each value corresponds to the x coordinate of the pixel for the left and right lane line respectively. The y-values are simply the y  coordinate at the center of each horizontal window.

![alt text][image10]	
![alt text][image11]	


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

- Talk about Tracker Class