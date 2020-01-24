# **Advanced Lane Finding** 

### Completed for Udacity Self Driving Car Engineer - 2018/03

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

[image1]: ./ReportPics/calibration3.jpg "Chess Board Corners"
[image2]: ./ReportPics/calibration6.jpg " Chess Board Corners 2"
[image3]: ./ReportPics/ImageDistortion.png "Image Distortion Examples"
[image4]: ./ReportPics/HLV2.jpg "Hue Saturation Lightness"
[image5]: ./ReportPics/ColorThreshold.png "Color Threshold Examples"
[image6]: ./ReportPics/GradientThreshold.png "Gradient Threshold"
[image7]: ./ReportPics/CombinedThreshold.jpg "Combined Threshold"
[image8]: ./ReportPics/Perspective.png "Illustrating Perspective Transform"
[image9]: ./ReportPics/PerspectiveTransform.jpg "Illustrating Perspective Transform"
[image10]: ./ReportPics/LaneLines1.png "Plotting Lane Lines"
[image11]: ./ReportPics/LaneLines2.png "Plotting Lane Lines"
[image12]: ./ReportPics/RCurve.png "Curvature Formula"
[image13]: ./ReportPics/output1.png "Pipeline Results"
[image14]: ./ReportPics/output2.png "Pipeline Results"
[image15]: ./ReportPics/output3.png "Pipeline Results"




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

Once the object points and image points were calculated, the camera is calibrated using the following line of code:

```
ret, mtx, dist, rvecs, tvecs =cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
```

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Distortion correction is the first step in the processing pipeline. 

The camera calibration was verified by undistorting test images using the the cv2.undistort(...) function with the mtx and dist matricies computed in the above calibration. This takes place in the 2nd notebook code cell.  Images in the pipeline are un-distorted using this line of code:

```
undistorted = cv2.undistort(test_img, mtx,dist,None,mtx)
```

Here are a few test images illustrating un-distortion. Looking at the back of the white car on the 1st and second images proves that the distortion correct was done correctly.

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

Using a similar approach, I chose the red channel from the RGB image to threshold for lane lines:
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

I applied  gradient thresholding in the gradient_threhsold(img) function, found inside the 4th code cell. I started by converting the RGB image to  grayscale and then using cv2.Sobel(...) with a kernel of 9 on both the x and y axes. 
```
sobel_kernel = 9 
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, sobel_kernel)
sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,sobel_kernel) 
```
Next I computed the gradient angle and thresholded the output around the vertical to extract lane lines. This yielded poor results and struggled to differentiate lane lines from general road noise ( I left the code commented inside the gradient_threshold(img) function ). I had more success using the magnitude of the gradient and thresholding, shown in the code below:

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

Here are some sample images showing the results of the binary threshold.  I have isolated the gradient(green) from the color (blue) threshold to highlight their differences:

![alt text][image7]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is included in the jupyter notebook under the "Perform Perspective Transform" Heading, the 6th code cell. The function WarpImage(img,src,dst) manages the transformation using the cv2.warpPerspective(...) function.
```
def WarpImage(img,src,dst):
	M = cv2.getPerspectiveTransform(src,dst)    
	return cv2.warpPerspective(img, M, (img.shape[1] , img.shape[0]), flags=cv2.INTER_LINEAR)
```

The function takes an image (img), along with the source (src) & destination (dst) arrays that are used to outline the transformation boundary. I chose to calibrate the source and destination arrays on a straight lined image as I would be able to easily gauge the effectivness of the chosen boundaries. 


Starting with the easier array, destination (dst), I chose a rectangle that was offset by 25% of the image width and spanned the entire image height. This yielded the following results :

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
With the source (src) array, the goal was to tune it to encapsulate both left and right lane lines evenly, while spanning sufficiently 'back' into the image. This was measured by displaying a trapezoid in red overtop of the undistorted test image. In order to accomplish this, I setup a series of parameters, numbers between 0 and 1 used to scale the trapezoid area. The parameters are described in comments beside their definition in the code snippet below. The OffC value is used to shift the trapezoid to the center of the image, given that the driver was not directly in the center of the lane line for the straight_lines1.jpg test image.

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
 The source and destination arrays are fed into the WarpImage(...) function in the pipeline.

I verified that my perspective transform was working as expected by drawing the `src` points onto a test image and warped it to verify that the lines appear parallel in the warped image:

![alt text][image8]

The perspective transform was applied on other images and shown to be somewhat sufficient (read ahead to further sections to see how the lack of smoothess in test6.jpg was managed).

![alt text][image9]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Notebook code cells 7, 8, & 9 complete the remainder of the image pipeline. Code Cell 7 contains a Tracker class that is used to keep track of previous frames. This is only used when processing images consecutively as part of a video, and not while developing for a single image at a time. The Tracker class is explained in more detail in the discussion at the end of the report. Code cell 8 contains the code for questions 4, 5, and 6 within the FitLaneLine(...) function. Code cell #9 summarizes the entire pipeline into a single function, process_image(img)

I chose to use the convolution method from the lectures for selecting lane lines.  Window width & height are set to 45 & 80 respectively, slicing the image into 9 horizontal slats whereby convolutions will be performed on each layer (starting from 2) to determine the most likely position of the lane line, aka where the most amount of white pixels are. The bottom 3/4 of the image is convoluted with a row of 1's to determine the pixel coordinate (in x) that corresponds to lane line position for both left (left half of image) and right (right half of image) lanes. Also, I should point out that the left 75 pixels (variable L_offset) are ignored due to noise found on the left shoulder that is picked up in color thresholding. The starting x-coordinates for the lane lines are computed as followes (lines 44-47 in FitLaneLine(...)):

```
l_sum = np.sum(image[int(3*img_height/4):,L_offset:int(img_width/2)], axis=0)    
l_center = np.argmax(np.convolve(window,l_sum))-window_width/2 + L_offset
r_sum = np.sum(image[int(3*img_height/4):,int(img_width/2):img_width], axis=0)
r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(image.shape[1]/2)
```
After the starting coordinates of left and right lane lines are understood, a for loop iterates over the remaining layers and applies the same convolution technique that was used on the bottom layer. A convolution threshold of 50 is used to disregard any new layers that do not have a sufficient number of pixels to constitute a lane marking.

After looping over each layer, we are left with two 1x9 arrays whereby each value corresponds to the x coordinate of the pixel for the left and right lane line respectively. The y-values are simply the y  coordinate at the center of each horizontal window.

At this point, I could fit each of the lane lines to a polynomial using :

```
# y-values corresponding to the middle of each window, starting from bottom up
window_yvals = np.arange(img_height - window_height/2, 0 ,-window_height )    
# Compute polynomial coefficents for current image, f(y) = x = a*y*y + b*y + c     
lpoly = np.polyfit(window_yvals , leftx , 2).tolist()
rpoly = np.polyfit(window_yvals , rightx , 2).tolist()
```

With the coeffieients contained in lpoly & rpoly, the equation of the lane lines were in the quadratic form f(y) = x = a*y*y + b*y + c. The function is with respect to y to avoid extra large coefficients for large vertical lane lines. f(y) was then computed for each y value from 0 - 920 and then plotted over top of the 2D warped perspective image. The convolutions can be seen in green while the lane line is in red. The line of best fit does an effective job at tracking the convolution windows shown in green.

![alt text][image10]	
![alt text][image11]	


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of curvature and vehicle position offset were calculated at the end of FitLaneLine(...) function. To start, the coefficients from the polynomials are scaled to meters using a conversion factor for pixels to meters in both x and y:

```
# Pixel to Meters Conversion Factor
ym_per_pix = 30./720
xm_per_pix = 3.7/700

# Scale the polynomials to meters:     
left_fit_meters  = [left_poly[0]*xm_per_pix/(ym_per_pix**2),
                    left_poly[1]*xm_per_pix/ym_per_pix,
                    left_poly[2]]
right_fit_meters  = [right_poly[0]*xm_per_pix/(ym_per_pix**2),
                     right_poly[1]*xm_per_pix/ym_per_pix,
                     right_poly[2]]

``` 

With the scaled polynomial coefficients, the curvature can be calculated using the formula provided in the lectures, shown below. The y value for the calculation was chosen as the bottom of the image (start of the lane line). The curvature is calculated for both left and right lane lines and then averaged.

![alt text][image12]



The lane position offset calculation was made by subtracting the pixel location of the center of the vehicle (image) by the center of the lane (average of left and right lane starting x coordinates). The resulting value is multiplied by the x-scale to give the offset in meters.

```  
#Calculate vehicle distance from center:    
vehicle_center = img_width/2 # Compute the center of the vehicle
lane_center = round((leftx[0] + rightx[0]) / 2.0,1) # Compute the center of the lane based on start points                                
pixel_off_center = (lane_center - vehicle_center)  # Compute number of pixels vehicle is off center
offC = pixel_off_center*xm_per_pix # Convert pixels to meters
```

Finally, as the last step in the FitLaneLine(...) function, the curvature and center offset are displayed on the image as strings.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The final image is generated by warping the perspective transform image (2D birds eye view) with the lanes drawn on in red, using the cv2.warpPerspective(...) function with M inverse. The lane area is also transformed back to the driver view using the same technique.

Once all images are in the same perspective, they are joined using the cv2.addWeighted(...). The code is provided below, and can be seen near the end of the FitLaneLine(...) function.

```
# Insert points onto a blank image:
zeros = np.zeros_like(warped)
points_img = np.zeros_like(warped)
road_bkg = np.zeros_like(warped)
road = np.zeros_like(warped)
lane_area= np.dstack((zeros, zeros, zeros)) # Image to overlay middle lane area
thickness = 35

# Create art
cv2.polylines(points_img,[L_points], False, (255,255,255), thickness)    
cv2.polylines(points_img, [R_points], False,  (255,255,255) ,thickness)
cv2.fillPoly(lane_area, ([combined_pts]), (0,255, 0))

road = np.array(cv2.merge((points_img, zeros,zeros)),np.uint8)
road_bkg = np.array(cv2.merge((points_img, points_img,points_img)),np.uint8)

# Draw Lines & UnWarp Image back to normal perspective:
img_size = (zeros.shape[1],zeros.shape[0])     
road_warped = cv2.warpPerspective(road,Minv,img_size,flags = cv2.INTER_LINEAR)
road_warped_bkg = cv2.warpPerspective(road_bkg,Minv,img_size,flags = cv2.INTER_LINEAR)
lane_area_warped = cv2.warpPerspective(lane_area,Minv,img_size,flags = cv2.INTER_LINEAR)

# Superimpose lane lines & lane area over original undistorted image:    
base = cv2.addWeighted(undistorted_img, 1, road_warped_bkg, -1, 0.0)
result = cv2.addWeighted(base, 1, road_warped, 1, 0.0)  
result = cv2.addWeighted(result, 1, lane_area_warped, 0.5, 0.0)
``` 

Output is shown for test images:
![alt text][image13]
![alt text][image14]
![alt text][image15]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I would like to highlight the importance of using previous frame results when processing consecutive frames in the video. Because the lane line positions hardly change from frame to frame, I was able to smooth out the lane lines between frames very effectively. I accomplished this by developing a Tracker class to store values computed for previous images. The Tracker class was used in the following ways:

Identifying the starting lane line position:
* The starting point of the left / right lane lines hardly changes between video frames. The starting positions are stored in the Tracker variable LastLeftLaneStart and LastRightLaneStart to mark the x coordinates of the beggining of the lane lines for the previous frame. If the calculated starting positions are too far away from the previous frame, then they are dis-regarded and the old positions are used. Lines 54- 61 in FitLaneLine(...):
```
if(np.abs(l_center - Tracker.LastLeftLaneStart) < LaneStartOffset):
    Tracker.UpdatePrevLeftLaneStart(l_center)
else:
    l_center =  Tracker.LastLeftLaneStart 
if(np.abs(r_center - Tracker.LastRightLaneStart) < LaneStartOffset):
    Tracker.UpdatePrevRightLaneStart(r_center)
else:
    r_center =  Tracker.LastRightLaneStart      
```
Filter Curvature & Vehicle Offset:
* Secondly, the Tracker class is used to average the curvature and offset values for the previous 50 frames, acting as a simple filter.

Average Polynomials:
* Finally, the coefficients for each line are averaged for the past 10 frames to reduce wobbly lines and help them follow the lane smoothly. Lines 160-163 in FitLaneLine(...):
```
if(frame_counter  > Tracker.GetPolyThreshold()):
    Tracker.PopCoef() 
Tracker.AddLines(lpoly,rpoly)        
left_poly,right_poly = Tracker.AverageCoef()
```


In hindsight, the image pipeline will fail in situations where there is little contrast between lane markings and road surface. The binary thresholding was tuned specifically to the lighting conditions in the project_video.mp4 and thus will not generalize well to all driving situations. To make the pipeline more robust, I would spend more time generating the binary threshold image using more complex color and gradient thresholding techniques for a wide variety of lighting and road surface / lane marking combinations. Being able to extract the lane lines effectively was the best indicator of success in this project.
