## Advanced Lane Finding

### Camera Calibration

Camera was calibrated using 20 chessboard pattern images provided in the project repository.

TODO - add link with code
The code that performs camera calibration can be found here.

calibrateCamera() takes as input the path to the calibration images directory and outputs the calibration matrix and distortion coefficients. This method uses OpenCV methods findChessboardCorners() and calibrateCamera() to calculate the calibration coefficients. These coefficients are saved in a pickle file in order to avoid calculating them over and over again.

An example of an original chessboard image and the corresponding distortion corrected image is provided below.

<table>
  <tr>
    <th>Original</th>
    <th>Undistorted</th>
  <tr>
  <tr>
   <td><img src="camera_cal/calibration1.jpg"></td>
   <td><img src="output_images/undistorted_calibration.jpg"></td>
  </tr>
</table>

### Pipeline (test images)

The image processing pipeline consists of the following steps:
1. Apply gaussian blur to original image to reduce noise
2. Un-distort the resulting image using the camera calibration coefficients calculated previously.
3. Apply perspective transform to create a bird-eye view of the image.
4. Create a binary image by applying color and/or gradient thresholding
5. Create a binary image for each of the lines
6. Apply polynomial fit to each binary line images
7. Draw the lines on the bird-eye view
8. Apply inverse perspective transform to convert the image from bird-eye view to regular view
9. Overlap image with lines to undistorted image
10. Calculate Curvature and vehicle position with respect to center and render them on the image

The set of transformation applied to an image is presented below.

<table>
  <tr>
    <th>Original</th>
    <th>Undistorted</th>
  <tr>
  <tr>
   <td><img src="output_images/original.jpg"></td>
   <td><img src="output_images/undistorted.jpg"></td>
  </tr>
  <tr>
    <th>Bird Eye</th>
    <th>Color Binary</th>
  <tr>
  <tr>
   <td><img src="output_images/bird_eye.jpg"></td>
   <td><img src="output_images/color_binary.jpg"></td>
  </tr>
  <tr>
    <th>Gradient Binary</th>
    <th>Combined Binary</th>
  <tr>
  <tr>
   <td><img src="output_images/gradient_binary.jpg"></td>
   <td><img src="output_images/combined_binary.jpg"></td>
  </tr>
  <tr>
    <th>Left</th>
    <th>Right</th>
  <tr>
  <tr>
   <td><img src="output_images/left.jpg"></td>
   <td><img src="output_images/right.jpg"></td>
  </tr>
  <tr>
    <th>Lines Fit</th>
    <th>Lines Fit Unwarped</th>
  <tr>
  <tr>
   <td><img src="output_images/bird_eye_lines.jpg"></td>
   <td><img src="output_images/bird_eye_unwraped.jpg"></td>
  </tr>
  <tr>
    <th>Overlapped</th>
    <th>Result</th>
  <tr>
  <tr>
   <td><img src="output_images/overlaped.jpg"></td>
   <td><img src="output_images/result.jpg"></td>
  </tr>
</table>

### Perspective transform
The goal of the perspective transform is to provide a bird eye view of the lines so they appear parallel in the transformed image. Transforming back and forth between regular and bird-eye view is done using two transformation matrices.

The method that calculates these matrices is called get_transformation_matrices() and can be found in Camera class. This method uses OpenCV's getPerspectiveTransform() and manually provided source and destination points. An example of the bird-eye image was presented above and as expected shows the parallel lanes.

### Creating the binary image
The binary image is created using a combination of color gradient thresholding. Color thresholding first transforms the image from RGB to HLS space the applies a threshold on S channel. 

Gradient thresholding first transform the image into the gray scale the calculates Sobel gradient along x axis because this gradient will emphasize vertical lines.

The combined binary puts the two binaries together. Examples of these binary images can be found above.

### Line identification
Describe

### Radius of curvature and vehicle position
Describe

#### Pipeline (video)
 * Link to video

#### Discussion
Briefly discuss any problems / issues you faced in your implementation of this project. Where will your pipeline likely fail? What could you do to make it more robust?

Discussion includes some consideration of problems/issues faced, what could be improved about their algorithm/pipeline, and what hypothetical cases would cause their pipeline to fail.
