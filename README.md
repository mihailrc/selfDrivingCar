# Self Driving Car
Projects for Udacity's Self Driving Car nano degree

### Finding Lane Lines on the Road

The goal of the project is to use Computer Vision techniques to identity lane lines on the road. The solution uses color selection, Gaussian smoothing to reduce noice, Canny Edge Detection and Hough Transform Line detection. The algorithm used to detect the lines is very simple and naive but works surprisingly well.

 - [Solution](findingLaneLines/P1.ipynb)
 - [Udacity review](findingLaneLines/Udacity_Review.pdf)

### Traffic Signs Classification
  The goal of the project is to train a model on the German Traffic Sign Dataset so it can classify traffic signs from natural images. The model uses a 4 Layer Convolutional Neural Network and has an accuracy of of 97.78% on the testing dataset when trained over 50 epochs. For comparison Human accuracy on this task is 98.84%. This model can be trained in approximately 1 hour on a 2013 MackBook Pro GPU.

 - [Solution](Traffic_Signs_Recognition.ipynb)
 - [Udacity review](trafficSigns/Udacity_Review.pdf)


### Behavioral Cloning

  The goal of the project is to train a deep neural network to drive around a track using Udacity's simulator. Training data was collected by recording a user driving around the track.

   - [Solution](behavioralCloning)
   - [Udacity review](behavioralCloning/Udacity_Review.pdf)

### Advanced Lane Finding

 The goal of the project is to use advanced Computer Vision techniques to identify lane lines on the road. The solution transforms the images into a bird eye-view then applies Sobel gradient and color thresholding to create a binary image. The binary image is further processed to identify lane pixels that are fitted to a second degree polynomial. The calculated line is rendered back to the original image.

  - [Solution](advancedLaneLines)
  - [Udacity review](advancedLaneLines/Udacity_Review.pdf)

### Vehicle Detection

 The goal of this project is to write a software pipeline to identify vehicles in a video. The solution uses Histogram of Oriented Gradients(HOG) and Color Histogram to extract features from images. The images are classified into cars and non cars using SVM. Sliding windows of different sizes are used to locate cars on the images, then a couple of thresholding techniques are used to reduce false positives.

  - [Solution](vehicleDetection)
  - [Udacity review](vehicleDetection/Udacity_Review.pdf)
