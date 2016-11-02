# Lane Detection

./findLines.sh - finds lines in test_images and movies in test_videos

The algorithm consists of the following steps:
 - transform image to gray or HSV if gray does not work
 - apply gaussian blur to remove noise
 - apply Canny to find edges
 - define region of the interest as a trapezoid
 - find line segments within this region of interest using Hough transform
 - filter out line segments that have slope that is too unreasonable e.g. almost parallel to front of the car
 - filter out line segments that do not intersect the x bottom axis at reasonable intervals
 - if y = a*x + b find the average for a and b for the remaining line segments. Define a line using these average parameters.
 - if no line is found try a different collor space and find lanes in this color space. E.g. yellow line on gray road using gray scalling is challenging so I used HSV instead.
 - finally apply smoothing to reduce jitter

The algorithm can be improved in several ways:
- use a more sophisticated way of finding the line parameters. E.g. fit the line properly instead of simple averaging.
- improve the way the algorithm finds the best candidate. For example if lines found in both color spaces are poor candidates use the existing line for a few frames. Search more color spaces if needed.
- exclude lines that show a sudden change in slope since the are likely wrong.

The algorithm is not going to work too well if there line segments that have the correct orientation but have a position that is not ideal. This may be caused by things like shade from a tree. For example the challenge exercise had this exact problem. Fortunately these edges were filtered out because of wrong slope.
