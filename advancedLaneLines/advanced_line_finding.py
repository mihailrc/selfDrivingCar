import os
import cv2

import matplotlib

import matplotlib.image as mpimg

from advancedLaneLines.sdc import utils
from advancedLaneLines.sdc import line_detector
from moviepy.editor import VideoFileClip


matplotlib.use('TkAgg')

# images = ['test_images/straight_lines1.jpg','test_images/straight_lines2.jpg', 'test_images/test1.jpg',
#           'test_images/test2.jpg','test_images/test3.jpg','test_images/test4.jpg','test_images/test5.jpg',
#           'test_images/test6.jpg']

images = ['test_images/test4.jpg']

lineDetector = line_detector.LineDetector()

def process_images(images):
    for image in images:
        print(image)
        img = mpimg.imread(image)
        [undistorted, bird_eye, binary, gradient_binrary] = lineDetector.camera.process_image(img)
        left, right = lineDetector.extract_lines(binary)
        if(left is None):
            left, right = lineDetector.extract_lines(gradient_binrary)
        if(left is None):
            left, right = binary, binary

        print(left.shape, right.shape)
        processed = lineDetector.process_image(img)
        utils.plot_images([bird_eye, binary, lineDetector.leftLine.currentLine, lineDetector.rightLine.currentLine], 2, 2, ['Brdie', 'Binary', 'Left', 'Right'])
        utils.plot_images([processed], 1,1,['Result'])

input_movies_dir = "./"
output_movies_dir = "output_videos/"

try:
    os.stat(output_movies_dir)
except:
    os.mkdir(output_movies_dir)

def processVideo(filename):
    clip1 = VideoFileClip(input_movies_dir + filename)
    # clip1.save_frame("test_images/test12.jpg", t='00:00:42.12')
    output_clip = clip1.fl_image(lineDetector.process_image)
    output_clip.write_videofile(output_movies_dir + filename, audio=False)

# processVideo("project_video.mp4")
process_images(images)






