import numpy as np
import cv2

class LineDetector:
    def __init__(self):
        self.curentLines = [[[0,0,0,0]], [[0,0,0,0]]]
        self.smoothingFactor = 0.3

    def convertImageToChannel(self, img, channel = cv2.COLOR_BGR2GRAY):
        """Transforms from BGR to specified channel"""
        return cv2.cvtColor(img, channel)

    def canny(self, img, low_threshold, high_threshold):
        """Applies the Canny transform"""
        return cv2.Canny(img, low_threshold, high_threshold)

    def gaussian_blur(self, img, kernel_size):
        """Applies a Gaussian Noise kernel"""
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    def region_of_interest(self, img, vertices):
        """
        Applies an image mask.

        Only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black.
        """
        # defining a blank mask to start with
        mask = np.zeros_like(img)

        # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(img.shape) > 2:
            channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        # filling pixels inside the polygon defined by "vertices" with the fill color
        cv2.fillPoly(mask, vertices, ignore_mask_color)

        # returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

    def get_line_info(self, line):
        """
        given y = ax + b it calculates a, b
        :param line:
        :param image_shape:
        :return:
        """
        for x1, y1, x2, y2 in line:
            denominator = (x2 - x1)
            if (denominator == 0):
                denominator = 0.000001
            a = (y2 - y1) / denominator
            if (a == 0):
                a = 0.000001

            b = (y1 * x2 - y2 * x1) / denominator
            return np.array([a, b])

    def get_lines_info(self, lines):
        lines_info = np.empty((0, 2))
        for line in lines:
            lines_info = np.append(lines_info, [self.get_line_info(line)], axis=0)
        return lines_info

    def is_left_line(self, line, image_shape):
        line_info = self.get_line_info(line)
        a = line_info[0]
        b = line_info[1]
        intersection_point = (image_shape[0] - b) / a
        return (intersection_point > 0 and intersection_point < image_shape[1] * 0.25 and a > -0.9 and a < -0.5)

    def is_right_line(self, line, image_shape):
        line_info = self.get_line_info(line)
        a = line_info[0]
        b = line_info[1]
        intersection_point = (image_shape[0] - b) / a
        return (
        intersection_point > image_shape[1] * 0.75 and intersection_point < image_shape[1] and a > 0.4 and a < 0.8)

    def filter_lanes(self, lines, image_shape, filter):
        filtered_lanes = np.empty((0, 1, 4))
        for line in lines:
            if (filter(line, image_shape)):
                filtered_lanes = np.append(filtered_lanes, [line], axis=0)
        return filtered_lanes

    def calculate_lane(self, lines, image_shape):
        lanes_info = self.get_lines_info(lines)
        if (lanes_info.size > 0):
            lane_params = np.average(lanes_info, axis=0)
            y1 = image_shape[0]
            x1 = (y1 - lane_params[1]) / lane_params[0]
            y2 = image_shape[0] * 5.4 / 9
            x2 = (y2 - lane_params[1]) / lane_params[0]
            return np.array([[x1, y1, x2, y2]])
        return np.array([[0, 0, 0, 0]])

    def determine_lane(self, lines, image_shape, filter):
        newLines = self.filter_lanes(lines, image_shape, filter)
        line = self.calculate_lane(newLines, image_shape)
        return line.astype(int)

    def draw_lines(self, img, lines, color, thickness):
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    def hough_lines(self, img, channel = cv2.COLOR_BGR2GRAY ,
                    rho=1, theta=np.pi / 180, threshold=15, min_line_len=15, max_line_gap=15):
        """
        `img` is the original image

        Returns hough lines
        """
        newImage = self.pre_process_image(img, channel)
        return cv2.HoughLinesP(newImage, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                                maxLineGap=max_line_gap)

    def pre_process_image(self, image, channel = cv2.COLOR_BGR2GRAY, kernel_size=5,
                          low_threshold = 50, high_threshold = 150):
        """
        turns original image into a masked canny image that can be fed into hough_lines
        """

        imshape = image.shape
        gray_image = self.convertImageToChannel(image, channel)
        blur_gray = self.gaussian_blur(gray_image, kernel_size)
        canny_img = self.canny(blur_gray, low_threshold, high_threshold)
        vertices = np.array([[(0, imshape[0]), (imshape[1] * 0.4, 5.5 / 9 * imshape[0]),
                              (imshape[1] * 0.6, 5.5 / 9 * imshape[0]), (imshape[1], imshape[0])]], dtype=np.int32)
        return self.region_of_interest(canny_img, vertices)



    def isEmpty(self, line):
        #this is dumb but it seems comparing arrays directly throws a fit
        return line[0][0]== 0 and line[0][1] == 0 and line[0][2] == 0 and line[0][3] == 0

    def isTooFar(self, candidate, existing):
        #don't want the tip of the line to be too far
        return abs(candidate[0][0] - existing[0][0]) > 40 or abs(candidate[0][2] - existing[0][2]) > 40

    def isInvalidCandidate(self, candidate, existing):
        return  self.isEmpty(candidate) or self.isTooFar(candidate, existing)

    def determine_best_lines(self, img):
        #use matches from gray image first
        hough_lines = self.hough_lines(img, cv2.COLOR_BGR2GRAY)
        leftLane = self.determine_lane(hough_lines, img.shape,
                                       lambda line, image_shape: ld.is_left_line(line, image_shape))
        rightLane = self.determine_lane(hough_lines, img.shape,
                                        lambda line, image_shape: ld.is_right_line(line, image_shape))


        # initialize the values
        if(self.isEmpty(self.curentLines[0])):
            self.curentLines[0] = leftLane

        if(self.isEmpty(self.curentLines[1])):
             self.curentLines[1] = rightLane
        #
        # # use another color space to find better candidates if needed
        if(self.isInvalidCandidate(leftLane, self.curentLines[0]) or self.isInvalidCandidate(rightLane, self.curentLines[1])):
            hough_lines = self.hough_lines(img, cv2.COLOR_BGR2HSV)

            if(self.isInvalidCandidate(leftLane, self.curentLines[0])):
                leftLane = self.determine_lane(hough_lines, img.shape,
                                           lambda line, image_shape: ld.is_left_line(line, image_shape))

            if(self.isInvalidCandidate(rightLane, self.curentLines[1])):
                rightLane = self.determine_lane(hough_lines, img.shape,
                                            lambda line, image_shape: ld.is_right_line(line, image_shape))

        self.curentLines = [(leftLane * self.smoothingFactor + (1-self.smoothingFactor) * self.curentLines[0]).astype(int),
                            (rightLane * self.smoothingFactor + (1 - self.smoothingFactor) * self.curentLines[1]).astype(int)]
        return self.curentLines

    def process_image(self, img):
        lines = self.determine_best_lines(img)
        self.draw_lines(img, lines, [255, 0, 0], 7)
        result = img
        return result



import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def show_image(img):
    plt.imshow(img, cmap='gray')
    plt.show()


import os

images_dir = 'test_images/'
output_dir = 'output_images/'
files = [f for f in os.listdir(images_dir) if f.endswith(".jpg")]

ld = LineDetector()

try:
    os.stat(output_dir)
except:
    os.mkdir(output_dir)

for file in files:
    image = mpimg.imread(images_dir + file)
    mpimg.imsave(output_dir + file, ld.process_image(image))

from moviepy.editor import VideoFileClip
input_movies_dir = "test_videos/"
output_movies_dir = "output_videos/"

try:
    os.stat(output_movies_dir)
except:
    os.mkdir(output_movies_dir)

def processVideo(lineDetector, filename):
    clip1 = VideoFileClip(input_movies_dir + filename)
    white_clip = clip1.fl_image(lineDetector.process_image)
    white_clip.write_videofile(output_movies_dir + filename, audio=False)

processVideo(ld, "solidWhiteRight.mp4")
processVideo(ld, "solidYellowLeft.mp4")
processVideo(ld, "challenge.mp4")



