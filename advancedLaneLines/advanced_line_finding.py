import numpy as np
import pandas
import os
import json
import cv2
import glob

import pickle as pickle

import matplotlib
import scipy.misc

import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec

matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

def calibrateCamera(imagesPath, rows, cols):
    images = glob.glob(imagesPath)

    objpoints = []
    imgpoints = []

    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)

    for imagePath in images:
        img = mpimg.imread(imagePath)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (cols, rows), None)

        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return mtx, dist

def save_calibration_data():
    rows = 6
    cols = 9
    images = 'camera_cal/calibration*.jpg'
    mtx,dist = calibrateCamera(images,rows,cols)

    pickle.dump( [mtx,dist], open( "camera_calibration.pkl", "wb" ) )

def read_calibration_data():
    [mtx,dist] = pickle.load(open("camera_calibration.pkl", "rb"))
    return mtx,dist

def plot_images(images, rows, cols, titles):
    # gs = gridspec.GridSpec(rows, cols, top=1.0, bottom=.0, right=.7, left=0., hspace=0.3,
    #                        wspace=0.1)
    gs = gridspec.GridSpec(rows, cols)
    for index, g in enumerate(gs):
        ax = plt.subplot(g)
        img = images[index]
        ax.imshow(img)
        plt.imshow(img, cmap=plt.get_cmap('gray'))
        ax.set_xticks([])
        ax.set_yticks([])
        plt.title(titles[index])

    plt.show()

def draw_lines(img, points, color):
    points = np.int_(points)
    intervals = points.shape[1] - 1
    for i in range(intervals):
        x1 = points[0][i][0]
        y1 = points[0][i][1]
        x2 = points[0][i + 1][0]
        y2 = points[0][i + 1][1]
        cv2.line(img, (x1, y1), (x2, y2),color,50)

def moving_average(img, n=3):
    ret = np.cumsum(img, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def overlap_images(first, second):
    return cv2.addWeighted(first, 1, second, 0.5, 0)


class Camera():
    def __init__(self, img_shape):
        # camera distortion matrices
        [self.mtx, self.dist] = read_calibration_data()

        # perspective transformation matrices
        self.M,self.Minv = self.get_transformation_matrices(img_shape)

    def get_transformation_matrices(self, img_shape):
        # top_window = np.uint(img_shape[0]/1.6)
        top_window = np.uint(img_shape[0] / 1.5)
        # src = np.float32([[0,img_shape[0]],[img_shape[1],img_shape[0]],[0.55*img_shape[1],top_window],[0.45*img_shape[1],top_window]])
        src = np.float32([[0, img_shape[0]], [img_shape[1], img_shape[0]],
                          [0.6 * img_shape[1], top_window], [0.4 * img_shape[1], top_window]])

        dst = np.float32([[0, img_shape[0]], [img_shape[1], img_shape[0]],
                          [img_shape[1], 0], [0, 0]])
        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
        return M, Minv

    def gaussian_blur(self, img, kernel=5):
        blur = cv2.GaussianBlur(img, (kernel, kernel), 0)
        return blur

    def undistort_image(self, img):
        undist_img = cv2.undistort(img, self.mtx, self.dist, None, mtx)
        return undist_img

    def warp_image(self, img, img_size):
        return cv2.warpPerspective(img, self.M, img_size, flags=cv2.INTER_LINEAR)

    def unwarp_image(self, img, img_size):
        return cv2.warpPerspective(img, self.Minv, img_size, flags=cv2.INTER_LINEAR)


# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #[y, x, polynomial coefficients]
        self.current_fit = [None, None, None]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None
        #current line image
        self.currentLine = None

class LineDetector():
    def __init__(self):
        self.camera = Camera((720, 1280, 3))
        self.leftLine = Line()
        self.rightLine = Line()
        self.skipped_frames = 0

    def color_binary(self, img):
        # Convert to HLS color space and separate the S channel
        # Note: img is the undistorted image
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        s_channel = hls[:, :, 2]

        # Threshold color channel
        s_binary = np.zeros_like(s_channel)
        s_thresh = cv2.inRange(s_channel.astype('uint8'), 175, 255)

        s_binary[(s_thresh == 255)] = 1
        return s_binary

    def gradient_binary(self, img):
        # Grayscale image
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Sobel x
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)  # Take the derivative in x
        abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

        # Threshold x gradient
        thresh_min = 30
        thresh_max = 150
        sxbinary = np.zeros_like(scaled_sobel)
        # sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
        retval, sxthresh = cv2.threshold(scaled_sobel, 30, 150, cv2.THRESH_BINARY)
        sxbinary[(sxthresh >= thresh_min) & (sxthresh <= thresh_max)] = 1
        return sxbinary

    def combined_binary(self, img):
        color_binary = self.color_binary(img)
        gradient_binary = self.gradient_binary(img)
        # Combine the two binary thresholds
        combined_binary = np.zeros_like(gradient_binary)
        combined_binary[(color_binary == 1) | (gradient_binary == 1)] = 1
        return combined_binary, gradient_binary

    def process_image(self, img):
        img = self.camera.gaussian_blur(img, 3)
        undistorted = self.camera.undistort_image(img)
        img_size = img.shape
        bird_eye = self.camera.warp_image(undistorted, (img_size[1], img_size[0]))

        binary, gradient_binary = self.combined_binary(bird_eye)
        return [undistorted, bird_eye, binary, gradient_binary]


    def extract_lines(self, binary):
        max_width = 150
        mov_filtsize = 20
        mean_lane = np.mean(binary[:, :], axis=0)
        mean_lane = moving_average(mean_lane, mov_filtsize)

        # plt.plot(mean_lane > .075)
        # plt.plot(mean_lane)
        # plt.xlabel('image x')
        # plt.ylabel('mean intensity')
        # plt.show()

        img_size = binary.shape
        above_threshold = np.argwhere(mean_lane>.075)

        left = np.copy(binary)
        above_threshold_left = above_threshold[above_threshold<img_size[1]/2.]

        valid_lines = True

        if(above_threshold_left.size>0):
            above_threshold_min = np.min(above_threshold_left)
            above_threshold_max = np.max(above_threshold_left)

            if(above_threshold_max-above_threshold_min > max_width):
                valid_lines = False

            left[:,0:above_threshold_min] = 0
            left[:,above_threshold_max:] = 0

        right = np.copy(binary)
        above_threshold_right = above_threshold[above_threshold>img_size[1]/2.]

        if(above_threshold_right.size > 0):
            above_threshold_min = np.min(above_threshold_right)
            above_threshold_max = np.max(above_threshold_right)
            if(above_threshold_max - above_threshold_min > max_width):
                valid_lines = False
            right[:,0:above_threshold_min] = 0
            right[:,above_threshold_max:img_size[1]] = 0

        # if left lane is found but right lane is not found use offset from left lane
        if(above_threshold_left.size>0 and above_threshold_right.size == 0):
            above_threshold_min = np.min(above_threshold_left) + 900 - 30
            above_threshold_max = np.max(above_threshold_left) + 900  + 30
            right[:, 0:above_threshold_min] = 0
            right[:, above_threshold_max:img_size[1]] = 0

        if (above_threshold_right.size > 0 and above_threshold_left.size == 0):
            above_threshold_min = np.min(above_threshold_right) - 900 - 30
            above_threshold_max = np.max(above_threshold_right) - 900 + 30
            left[:, 0:above_threshold_min] = 0
            left[:, above_threshold_max:img_size[1]] = 0

        if(not valid_lines):
            left = None
            right = None

        return left, right

    def fit_line(self, line):
        points = np.nonzero(line)
        all_x = points[0]
        all_y = points[1]
        if(all_x.size>0):
            fit = np.polyfit(all_x, all_y, 2)
            y = np.arange(11) * line.shape[0] / 10
            x = fit[0] * y ** 2 + fit[1] * y + fit[2]
        else:
            x = None
            y = None
            fit = None
        return y,x,fit

    def calculate_points(self, fit, size):
        y = np.arange(11) * size / 10
        x = fit[0] * y ** 2 + fit[1] * y + fit[2]
        return x,y

    def calculate_curvature_pixels(self, fit, y):
        return (1 + (2 * fit[0] * y + fit[1]) ** 2) ** 1.5 / 2 / fit[0]

    def calculate_curvature_meters(self, all_y, all_x, y):
        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 900  # meteres per pixel in x dimension
        # points = np.nonzero(line)
        # all_x = points[0]
        # all_y = points[1]
        if(all_x.size>0):
            fit = np.polyfit(all_y*ym_per_pix, all_x*xm_per_pix, 2)
            curvature = ((1 + (2 * fit[0] * y + fit[1]) ** 2) ** 1.5) \
                            / np.absolute(2 * fit[0])
        else:
            curvature = None
        return curvature

    def has_small_deviation(self, new_x, current_x):
        if(new_x is None):
            return False
        else:
            return abs(new_x[10]-current_x[10])<40

    def are_lines_parallel(self, left_x, right_x):
        if(left_x is None or right_x is None):
            return False
        else:
            return abs(right_x[10] - left_x[10] - 900) < 150 \
                   and  abs(right_x[5] - left_x[5] - 900) < 150 \
                   and abs(right_x[0] - left_x[0] - 900) < 150

    def is_frame_valid(self, left_x, right_x):
        return self.has_small_deviation(left_x, self.leftLine.current_fit[1]) \
               and self.has_small_deviation(right_x, self.rightLine.current_fit[1]) \
               and self.are_lines_parallel(left_x, right_x)

    # extract lines from combined binary. If that fails try gradient binary only. If that still fails return binary
    def extract_best_lines(self, binary, gradient_binary):
        left, right = self.extract_lines(binary)

        if (left is None):
            left, right = self.extract_lines(gradient_binary)

        if (left is None):
            left, right = binary, binary

        return left, right

    def has_current_fit(self):
        return (self.leftLine.current_fit[0] is not None) and (self.rightLine.current_fit[0] is not None)

    def fast_extract(self, binary):
        img_size = binary.shape
        width = 75

        left = np.copy(binary)
        left_top_x = self.leftLine.current_fit[1][3]
        left_bottom_x = self.leftLine.current_fit[1][7]
        left[:img_size[0]/2, 0:left_top_x-width] = 0
        left[:img_size[0] / 2, left_top_x + width:] = 0
        left[img_size[0] / 2:, 0:left_bottom_x - width] = 0
        left[img_size[0] / 2:, left_bottom_x + width:] = 0

        right = np.copy(binary)
        right_top_x = self.rightLine.current_fit[1][3]
        right_bottom_x = self.rightLine.current_fit[1][7]
        right[:img_size[0] / 2, 0:right_top_x - width] = 0
        right[:img_size[0] / 2, right_top_x + width:] = 0
        right[img_size[0] / 2:, 0:right_bottom_x - width] = 0
        right[img_size[0] / 2:, right_bottom_x + width:] = 0

        return left, right

    #takes the bird eye view, draws lines on it then performes a perspective transform
    def draw_area_between_lines(self, bird_eye):
        bird_eye_fill = np.zeros_like(bird_eye).astype(np.uint8)

        left_points = np.array([np.transpose(np.vstack([self.leftLine.current_fit[1], self.leftLine.current_fit[0]]))])
        right_points = np.array(
            [np.flipud(np.transpose(np.vstack([self.rightLine.current_fit[1], self.rightLine.current_fit[0]])))])
        lanes_points = np.hstack((left_points, right_points))

        fill_color = (0, 255, 0)
        cv2.fillPoly(bird_eye_fill, np.int_([lanes_points]), fill_color)

        lane_color = (255, 255, 0)
        draw_lines(bird_eye_fill, np.int_(left_points), lane_color)
        draw_lines(bird_eye_fill, np.int_(right_points), lane_color)

        return bird_eye_fill, self.camera.unwarp_image(bird_eye_fill, (bird_eye.shape[1], bird_eye.shape[0]))

    def draw_information(self, image):
        left_curvature_m = self.calculate_curvature_meters(self.leftLine.current_fit[0],self.leftLine.current_fit[1], image.shape[0])
        right_curvature_m = self.calculate_curvature_meters(self.rightLine.current_fit[0],self.rightLine.current_fit[1], image.shape[0])
        left_curvature = self.calculate_curvature_pixels(self.leftLine.current_fit[2], image.shape[0])
        right_curvature = self.calculate_curvature_pixels(self.rightLine.current_fit[2], image.shape[0])
        font = cv2.FONT_HERSHEY_COMPLEX
        curvature_string_m = 'Curvature M: Left = {0}, Right = {1}'.format(np.round(left_curvature_m, 2),
                                                                       np.round(right_curvature_m, 2))
        curvature_string = 'Curvature: Left = {0}, Right = {1}'.format(np.round(left_curvature, 2),
                                                                       np.round(right_curvature, 2))
        cv2.putText(image, curvature_string, (30, 60), font, 1, (0, 255, 0), 2)
        cv2.putText(image, curvature_string_m, (30, 90), font, 1, (0, 255, 0), 2)
        offset = ((self.leftLine.current_fit[1][10] + self.rightLine.current_fit[1][10]) / 2 - image.shape[1] / 2) * 3.7 / 900
        offset_string = 'Lane deviation: {} cm.'.format(np.round(offset * 100, 2))
        cv2.putText(image, offset_string, (30, 120), font, 1, (0, 255, 0), 2)
        return image

    def find_best_fit(self, binary, gradient_binary):
        # find best fit
        # 1. if current fit found - fast extract from binary
        # 2. if frame not valid do full scan
        # 3. if this fails use current frame if less than 5 frames
        # 4. otherwise use results from full scan
        if (self.has_current_fit()):
            left, right = self.fast_extract(binary)
            left_y, left_x, left_fit = self.fit_line(left)
            right_y, right_x, right_fit = self.fit_line(right)

            if (not self.is_frame_valid(left_x, right_x)):
                print('doing full scan')
                left, right = self.extract_best_lines(binary, gradient_binary)
                left_y, left_x, left_fit = self.fit_line(left)
                right_y, right_x, right_fit = self.fit_line(right)
            else:
                self.skipped_frames = 0
            if (not self.is_frame_valid(left_x, right_x) and self.skipped_frames < 5):
                left = self.leftLine.currentLine
                right = self.rightLine.currentLine
                [left_y, left_x, left_fit] = self.leftLine.current_fit
                [right_y, right_x, right_fit] = self.rightLine.current_fit
                self.skipped_frames = self.skipped_frames + 1
            else:
                self.skipped_frames = 0
        else:
            left, right = self.extract_best_lines(binary, gradient_binary)
            left_y, left_x, left_fit = self.fit_line(left)
            right_y, right_x, right_fit = self.fit_line(right)

        # refresh line fields
        self.leftLine.currentLine = left
        self.rightLine.currentLine = right

        if (self.has_current_fit()):
            # apply smoothing
            sf = 0.2
            self.leftLine.current_fit[1] = sf * left_x + (1 - sf) * self.leftLine.current_fit[1]
            self.leftLine.current_fit[2] = sf * left_fit + (1 - sf) * self.leftLine.current_fit[2]
            self.rightLine.current_fit[1] = sf * right_x + (1 - sf) * self.rightLine.current_fit[1]
            self.rightLine.current_fit[2] = sf * right_fit + (1 - sf) * self.rightLine.current_fit[2]
        else:
            self.leftLine.current_fit = [left_y, left_x, left_fit]
            self.rightLine.current_fit = [right_y, right_x, right_fit]

    def process_image_2(self,img):
        [undistorted, bird_eye, binary, gradient_binary] = self.process_image(img)

        self.find_best_fit(binary, gradient_binary)

        # if(not self.are_lines_parallel(left_x, right_x) and left_x is not None and right_x is not None):
        #     message = "Lines not parallel bottom {0} middle {1} top {2}".format(right_x[10] - left_x[10] - 900,
        #                                                                     right_x[5] - left_x[5] - 900,
        #                                                                     right_x[0] - left_x[0] - 900)
        #     print(message)


        bird_eye_lines, unwarped = self.draw_area_between_lines(bird_eye)

        image_with_lanes  = overlap_images(undistorted, unwarped)

        result = self.draw_information(image_with_lanes)

        return result

mtx,dist = read_calibration_data()

# images = ['test_images/straight_lines1.jpg','test_images/straight_lines2.jpg', 'test_images/test1.jpg',
#           'test_images/test2.jpg','test_images/test3.jpg','test_images/test4.jpg','test_images/test5.jpg',
#           'test_images/test6.jpg']

images = ['test_images/test4.jpg']

lineDetector = LineDetector()

# for image in images:
#     print(image)
#     img = mpimg.imread(image)
#     [undistorted, bird_eye, binary, gradient_binrary] = lineDetector.process_image(img)
#     left, right = lineDetector.extract_lines(binary)
#     if(left is None):
#         left, right = lineDetector.extract_lines(gradient_binrary)
#     if(left is None):
#         left, right = binary, binary
#
#     print(left.shape, right.shape)
#     processed = lineDetector.process_image_2(img)
#     plot_images([bird_eye, binary, lineDetector.leftLine.currentLine, lineDetector.rightLine.currentLine], 2, 2, ['Brdie', 'Binary', 'Left', 'Right'])
#     plot_images([processed], 1,1,['Result'])

from moviepy.editor import VideoFileClip
#
input_movies_dir = "./"
output_movies_dir = "output_videos/"

try:
    os.stat(output_movies_dir)
except:
    os.mkdir(output_movies_dir)

def processVideo(filename):
    clip1 = VideoFileClip(input_movies_dir + filename)
    # clip1.save_frame("test_images/test12.jpg", t='00:00:42.12')
    output_clip = clip1.fl_image(lineDetector.process_image_2)
    output_clip.write_videofile(output_movies_dir + filename, audio=False)
#
# processVideo("project_video.mp4")

img = mpimg.imread('test_images/test4.jpg')

gaussian_blur = lineDetector.camera.gaussian_blur(img)

[undistorted, bird_eye, binary, gradient_binary] = lineDetector.process_image(img)

color_binary = lineDetector.color_binary(bird_eye)

left, right = lineDetector.extract_lines(binary)

lineDetector.find_best_fit(binary, gradient_binary)

bird_eye_lines, bird_eye_unwraped = lineDetector.draw_area_between_lines(bird_eye)

overlaped = overlap_images(undistorted, bird_eye_unwraped)

result = lineDetector.draw_information(overlaped)


# plot_images([img,undistorted, bird_eye, color_binary, gradient_binary,
#              binary, left, right,bird_eye_lines, bird_eye_unwraped, overlaped, overlaped], 4 , 3,
#             ['Original',  'Undistorted', 'Bird Eye', 'Color Binary', 'Gradient Binary',
#              'Combined Binary', 'Left Line', 'Right Line', 'Lines Fit', 'Lines Fit Unwarped', 'Overlaped', 'Result'])

mpimg.imsave('output_images/original.jpg', img)
# mpimg.imsave('output_images/gaussian_blur.jpg', gaussian_blur)
# mpimg.imsave('output_images/undistorted.jpg', undistorted)
# mpimg.imsave('output_images/bird_eye.jpg', bird_eye)
# mpimg.imsave('output_images/combined_binary.jpg', binary)
# mpimg.imsave('output_images/gradient_binary.jpg', gradient_binary)
# mpimg.imsave('output_images/color_binary.jpg', color_binary)
# mpimg.imsave('output_images/left.jpg', left)
# mpimg.imsave('output_images/right.jpg', right)
# mpimg.imsave('output_images/bird_eye_lines.jpg', bird_eye_lines)
# mpimg.imsave('output_images/bird_eye_unwraped.jpg', bird_eye_unwraped)
# mpimg.imsave('output_images/overlaped.jpg', overlaped)
mpimg.imsave('output_images/result.jpg', result)






