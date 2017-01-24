import numpy as np
import cv2

from advancedLaneLines.sdc import camera
from advancedLaneLines.sdc import line
from advancedLaneLines.sdc import utils

class LineDetector():
    def __init__(self):
        self.camera = camera.Camera((720, 1280, 3))
        self.leftLine = line.Line()
        self.rightLine = line.Line()
        self.skipped_frames = 0

    def extract_lines(self, binary):
        max_width = 150
        filter_size = 20
        mean_lane = np.mean(binary[:, :], axis=0)
        mean_lane = utils.moving_average(mean_lane, filter_size)

        # plt.plot(mean_lane > .075)
        # plt.plot(mean_lane)
        # plt.xlabel('X')
        # plt.ylabel('Moving Average')
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

        # extract lines from combined binary. If that fails try gradient binary only. If that still fails return binary

    def extract_best_lines(self, binary, gradient_binary):
        left, right = self.extract_lines(binary)

        if (left is None):
            left, right = self.extract_lines(gradient_binary)

        if (left is None):
            left, right = binary, binary

        return left, right

    #uses existing fit to identify line pixels
    def fast_extract(self, binary):
        img_size = binary.shape
        width = 75

        #split the image into top and bottom section
        left = np.copy(binary)
        #get x position in the middle of the top section
        left_top_x = self.leftLine.current_fit[1][3]
        #get x position in the middle of the bottom section
        left_bottom_x = self.leftLine.current_fit[1][7]
        #retain pixels to left and right of left_top_x and left_bottom_x
        left[:img_size[0] / 2, 0:left_top_x - width] = 0
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

    def has_current_fit(self):
        return (self.leftLine.current_fit[0] is not None) and (self.rightLine.current_fit[0] is not None)

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
        utils.draw_lines(bird_eye_fill, np.int_(left_points), lane_color)
        utils.draw_lines(bird_eye_fill, np.int_(right_points), lane_color)

        return bird_eye_fill, self.camera.unwarp_image(bird_eye_fill, (bird_eye.shape[1], bird_eye.shape[0]))

    def calculate_vehicle_offset(self, image):
        return ((self.leftLine.current_fit[1][10] + self.rightLine.current_fit[1][10]) / 2 - image.shape[1] / 2) * 3.7 / 900

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
        offset = self.calculate_vehicle_offset(image)
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

    def process_image(self,img):
        [undistorted, bird_eye, binary, gradient_binary] = self.camera.process_image(img)

        self.find_best_fit(binary, gradient_binary)

        # if(not self.are_lines_parallel(left_x, right_x) and left_x is not None and right_x is not None):
        #     message = "Lines not parallel bottom {0} middle {1} top {2}".format(right_x[10] - left_x[10] - 900,
        #                                                                     right_x[5] - left_x[5] - 900,
        #                                                                     right_x[0] - left_x[0] - 900)
        #     print(message)


        bird_eye_lines, unwarped = self.draw_area_between_lines(bird_eye)

        image_with_lanes  = utils.overlap_images(undistorted, unwarped)

        result = self.draw_information(image_with_lanes)

        return result