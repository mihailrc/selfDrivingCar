import numpy as np
import cv2

import matplotlib

matplotlib.use('TkAgg')

from advancedLaneLines.sdc import utils

class Camera():
    def __init__(self, img_shape):
        # camera distortion matrices
        [self.mtx, self.dist] = utils.read_calibration_data()

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
        undist_img = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        return undist_img

    def warp_image(self, img, img_size):
        return cv2.warpPerspective(img, self.M, img_size, flags=cv2.INTER_LINEAR)

    def unwarp_image(self, img, img_size):
        return cv2.warpPerspective(img, self.Minv, img_size, flags=cv2.INTER_LINEAR)

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
        img = self.gaussian_blur(img, 3)
        undistorted = self.undistort_image(img)
        img_size = img.shape
        bird_eye = self.warp_image(undistorted, (img_size[1], img_size[0]))
        binary, gradient_binary = self.combined_binary(bird_eye)
        return [undistorted, bird_eye, binary, gradient_binary]