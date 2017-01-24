import numpy as np
import cv2
import glob

import pickle as pickle

import matplotlib

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