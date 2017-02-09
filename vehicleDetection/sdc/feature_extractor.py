import numpy as np
import cv2
from skimage.feature import hog
import scipy.misc as scimsc

def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_windows = np.int((xspan - xy_window[0]) / nx_pix_per_step)
    ny_windows = np.int((yspan - xy_window[1]) / ny_pix_per_step)

    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    #     Note: you could vectorize this step, but in practice
    #     you'll be considering windows one by one with your
    #     classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs * nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys * ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return features

# Define a function to compute color histogram features
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0],channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

#Define a function to return HOG features for all channels
def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                     feature_vec=True):
    # Call with two outputs if vis==True
    channel1_hog = hog(img[:, :, 0], orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                    cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                    visualise=False, feature_vector=feature_vec)
    channel2_hog = hog(img[:, :, 1], orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                    cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                    visualise=False, feature_vector=feature_vec)
    channel3_hog = hog(img[:, :, 2], orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                    cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                    visualise=False, feature_vector=feature_vec)
    return channel1_hog, channel2_hog, channel3_hog

# Define a function to extract features from a single image window.
# Only extracts color histogram and HOG features
def extract_features(image, color_space='RGB',
                     hist_bins=32, hist_range=(0, 256), orient=9,
                     pix_per_cell=8, cell_per_block=2):
    # apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif color_space == 'LAB':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCR_CB)
    else:
        feature_image = np.copy(image)

    hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
    channel1_hog, channel2_hog, channel3_hog = get_hog_features(feature_image, orient,
                                                                           pix_per_cell, cell_per_block,
                                                                           feature_vec=True)
    features = np.concatenate((hist_features, channel1_hog, channel2_hog, channel3_hog))

    # Return list of feature vectors
    return features


def extract_features_from_files(imgs, color_space='RGB', spatial_size=(32, 32),
                                hist_bins=32, hist_range=(0, 256), orient=9,
                                pix_per_cell=8, cell_per_block=2):
    # Create a list to append feature vectors to
    features = []
    for file in imgs:
        image = scimsc.imread(file)
        image_features = extract_features(image, color_space=color_space,
                                          hist_bins=hist_bins, hist_range=hist_range, orient=orient,
                                          pix_per_cell=pix_per_cell, cell_per_block=cell_per_block)

        features.append(image_features)
    #return features for all images
    return features



