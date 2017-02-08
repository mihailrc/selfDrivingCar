import matplotlib.image as mpimg
import numpy as np
import cv2
from skimage.feature import hog
import scipy.misc as scimsc


class FeatureExtractor:

    def __init__(self):
        self.image = []
        self.x_start_stop = [None, None]
        self.y_start_stop=[None, None]
        self.xy_window=(64, 64)
        #step in multiples of eight. The step is applied after resizing windows so we get 64x64 images
        self.step_size = None

    #to do
    # 1. extract region of interest from image
    # 2. resize image so windows will be 64x64
    # 3. hog for the whole image
    # 4. extract image given (x,y) coordinates
    # 5. extract hog given (x,y) coordinates
    # 6. extract features given (x,y) coordinates


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

    # print('winds', nx_windows, ny_windows, xy_window)
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

# # Define a function to return HOG features and visualization
# def get_hog_features(img, orient, pix_per_cell, cell_per_block,
#                      vis=False, feature_vec=True):
#     # Call with two outputs if vis==True
#     if vis == True:
#         features, hog_image = hog(img, orientations=orient,
#                                   pixels_per_cell=(pix_per_cell, pix_per_cell),
#                                   cells_per_block=(cell_per_block, cell_per_block),
#                                   transform_sqrt=True,
#                                   visualise=vis, feature_vector=feature_vec)
#         return features, hog_image
#     # Otherwise call with one output
#     else:
#         features = hog(img, orientations=orient,
#                        pixels_per_cell=(pix_per_cell, pix_per_cell),
#                        cells_per_block=(cell_per_block, cell_per_block),
#                        transform_sqrt=True,
#                        visualise=vis, feature_vector=feature_vec)
#         return features
#
#
# # Define a function to compute binned color features
# def bin_spatial(img, size=(32, 32)):
#     # Use cv2.resize().ravel() to create the feature vector
#     features = cv2.resize(img, size).ravel()
#     # Return the feature vector
#     return features
#
#
# # Define a function to compute color histogram features
# # NEED TO CHANGE bins_range if reading .png files with mpimg!
# def color_hist(img, nbins=32, bins_range=(0, 256)):
#     # Compute the histogram of the color channels separately
#     channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
#     channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
#     channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
#     # Concatenate the histograms into a single feature vector
#     hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
#     # Return the individual histograms, bin_centers and feature vector
#     return hist_features
#
#
# # Define a function to extract features from a single image window
# # This function is very similar to extract_features()
# # just for a single image rather than list of images
# def extract_features(img, color_space='RGB', spatial_size=(32, 32),
#                      hist_bins=32, orient=9,
#                      pix_per_cell=8, cell_per_block=2, hog_channel='ALL',
#                      spatial_feat=False, hist_feat=True, hog_feat=True):
#     # 1) Define an empty list to receive features
#     img_features = []
#     # 2) Apply color conversion if other than 'RGB'
#     if color_space != 'RGB':
#         if color_space == 'HSV':
#             feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
#         elif color_space == 'LUV':
#             feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
#         elif color_space == 'HLS':
#             feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
#         elif color_space == 'YUV':
#             feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
#         elif color_space == 'YCrCb':
#             feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
#     else:
#         feature_image = np.copy(img)
#     # 3) Compute spatial features if flag is set
#     if spatial_feat == True:
#         spatial_features = bin_spatial(feature_image, size=spatial_size)
#         # 4) Append features to list
#         img_features.append(spatial_features)
#     # 5) Compute histogram features if flag is set
#     if hist_feat == True:
#         hist_features = color_hist(feature_image, nbins=hist_bins)
#         # 6) Append features to list
#         img_features.append(hist_features)
#     # 7) Compute HOG features if flag is set
#     if hog_feat == True:
#         if hog_channel == 'ALL':
#             hog_features = []
#             for channel in range(feature_image.shape[2]):
#                 hog_features.extend(get_hog_features(feature_image[:, :, channel],
#                                                      orient, pix_per_cell, cell_per_block,
#                                                      vis=False, feature_vec=True))
#         else:
#             hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
#                                             pix_per_cell, cell_per_block, vis=False, feature_vec=True)
#         # 8) Append features to list
#         img_features.append(hog_features)
#
#     # 9) Return concatenated array of features
#     return np.concatenate(img_features).reshape(1, -1)[0]
#
# # Define a function to extract features from a list of images
# # Have this function call bin_spatial() and color_hist()
# def extract_features_from_files(imgs, color_space='RGB', spatial_size=(32, 32),
#                      hist_bins=32, orient=9,
#                      pix_per_cell=8, cell_per_block=2, hog_channel=0,
#                      spatial_feat=False, hist_feat=True, hog_feat=True):
#     # Create a list to append feature vectors to
#     features = []
#     # Iterate through the list of images
#     for file in imgs:
#         image = scimsc.imread(file)
#         file_features = extract_features(image, color_space, spatial_size, hist_bins,
#                                          orient, pix_per_cell, cell_per_block, hog_channel,
#                                          spatial_feat, hist_feat, hog_feat)
#         features.append(file_features)
#     # Return list of feature vectors
#     return features

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

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                     feature_vec=True):
    # Call with two outputs if vis==True
    featuresR = hog(img[:, :, 0], orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                    cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                    visualise=False, feature_vector=feature_vec)
    featuresG = hog(img[:, :, 1], orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                    cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                    visualise=False, feature_vector=feature_vec)
    featuresB = hog(img[:, :, 2], orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                    cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                    visualise=False, feature_vector=feature_vec)
    return featuresR, featuresG, featuresB

    # Define a function to extract features from a list of images
    # Have this function call bin_spatial() and color_hist()


def extract_features(image, color_space='RGB', spatial_size=(32, 32),
                     hist_bins=32, hist_range=(0, 256), orient=9,
                     pix_per_cell=8, cell_per_block=2, hog_channel=0):
    # If image is RGBA, conver to RGB
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

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
    # Apply bin_spatial() to get spatial color features
    # spatial_features = self.bin_spatial(feature_image, size=spatial_size)
    # Apply color_hist() also with a color space option now
    hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
    # start = time.time()
    # Call get_hog_features() with vis=False, feature_vec=True
    hog_features_R, hog_features_G, hog_features_B = get_hog_features(feature_image, orient,
                                                                           pix_per_cell, cell_per_block,
                                                                           feature_vec=True)
    # end = time.time()
    # print('hog only {0}'.format((end-start)))
    # Append the new feature vector to the features list
    # features = np.concatenate((spatial_features, hist_features, hog_features_R, hog_features_G, hog_features_B))
    # features = np.concatenate((spatial_features, hist_features, hog_features))
    features = np.concatenate((hist_features, hog_features_R, hog_features_G, hog_features_B))
    # features = np.concatenate((hog_features_R, hog_features_G, hog_features_B))

    # Return list of feature vectors
    return features


def extract_features_from_files(imgs, color_space='RGB', spatial_size=(32, 32),
                                hist_bins=32, hist_range=(0, 256), orient=9,
                                pix_per_cell=8, cell_per_block=2):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = scimsc.imread(file)

        image_features = extract_features(image, color_space=color_space, spatial_size=spatial_size,
                                          hist_bins=hist_bins, hist_range=hist_range, orient=orient,
                                          pix_per_cell=pix_per_cell, cell_per_block=cell_per_block)

        features.append(image_features)

    return features



