import matplotlib

matplotlib.use('TkAgg')

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

import cv2
import glob
from moviepy.editor import VideoFileClip
from vehicleDetection.sdc.feature_extractor import *
import time

from functools import partial
from vehicleDetection.sdc.car_tracker import *
import pickle
import matplotlib.image as mpimg

def plot_images(images, rows, cols, titles):
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

# slides windows over the original image to find car boxes. It returns all boxes that
# are classified as cars
def find_car_boxes(img, windows, clasifier, scaler):
    car_boxes = []
    for window in windows:
        windows_image = img[window[0][1]:window[1][1], window[0][0]:window[1][0]]
        windows_image_resized = cv2.resize(windows_image, (64, 64))

        windows_image_features = extract_features(windows_image_resized, color_space='YCrCb')
        windows_image_features = windows_image_features.reshape(1, -1)

        scaled = scaler.transform(windows_image_features)
        if clasifier.predict(scaled) == 1:
            confidence_score = clasifier.predict_proba(scaled)
            car_boxes.append((window, confidence_score[0][1]))
    return car_boxes

def get_all_sliding_windows(img, overlap=0.8):
    windows_128 = slide_window(img, x_start_stop=[None, None], y_start_stop=[380, 650],
                           xy_window=(128, 128), xy_overlap=(overlap, overlap))
    windows_64 = slide_window(img, x_start_stop=[280, 1000], y_start_stop=[380, 520],
                           xy_window=(64, 64), xy_overlap=(overlap, overlap))
    # windows_64 = slide_window(img, x_start_stop=[None, None], y_start_stop=[380, 520],
    #                           xy_window=(64, 64), xy_overlap=(overlap, overlap))
    return windows_128 + windows_64

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=1):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

def process_image(clasifier, scaler,car_tracker, img):
    windows = get_all_sliding_windows(img)
    possible_cars = find_car_boxes(img, windows, clasifier, scaler)
    car_tracker.add_possible_cars_for_frame(possible_cars)
    return car_tracker.draw_heatmap_boxes(img)
    # window_img = draw_boxes(img, car_tracker.all_possible_car_boxes(),color=(0,255,0),thick=2)
    # img1 =  car_tracker.draw_heatmap_boxes(img)
    # return diagnosis_screen(img1, car_tracker.build_heatmap(car_tracker.heatmap_threshold))

#Creates diagnosis screen with a processed image and the heatmap.
def diagnosis_screen(processed_image, heatmap, multiplication_factor = 20):
    diagScreen = np.zeros((1440, 1280, 3), dtype=np.uint8)
    diagScreen[0:720, 0:1280] = processed_image
    ht = heatmap_image(heatmap, multiplication_factor)
    diagScreen[720:1440, 0:1280] = cv2.resize(ht, (1280, 720), interpolation=cv2.INTER_AREA)
    return diagScreen

#takes heatmap array with one channel and returns array with 3 channels with RED channel populated
def heatmap_image(heatmap, multiplication_factor = 20):
    ht = np.zeros((heatmap.shape[0], heatmap.shape[1], 3), dtype=np.uint8)
    heatmap = heatmap * multiplication_factor
    heatmap[heatmap > 255] = 255
    ht[:, :, 0] = heatmap[:, :, 0]
    return ht

clasifier = pickle.load(open("./../models/classifier2.pkl", "rb"))
X_scaler = pickle.load(open("./../models/scaler2.pkl", "rb"))

def process_images(path):
    images = glob.glob(path)

    for fname in images:
        print("Processing image {}".format(fname))
        img = mpimg.imread(fname)
        # instantiate new car object so images are processed independently
        ct = CarTracker(1, img.shape)
        print(img.shape)
        start = time.time()
        ds = process_image(clasifier, X_scaler, ct, img)
        end = time.time()
        print('processed in {0}'.format((end - start)))
        plot_images([ds], 1, 1, ['Diagnosis'])
    plt.show()

def process_video(path, output_video):
    clip = VideoFileClip(path)
    # clip = VideoFileClip(path).subclip(48, 51)
    # clip.save_frame("./../test_images/test9.jpg", t='00:00:50.36')
    ct = CarTracker(15, (720, 1280, 3), heatmap_threshold=5)
    process = partial(process_image, clasifier, X_scaler, ct)
    output_clip = clip.fl_image(process)
    output_clip.write_videofile(output_video, audio=False)


# process_images("./../test_images/test6.jpg")
process_video("./../project_video.mp4","./../output_videos/project_video_final.mp4")

