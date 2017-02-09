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

def find_car_boxes(img, bboxes, clasifier, scaler):
    bbox_list = []
    for bbox in bboxes:
        sub_image = img[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]]
        sub_image_resized = cv2.resize(sub_image, (64, 64))

        # sub_image_f = Features()
        # sub_image_features = sub_image_f.extract_features(sub_image_resized, cspace='YCrCb')
        sub_image_features = extract_features(sub_image_resized, color_space='YCrCb')
        sub_image_features = sub_image_features.reshape(1, -1)

        scaled = scaler.transform(sub_image_features)
        if clasifier.predict(scaled) == 1:
            confidence_score = clasifier.predict_proba(scaled)
            bbox_list.append((bbox, confidence_score[0][1]))
    return bbox_list

def get_all_sliding_windows(img, overlap=0.8):
    # windows_256 = slide_window(img, x_start_stop=[None, None], y_start_stop=[320, 720],
    #                        xy_window=(192, 192), xy_overlap=(overlap, overlap))
    windows_128 = slide_window(img, x_start_stop=[None, None], y_start_stop=[380, 650],
                           xy_window=(128, 128), xy_overlap=(overlap, overlap))
    windows_64 = slide_window(img, x_start_stop=[None, None], y_start_stop=[380, 520],
                           xy_window=(64, 64), xy_overlap=(overlap, overlap))
    # windows_64 = slide_window(img, x_start_stop=[None, None], y_start_stop=[380, 520],
    #                           xy_window=(64, 64), xy_overlap=(overlap, overlap))
    # cp = draw_boxes(img, windows_256, (255,0,0))
    # cp = draw_boxes(img, windows_128, (0,0,255))
    # cp = draw_boxes(cp, windows_64, (0,255,0))
    # plot_images([cp], 1,1, ['Boxes'])

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
    # return car_tracker.draw_heatmap_boxes(img)

    window_img = draw_boxes(img, car_tracker.all_possible_car_boxes(),color=(0,255,0),thick=2)
    # if(len(car_tracker.all_possible_cars) > 0 and len(car_tracker.all_possible_cars[0])>0):
    #     window_img = draw_boxes(window_img, car_tracker.first_frame_boxes(), (255,0,0),2)
    img1 =  car_tracker.draw_heatmap_boxes(img)

    # mpimg.imsave('../output_images/original.jpg', img)
    # mpimg.imsave('../output_images/car_boxes.jpg', window_img)
    # mpimg.imsave('../output_images/heatmap.jpg', car_tracker.build_heatmap(car_tracker.heatmap_threshold))
    # mpimg.imsave('../output_images/potential_cars.jpg', car_tracker.draw_heatmap_boxes(img1))

    return diagnosis_screen(img1, car_tracker.build_heatmap(car_tracker.heatmap_threshold))

def diagnosis_screen(processed_image, heatmap, multiplication_factor = 20):
    diagScreen = np.zeros((1440, 1280, 3), dtype=np.uint8)
    diagScreen[0:720, 0:1280] = processed_image
    ht = heatmap_image(heatmap, multiplication_factor)
    diagScreen[720:1440, 0:1280] = cv2.resize(ht, (1280, 720), interpolation=cv2.INTER_AREA)
    return diagScreen

def heatmap_image(heatmap, multiplication_factor = 20):
    ht = np.zeros((heatmap.shape[0], heatmap.shape[1], 3), dtype=np.uint8)
    heatmap = heatmap * multiplication_factor
    heatmap[heatmap > 255] = 255
    ht[:, :, 0] = heatmap[:, :, 0]
    return ht

# Load the classifier model
# clf = joblib.load('./../models/classifier.pkl')
# cal = joblib.load('./../models/calibrated.pkl')

# Load the standard scalar model
# X_scaler = joblib.load('./../models/scaler.pkl')

cal = pickle.load(open("./../models/classifier2.pkl", "rb"))
X_scaler = pickle.load(open("./../models/scaler2.pkl", "rb"))

def process_images(path):
    images = glob.glob(path)
    images.sort()

    for fname in images:
        print("Processing image {}".format(fname))
        img = mpimg.imread(fname)
        # instantiate new car object so images are processed independently
        ct = CarTracker(1, img.shape)
        print(img.shape)
        start = time.time()
        ds = process_image(cal, X_scaler, ct, img)
        end = time.time()
        print('processed in {0}'.format((end - start)))
        plot_images([ds], 1, 1, ['Diagnosis'])
    plt.show()

def process_video(path, output_video):
    # clip = VideoFileClip(path)
    clip = VideoFileClip(path).subclip(34, 38)
    # clip.save_frame("./../test_images/test9.jpg", t='00:00:50.36')

    ct = CarTracker(15, (720, 1280, 3), heatmap_threshold=5)
    process = partial(process_image, cal, X_scaler, ct)
    output_clip = clip.fl_image(process)
    output_clip.write_videofile(output_video, audio=False)


process_images("./../test_images/test6.jpg")
#
# process_video("./../project_video.mp4","./../project_video_diag.mp4")

