import numpy as np
from scipy.ndimage.measurements import label
import cv2

class CarTracker:

    def __init__(self, number_of_tracked_frames, img_shape, heatmap_threshold=1, confidence_threshold=0.8):
        self.number_of_tracked_frames = number_of_tracked_frames
        self.all_possible_cars=[]
        self.img_shape = img_shape
        self.heatmap_threshold = heatmap_threshold
        self.confidence_threshold = confidence_threshold
        self.previous_car_boxes = []

    def add_possible_cars_for_frame(self,possible_cars):
        self.previous_car_boxes = self.get_heatmap_boxes()
        keep = self.all_possible_cars[0:self.number_of_tracked_frames-1]
        self.all_possible_cars = []
        self.all_possible_cars.append(possible_cars)
        self.all_possible_cars[1:self.number_of_tracked_frames] = keep

    def first_frame_boxes(self):
        boxes = []
        for pc in self.all_possible_cars[0]:
            boxes.append(pc[0])
        return boxes

    def all_possible_car_boxes(self):
        car_boxes = []
        for possible_cars in self.all_possible_cars:
            for pc in possible_cars:
                if(pc[1]>self.confidence_threshold or self.box_belongs_to_existing_car(pc[0])):
                    car_boxes.append(pc[0])
        return car_boxes

    def build_heatmap(self, heatmap_threshold):
        all_boxes = self.all_possible_car_boxes()
        heatmap = np.zeros([self.img_shape[0], self.img_shape[1], 1]).astype(np.float)
        for box in all_boxes:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
        # Zero out pixels below the threshold
        heatmap[heatmap < heatmap_threshold] = 0
        return heatmap

    def find_heatmap_boxes(self, heatmap_threshold):
        heatmap = self.build_heatmap(heatmap_threshold)
        labels = label(heatmap)
        boxes = []
        # Iterate through all detected cars
        for car_number in range(1, labels[1] + 1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            boxes.append(bbox)
        return boxes

    def get_heatmap_boxes(self):
        return self.find_heatmap_boxes(self.heatmap_threshold)
        # currentBoxes =  self.find_heatmap_boxes(self.heatmap_threshold)
        # previousBoxes = self.previous_car_boxes
        # lowThresholdCandidates = self.find_heatmap_boxes(1)
        # add back boxes that pass a lower threshold and are close to one of previous boxes
        # for box in lowThresholdCandidates:
        #     if(self.box_close_to_one_of_boxes(box, previousBoxes) and not self.box_close_to_one_of_boxes(box, currentBoxes)):
        #         currentBoxes.append(box)
        # return currentBoxes


    def draw_heatmap_boxes(self, img):
        boxes = self.get_heatmap_boxes()
        # print('All boxes {}'.format(self.all_possible_cars))
        # print('All possible cars {}'.format(self.all_possible_car_boxes()))
        # print('Heatmap boxes:{0}'.format(boxes))
        # for box in boxes:
        #     print("center {0}".format(self.get_box_center(box)))
        img_copy = np.copy(img)
        for box in boxes:
            cv2.rectangle(img_copy, box[0], box[1], (0, 0, 255), 4)
        return img_copy

    def get_box_center(self, box):
        return ((box[0][0] + box[1][0])/2, (box[0][1] + box[1][1])/2)

    def boxes_are_close(self, box1, box2):
        distance = 20
        box_center_1 = self.get_box_center(box1)
        box_center_2 = self.get_box_center(box2)
        return abs(box_center_1[0]-box_center_2[0])<distance and abs(box_center_1[1]-box_center_2[1])<distance

    def box_belongs_to_existing_car(self, box):
        return self.box_close_to_one_of_boxes(box, self.previous_car_boxes)

    def box_close_to_one_of_boxes(self, box, boxes):
        for car_box in boxes:
            if(self.boxes_are_close(box, car_box)== True):
                return True
        return False

# ct = CarTracker(2)
# print(ct.all_possible_cars, len(ct.all_possible_cars))
# ct.add_possible_cars_for_frame([((1,1),(1,1))])
# cb = ct.all_possible_car_boxes()
# print(ct.all_possible_cars, len(ct.all_possible_cars), len(cb))
# ct.add_possible_cars_for_frame([((2,2),(2,2)),((2,2),(2,2))])
# print(ct.all_possible_cars, len(ct.all_possible_cars), len(ct.all_possible_car_boxes()))
# ct.add_possible_cars_for_frame([((3,3),(3,3)),((3,3),(3,3)),((3,3),(3,3))])
# print(ct.all_possible_cars, len(ct.all_possible_cars), len(ct.all_possible_car_boxes()))
