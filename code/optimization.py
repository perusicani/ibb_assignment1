import cv2
import itertools
import os
from os import listdir
from itertools import product
import numpy as np
from iou import get_iou
from utils import get_ground_truth
from yolo_bb_convert import yolo_to_bbox, classifier_to_bbox

def read_annotations_from_file(annotation_file, im_w, im_h):
    annotations = []
    with open(annotation_file, 'r') as file:
        for line in file:
            # Parse the line and extract annotation information
            components = line.strip().split()
            gt = map(float, components)

			# x1, y1, x2, y2
            yolo_bbox = yolo_to_bbox(gt, im_w, im_h)

            annotations.append((yolo_bbox))
    return annotations

def process_detected_ear(ears, gt, im_w, im_h):
	for (x,y,w,h) in ears:
		yolo_bbox =  yolo_to_bbox(gt, im_w, im_h)
		detected_bbox = classifier_to_bbox(x,y,w,h)

		iou = get_iou(detected_bbox, yolo_bbox)
		# print(f"IOU: {iou}")

		crop_img = gray[y:y+h, x:x+w]
		cv2.imwrite(detected_dir+image, crop_img)
		return iou

def evaluate_vj_classifier(directory, dataset, scaleFactor, minNeighbours):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    ious = 0
    count = 0
    for image in os.listdir(directory):
        if (image in dataset):
            base, _ = os.path.splitext(image)
            img = cv2.imread(f"{directory}/{image}")
            im_h, im_w, im_c = img.shape
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            left_ear_cascade = cv2.CascadeClassifier('code/haarcascade_mcs_leftear.xml')
            right_ear_cascade = cv2.CascadeClassifier('code/haarcascade_mcs_rightear.xml')

            left_ears = left_ear_cascade.detectMultiScale(gray, scaleFactor, minNeighbours)
            right_ears = right_ear_cascade.detectMultiScale(gray, scaleFactor, minNeighbours)

            # Read annotations from the associated annotation file
            annotations = read_annotations_from_file(f"{directory}/{base}.txt", im_w, im_h)

            if (not len(left_ears) == 0):
                count += 1
                for (x,y,w,h) in left_ears:
                    for annotation in annotations:
                        # print(f"Chekcing (x,y,w,h): {x,y,w,h} with annotation: {annotation}")
                        iou = get_iou(classifier_to_bbox(x,y,w,h), annotation)
                        ious+=iou
                        # print(f"IOU: {iou}")
                        if iou > 0.5:  # You can adjust this threshold as needed
                            true_positives += 1
                            break
                    else:
                        false_positives += 1
            elif (not len(right_ears) == 0):
                count += 1
                for (x,y,w,h) in right_ears:
                    for annotation in annotations:
                        # print(f"Chekcing (x,y,w,h): {x,y,w,h} with annotation: {annotation}")
                        iou = get_iou(classifier_to_bbox(x,y,w,h), annotation)
                        ious+=iou
                        # print(f"IOU: {iou}")
                        if iou > 0.5:  # You can adjust this threshold as needed
                            true_positives += 1
                            break
                    else:
                        false_positives += 1
            false_negatives += max(0, len(annotations) - true_positives)

    print(f"IOUs average: {ious/count}")
    print(f"True pos. {true_positives}")
    print(f"False pos. {false_positives}")
    print(f"False neg. {false_negatives}")
    precision = true_positives / (true_positives + false_positives + 1e-9)
    recall = true_positives / (true_positives + false_negatives + 1e-9)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-9)

    return f1_score

# ears dir path, dataset
def grid_search_vj_parameters(directory, dataset, scale_factors, min_neighbours):
    best_f1_score = 0
    best_params = None
    best_scaleFactor = None
    best_minNeighbour = None

    for scaleFactor, minNeighbours in product(scale_factors, min_neighbours):
        f1_score = evaluate_vj_classifier(directory, dataset, scaleFactor, minNeighbours)

        print(f"Parameters: scaleFactor={scaleFactor}, minNeighbours={minNeighbours}, F1 Score: {f1_score}")

        if f1_score > best_f1_score:
            best_f1_score = f1_score
            best_scaleFactor = scaleFactor
            best_minNeighbour = minNeighbours
            best_params = {'scaleFactor': scaleFactor, 'minNeighbours': minNeighbours}

    print(f"Best Parameters: {best_params}, Best F1 Score: {best_f1_score}")
    return best_scaleFactor, best_minNeighbour, best_f1_score


