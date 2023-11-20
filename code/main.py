# Assignment 1 - IBB
# Author: Ani Perušić

# Run VJ with diff params and compute Intersection-over-Union, based on SUPPLIED ground-truths.

# Code snippet from provided Jupyter notebook - minor changes (haarcascades and cropping)
import numpy as np
import cv2
from datetime import datetime
# modules for going through images
import os
from os import listdir
import shutil
# own functions
from iou import get_iou
from yolo_bb_convert import yolo_to_bbox, classifier_to_bbox
from utils import get_ground_truth, save_to_txt, read_from_txt
from dataset_split import split_dataset
from lbp import extract_lbp_vector_regular, extract_lbp_vector_uniform, lib_lbp, p2p
from optimization import grid_search_vj_parameters

# What should run? - for easier development only
RESET_DATASET = False # Don't if not necessary
RESET_FODLERS = False # Only if reset dataset has been chosen
USE_TEST_SET = False # Final calculations
RESET_PARAMS = False # Only 'train' on training set - leave off when test set active
CROP_GT = True # When changing datasets
VJ = True # When changing datasets
OWN_LBP = True
OWN_LBP_UNIFORM = True
LIB_LBP = True
P2P = True

# Directory names and paths
root = os.getcwd()

# Expects the 'ears' folder containing all images and ground truths to be at root
ears_dir = 'ears'
gt_dir_name = 'gt'
detected = 'detected'
identities = 'identities.txt'
			
gt_dir = root+f'/{gt_dir_name}/'
detected_dir = root+f'/{detected}/'

# IOU calculations for parameter tuning
def process_detected_ear(ears, gt, im_w, im_h):
	for (x,y,w,h) in ears:
		yolo_bbox =  yolo_to_bbox(gt, im_w, im_h)
		detected_bbox = classifier_to_bbox(x,y,w,h)
		
		iou = get_iou(detected_bbox, yolo_bbox)
		# print(f"IOU: {iou}")
		
		crop_img = gray[y:y+h, x:x+w]
		cv2.imwrite(detected_dir+image, crop_img)
		return iou

# Image cropping according to ground truths provided
def ground_truth_crops(root, img_dir, dataset):
	# get all ground truth crops
	for image in os.listdir(img_dir):
		# check if the image ends with png
		if (image.endswith(".png") and (image in dataset)):
			base, ext = os.path.splitext(image)
			img = cv2.imread(f"{root}/{img_dir}/{image}")
			im_h, im_w, im_c = img.shape
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			gt = get_ground_truth(f"{root}/{img_dir}/{base}.txt")
			
			# x1, y1, x2, y2
			yolo_bbox = yolo_to_bbox(gt, im_w, im_h)

			crop_img = gray[yolo_bbox[1]:yolo_bbox[3], yolo_bbox[0]:yolo_bbox[2]]
			if (len(crop_img) == 0):
				print(f"Image size: {im_h}, {im_w}")
				print(yolo_bbox)
				print("Cropped image is empty")
			else:
				cv2.imwrite(gt_dir+image, crop_img)

def euclidean_distance(vector1, vector2):
    return np.linalg.norm(np.array(vector1) - np.array(vector2))

# Return identity from identities.txt according to image name provided
def check_identity(input_string):
	with open(identities, 'r') as file:
		for line in file:
			columns = line.strip().split()
			if columns and columns[0] == input_string:
				return int(columns[1])  # Assuming the column containing the number is an integer
	return -1 # If no match is found

# Takes in a feature vector of the image we want to classify and goes through all ground truth cropps, extracts their feature vectors and measures the euclidian distance between them.
# Returns the image name and distance of the closest vector.
def feature_extraction_algorithm(detected_vector, dataset, detection_function):
	closest_distance = float('inf')
	closest_image = None

	# # find match in ground truth images
	# lbp_vector_gt = extract_lbp_vector(f"{gt_dir}{image}")

	# print(f"Comparing against images in folder: {gt_dir}")
	for image in os.listdir(gt_dir):
		if (image.endswith(".png") and (image in dataset)):
			# print(f"Comparing against: {image}")
			lbp_vector_gt = detection_function(f"{gt_dir}{image}")
			distance = euclidean_distance(detected_vector, lbp_vector_gt)

			# print(f"Distance between current image and image {image} calculated: {distance}. The current closest: {closest_distance}")
			if (distance < closest_distance):
				# print(f"Found new closest distance {distance} with image {image}")
				closest_distance = distance
				closest_image = image
	
	return closest_image, closest_distance

# Reset folders helper
if (RESET_FODLERS):
	if not os.path.exists(detected_dir):
		print("Directory for detected does not exist, creating.")
		os.mkdir(detected_dir)
	else: 
		print("Directory for detected exists, resetting.")
		shutil.rmtree(detected_dir) # reset basically
		os.mkdir(detected_dir)
	if not os.path.exists(gt_dir):
		print("Directory for gt does not exist, creating.")
		os.mkdir(gt_dir)
	else: 
		print("Directory for gt exists, resetting.")
		shutil.rmtree(gt_dir) # reset basically
		os.mkdir(gt_dir)

# Load haar cascades - expects them to be located in '[root]/code/' directory (where the python scripts are found)
left_ear_cascade = cv2.CascadeClassifier('code/haarcascade_mcs_leftear.xml')
right_ear_cascade = cv2.CascadeClassifier('code/haarcascade_mcs_rightear.xml')

# Split into training and test set
# Only if flag is set or if any of the files don't exist
if (RESET_DATASET) or not os.path.exists('code/train_set.txt') or not os.path.exists('code/test.txt'):
	# Split dataset and save them to files
	train_set, test_set = split_dataset(identities) 
	save_to_txt('code/train_set.txt', train_set)
	save_to_txt('code/test_set.txt', test_set)
else:
	# Read from files 
	train_set = read_from_txt('code/train_set.txt')
	test_set = read_from_txt('code/test_set.txt')
	
dataset = train_set
	
if (USE_TEST_SET):
	dataset = test_set

print(f"Dataset used: {dataset}")
# print('\n'.join('{}: {}'.format(*k) for k in enumerate(dataset)))

if (RESET_PARAMS or not os.path.exists('code/scaleFactor.txt') or not os.path.exists('code/minNeighbours.txt')):
	# Define the parameter grid
	scaleFactors = [1.05, 1.1, 1.15, 1.2, 1.25, 1.3]
	minNeighs = [3, 4, 5, 6]

	# Perform grid search
	print(f"Param optimization. Start: {datetime.now()}")
	scaleFactor, minNeigh, best_f1 = grid_search_vj_parameters(ears_dir, dataset, scaleFactors, minNeighs)
	# best_params = {'scaleFactor': scaleFactor, 'minNeighbours': minNeighbours}
	save_to_txt('code/scaleFactor.txt', [scaleFactor])
	save_to_txt('code/minNeighbours.txt', [minNeigh])
	print(f"Param end. Start: {datetime.now()}")
else:
	scaleFactor = float(read_from_txt('code/scaleFactor.txt')[0])
	minNeigh = int(read_from_txt('code/minNeighbours.txt')[0])


# Get ground truth crops helper
if (CROP_GT):
	print(f"Cropping ground truth images. Start: {datetime.now()}")
	ground_truth_crops(root, ears_dir, dataset)
	print(f"Cropping ground truth images. End: {datetime.now()}")

# Get best params and extract images
# params.txt and readme.txt contain some additional info
if (VJ):
	print(f"Detecting ear images. Start: {datetime.now()}")
	for image in os.listdir(ears_dir):
		# check if the image ends with png
		if (image.endswith(".png") and (image in dataset)):
			base, ext = os.path.splitext(image)
			# issues with macOS and path and VSCode
			# print(os.path.join(root, 'ears/', image))
			# print(root + '/ears/' + image)
			# Read image
			img = cv2.imread(f"{root}/{ears_dir}/{image}")
			im_h, im_w, im_c = img.shape
			# Grayscale it
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			
			## HERE ADJUST PARAMS FOR BEST RESULTS -according to IOU calculated
			left_ears = left_ear_cascade.detectMultiScale(gray, scaleFactor, minNeigh)
			right_ears = right_ear_cascade.detectMultiScale(gray, scaleFactor, minNeigh)

			gt = get_ground_truth(f"{root}/{ears_dir}/{base}.txt")
			
			if(len(left_ears) != 0): 
				iou = process_detected_ear(left_ears, gt, im_w, im_h)
			if(len(right_ears) != 0): 
				iou = process_detected_ear(right_ears, gt, im_w, im_h)
	print(f"Detecting ear images. End: {datetime.now()}")

if (OWN_LBP):
	print(f"own-LBP. Start: {datetime.now()}")
	comparisons = 0
	correct_identifications = 0
	incorrect_identifications = 0
	for image in os.listdir(detected_dir):
		if (image in dataset):
			# print(f"Check identity for detected image: {image}")
			lbp_vector_detected = extract_lbp_vector_regular(f"{detected_dir}{image}")
			
			img, dist = feature_extraction_algorithm(lbp_vector_detected, dataset, extract_lbp_vector_regular)
			
			# Check identities
			current_identity = check_identity(image) # Identity of the image we are identifying
			closest_detected_identity = check_identity(img) # Identity of the image with the smallest distance

			comparisons += 1
			if(current_identity == closest_detected_identity):
				print(f"Correct identification of image {image} with gt image {img}")
				correct_identifications += 1
			else:
				print(f"Incorrect identification of image {image} with gt image {img}")
				incorrect_identifications += 1
		
	print(f"own-LBP: Results:")
	print(f"Comparisons made: {comparisons}")
	print(f"Correct identifications: {correct_identifications}")
	print(f"Incorrect identifications: {incorrect_identifications}")
	print(f"Correct/all: {correct_identifications/comparisons}")

	print(f"own-LBP. End: {datetime.now()}")

if (OWN_LBP_UNIFORM):
	print(f"own-LBP_uniform. Start: {datetime.now()}")
	comparisons = 0
	correct_identifications = 0
	incorrect_identifications = 0
	for image in os.listdir(detected_dir):
		if (image in dataset):
			print(f"Check identity for detected image: {image}")
			lbp_vector_detected = extract_lbp_vector_uniform(f"{detected_dir}{image}")
			
			img, dist = feature_extraction_algorithm(lbp_vector_detected, dataset, extract_lbp_vector_uniform)
			
			# Check identities
			current_identity = check_identity(image) # Identity of the image we are identifying
			closest_detected_identity = check_identity(img) # Identity of the image with the smallest distance

			comparisons += 1
			if(current_identity == closest_detected_identity):
				print(f"Correct identification of image {image} with gt image {img}")
				correct_identifications += 1
			else:
				print(f"Incorrect identification of image {image} with gt image {img}")
				incorrect_identifications += 1
		
	print(f"own-LBP_uniform: Results:")
	print(f"Comparisons made: {comparisons}")
	print(f"Correct identifications: {correct_identifications}")
	print(f"Incorrect identifications: {incorrect_identifications}")
	print(f"Correct/all: {correct_identifications/comparisons}")

	print(f"own-LBP_uniform. End: {datetime.now()}")

if (LIB_LBP):
	print(f"lib-LBP. Start: {datetime.now()}")
	comparisons = 0
	correct_identifications = 0
	incorrect_identifications = 0
	for image in os.listdir(detected_dir):
		if (image in dataset):
			# print(f"Check identity for detected image: {image}")
			lbp_vector_detected = lib_lbp(f"{gt_dir}{image}")
			
			img, dist = feature_extraction_algorithm(lbp_vector_detected, dataset, lib_lbp)

			# Check identities
			current_identity = check_identity(image) # Identity of the image we are identifying
			closest_detected_identity = check_identity(img) # Identity of the image with the smallest distance

			comparisons += 1
			if(current_identity == closest_detected_identity):
				# print(f"Correct identification")
				correct_identifications += 1
			else:
				# print(f"Incorrect identification")
				incorrect_identifications += 1
		
	print(f"lib-LBP: Results:")
	print(f"Comparisons made: {comparisons}")
	print(f"Correct identifications: {correct_identifications}")
	print(f"Incorrect identifications: {incorrect_identifications}")
	print(f"Correct/all: {correct_identifications/comparisons}")

	print(f"lib-LBP. End: {datetime.now()}")

if (P2P):
	print(f"P2P. Start: {datetime.now()}")
	comparisons = 0
	correct_identifications = 0
	incorrect_identifications = 0
	for image in os.listdir(detected_dir):
		if (image in dataset):
			# print(f"Check identity for detected image: {image}")
			p2p_vector_detected = p2p(f"{gt_dir}{image}")
			
			img, dist = feature_extraction_algorithm(p2p_vector_detected, dataset, p2p)

			# Check identities
			current_identity = check_identity(image) # Identity of the image we are identifying
			closest_detected_identity = check_identity(img) # Identity of the image with the smallest distance

			comparisons += 1
			if(current_identity == closest_detected_identity):
				# print(f"Correct identification")
				correct_identifications += 1
			else:
				# print(f"Incorrect identification")
				incorrect_identifications += 1
		
	print(f"P2P: Results:")
	print(f"Comparisons made: {comparisons}")
	print(f"Correct identifications: {correct_identifications}")
	print(f"Incorrect identifications: {incorrect_identifications}")
	print(f"Correct/all: {correct_identifications/comparisons}")

	print(f"P2P. End: {datetime.now()}")

