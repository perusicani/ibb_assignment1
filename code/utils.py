import numpy as np
import cv2

def get_ground_truth(file):
    with open(file, 'r') as file:
        lines = file.readlines()

    for line in lines:
        components = line.strip().split()
        if len(components) == 5:
            class_label, center_x, center_y, width, height = map(float, components)
            return class_label, center_x, center_y, width, height
        else:
            print(f"Skipping invalid line: {line}")
            raise ValueError("Data provided cannot be read as YOLO bounding box data!")

# Function to resize and flatten an image
def resize_and_flatten(image, target_shape=(100, 100)):
    resized_image = cv2.resize(image, target_shape, interpolation=cv2.INTER_LINEAR)
    flattened_image = resized_image.flatten()
    return flattened_image

# Save a list of strings to a text file
def save_to_txt(file_path, data):
    with open(file_path, 'w') as file:
        for item in data:
            file.write("%s\n" % item)

# Read the contents of a text file into a list
def read_from_txt(file_path):
    with open(file_path, 'r') as file:
        data = [line.strip() for line in file.readlines()]
    return data