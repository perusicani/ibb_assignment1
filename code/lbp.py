import cv2
import numpy as np
from skimage import feature
from utils import resize_and_flatten

# OWN - as simple as it gets
def regular_lbp(img, P, R):
    rows, cols = img.shape
    # initialize empty lbp
    lbp_img = np.zeros((rows - 2 * R, cols - 2 * R), dtype=np.uint8)

    # calculate lbp vals
    for i in range(R, rows - R):
        for j in range(R, cols - R):
            center = img[i, j]
            pattern = 0

            for k in range(P):
                x = i + int(R * np.cos(2 * np.pi * k / P))
                y = j - int(R * np.sin(2 * np.pi * k / P))

                if img[x, y] >= center:
                    pattern |= (1 << (P - 1 - k))

            lbp_img[i - R, j - R] = pattern

    return lbp_img

# OWN - uniform
def uniform_lbp(img, P, R):
    rows, cols = img.shape
    lbp_img = np.zeros((rows - 2 * R, cols - 2 * R), dtype=np.uint8)

    # calculate lbp vals
    for i in range(R, rows - R):
        for j in range(R, cols - R):
            center = img[i, j]
            pattern = 0

            for k in range(P):
                x = i + int(R * np.cos(2 * np.pi * k / P))
                y = j - int(R * np.sin(2 * np.pi * k / P))

                if img[x, y] >= center:
                    pattern |= (1 << (P - 1 - k))

            transitions = bin(pattern).count('01') + bin(pattern).count('10')
            if transitions <= 2:
                lbp_img[i - R, j - R] = pattern
            else:
                # If not uniform, set to special val
                lbp_img[i - R, j - R] = 255

    return lbp_img

def compute_histogram_regular(lbp_img, P):
    hist, _ = np.histogram(lbp_img.flatten(), bins=np.arange(0, 2 ** P + 2), range=(0, 2 ** P + 1))
    
    # normalize - light intensity
    # hist.sum() calculates the sum of all values in the histogram.
    # 1e-7 is added to avoid division by zero.
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)

    return hist

def compute_histogram_uniform(lbp_img, P):
    # hist, _ = np.histogram(lbp_img.flatten(), bins=np.arange(0, P + 2), range=(0, P + 1))
    
    # # normalize
    # hist = hist.astype("float")
    # hist /= (hist.sum() + 1e-9)  # Ensure not to divide by zero
    
    # ------------------
    
    # # Calculate histogram using calcHist
    # hist = cv2.calcHist([lbp_img], [0], None, [P+1], [0, P+1])
    # hist = hist.flatten() # FORGOT TO NORMALIZE AAAAAA
    
    # ------------------

    # Calculate histogram using NumPy directly
    # hist, _ = np.histogram(lbp_img.flatten(), bins=np.arange(0, P + 2), range=(0, P + 1))

    # # normalize
    # hist = hist.astype("float")
    # hist /= (hist.sum() + 1e-9)

    # ------------------
    # Only way it worked, don't know why
    hist = cv2.calcHist([lbp_img], [0], None, [256], [0, 256])

    hist = hist.flatten() / (np.sum(hist) + 1e-9)

    return hist

# Actual function to call outside to get the histogram
def extract_lbp_vector_regular(image_path, P=8, R=1):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    lbp_img = regular_lbp(image, P, R)

    lbp_vector = compute_histogram_regular(lbp_img, P)

    return lbp_vector # histogram

# Issue - euclidian distance always 0.0?? Try different hist calculation
def extract_lbp_vector_uniform(image_path, P=8, R=1):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    lbp_img = uniform_lbp(image, P, R)
    
    lbp_vector = compute_histogram_uniform(lbp_img, P)

    return lbp_vector # histogram

# LIB - from skimage
def lib_lbp(image):
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    
    # Use uniform to give it the best shot - own is not uniform
    lbp_image = feature.local_binary_pattern(img, P=8, R=1, method='uniform')

    hist, _ = np.histogram(lbp_image, bins=np.arange(0, 10), range=(0, 10))

    return hist

# PIXEL2PIXEL - resized and flattened images to 1D vectors
def p2p(image):
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    # Reshape the image into a one-dimensional vector -> img.flatten()
    # Issue with euclidian when not normalized
    feature_vector = resize_and_flatten(img)
    # print(feature_vector)
    return feature_vector