import cv2
import numpy as np
import matplotlib.pyplot as plt

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

            # Check if the pattern is uniform
            transitions = bin(pattern).count('01') + bin(pattern).count('10')
            if transitions <= 2:
                lbp_img[i - R, j - R] = pattern
            else:
                # If not uniform, set it to a special value
                lbp_img[i - R, j - R] = 255

    return lbp_img

def extract_lbp_vector(image_path, P=8, R=1):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Could not read image at path: {image_path}")

    lbp_img = uniform_lbp(image, P, R)

    # Calculate histogram using cv2.calcHist
    hist = cv2.calcHist([lbp_img], [0], None, [P+1], [0, P+1])

    # normalize
    hist /= (np.sum(hist) + 1e-9)

    return hist.flatten()

# Example usage:
# Provide the image paths
image_path1 = 'detected/0503.png'
image_path2 = 'gt/0504.png'

# Read images
image1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)

# Check LBP patterns directly
lbp_img1 = uniform_lbp(image1, P=8, R=1)
lbp_img2 = uniform_lbp(image2, P=8, R=1)

# Calculate histograms directly from LBP images using OpenCV's calcHist
hist1 = cv2.calcHist([lbp_img1], [0], None, [256], [0, 256])
hist2 = cv2.calcHist([lbp_img2], [0], None, [256], [0, 256])

# Normalize histograms
hist1 = hist1.flatten() / (np.sum(hist1) + 1e-9)
hist2 = hist2.flatten() / (np.sum(hist2) + 1e-9)

# Compute Euclidean distance between histograms
euclidean_distance = np.linalg.norm(hist1 - hist2)

print("Euclidean Distance:", euclidean_distance)