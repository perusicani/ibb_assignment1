import cv2
import numpy as np

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

    # Calculate histogram using NumPy directly
    hist, _ = np.histogram(lbp_img.flatten(), bins=np.arange(0, P + 2), range=(0, P + 1))

    # normalize
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-9)

    return hist

print("Running test script since something is hella wrong")

# Example usage:
# Provide two simple images (e.g., small grayscale images) for testing
image_path1 = 'detected/0503.png'
image_path2 = 'gt/0504.png'

# Extract LBP vectors
lbp_vector1 = extract_lbp_vector(image_path1)
lbp_vector2 = extract_lbp_vector(image_path2)

# Compute Euclidean distance between histograms
euclidean_distance = np.linalg.norm(lbp_vector1 - lbp_vector2)

print("Euclidean Distance:", euclidean_distance)
print("Histogram 1:", lbp_vector1)
print("Histogram 2:", lbp_vector2)
