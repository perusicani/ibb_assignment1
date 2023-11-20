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

# Example usage:
# Provide a small grayscale image for testing
image_path = 'detected/0503.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise ValueError(f"Could not read image at path: {image_path}")

# Visualize the LBP image
P = 8
R = 1
lbp_img = uniform_lbp(image, P, R)

# Display the original and LBP images
cv2.imshow('Original Image', image)
cv2.imshow('LBP Image', lbp_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
