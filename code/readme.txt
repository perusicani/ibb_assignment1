Summary of the Python code

- Code is not optimized, and honestly ugly with no consistent naming(excuse that).

Helper constants for running specific code parts:
```
USE_TEST_SET = True     # Run using test dataset (if False, use training - larger)
RESET_FODLERS = True    # Reset folders, necessary since datasets get split every time
CROP_GT = True          # Crop images according to ground truths supplied 
VJ = True               # Run VJ detection and cropping
OWN_LBP = True          # Run own LBP extractor
LIB_LBP = True          # Run skimage LBP extractor
P2P = True              # Run pixel2pixel image flattening (with image resizing)
```

Train test dataset splitting is implemented, albiet just for show, since no training needs actually being done. So, just for shorter result runs, the test split has been used.

The Viola-Jones ear detection using the provided classifiers, has been run 12 times with parameters:
    scaleFactors = [1.05, 1.1, 1.15, 1.2, 1.25, 1.3]
    minNeighs = [3, 4, 5, 6]
as mentioned in `params.txt`.
According to the IOUs received from those runs, scaleFactor = 1.25 and minNeighbours = 6 has been chosen (gave out the best IOU scores).

Grayscaled images have been automatically cropped and put into the generated `detected` folder.
Also, grayscaled gorund truth cropped images have been put into the genearted `gt` folder.

All "identity check" functions (own LBP, library LBP and pixel2pixel) use the `feature_extraction_algorithm` function that takes the currently processed image feature vector and compares it to all feature vectors of images in the `gt` folder (in summary, it compares extracted image vector A of the image we wish to identify to all extracted image vectors of the ground truth cropped images).
It returns the closest distance it found, and the name of the image itself.

After that, each function checks if the identities of the currently processed image and the closest image match - which means we have the correct identification.

Implementations of the skimage LBP (uniform for better results) and the pixel2pixel feature extractors is trivial and mostly follows found internet sources and their respective documentations.

The "own" implementation of the LBP extractor is trivial, using histograms (following multiple internet and provided paper sources for understanding and guidance).

own_lbp:
    Input:
        img: Grayscale image on which Local Binary Pattern (LBP) will be calculated.
        P: Number of sampling points.
        R: Radius of the circular region for sampling points.
    Output:
        Returns a 2D NumPy array representing the LBP values for each pixel in the input image.
    Explanation:
        The function initializes an empty array (lbp_img) to store the LBP values.
        It iterates through each pixel of the input image, excluding a border of radius R to ensure that the circular region for sampling points remains within the image bounds.
        For each pixel, it calculates an LBP value by comparing the intensity of the central pixel with the intensities of P sampling points arranged in a circular pattern around the central pixel.
        The calculated LBP value is assigned to the corresponding location in the lbp_img array.
        The resulting lbp_img array contains the LBP values for each pixel in the input image.

compute_histogram:
    Input:
        lbp_img: 2D array containing the LBP values.
        P: Number of sampling points (used for histogram binning).
    Output:
        Returns a normalized histogram of the LBP values.
    Explanation:
        The function calculates a histogram (hist) of the flattened LBP values using np.histogram.
        The histogram is normalized based on the light intensity, ensuring that the sum of all values in the histogram is 1.
        A small constant (1e-7) is added to the denominator to avoid division by zero.
        The resulting normalized histogram is returned.

extract_lbp_vector:
    Input:
        image_path: Path to the input image.
        P: Number of sampling points for LBP calculation (default is 8).
        R: Radius of the circular region for sampling points (default is 1).
    Output:
        Returns the normalized LBP histogram vector.
    Explanation:
        The function reads a grayscale image using OpenCV (cv2.imread).
        It then calls the own_lbp function to compute the LBP values for each pixel.
        The LBP histogram is calculated using the compute_histogram function.
        The resulting normalized LBP histogram vector is returned.


Simple folder structure:
root
    - code
        - python scripts
        - haarcascades
        - params, readme, results (txt)
    - detected (VJ detected cropped reagions)
    - ears (provided assignment images and ground truths)
    - gt (ground truth cropped images)
    - identities.txt