import numpy as np
import math
import glob
import cv2
from time import time

from quadtree_algorithms import pad_images

input_folder = "test_images/input/"
output_folder = "test_images/output/"

image_size = (600, 400)
MIN_BLOCK_SIZE = (2, 2)  # Smallest blocksize in pixels
MAXIMUM_LEVEL = math.log(max(image_size[0] - 1, image_size[1] - 1)) + 1


start_time = time()

# Load images
images = []
for filename in glob.glob(input_folder + "*.jpg"):
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    images.append(np.float32(image))

print("--- Loading images took %s seconds ---" % (time() - start_time))

images = pad_images(images)

print("--- Program execution took %s seconds ---" % (time() - start_time))
