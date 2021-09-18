# type: ignore
import numpy as np
cimport numpy as np
from libc.math cimport fmax, floor, log, fabs

# Pad an array to be square to the closest power of 2 (>=axis_length)
cdef pad_array(np.ndarray[np.float32_t, ndim=2] array):
    cdef int y_length = array.shape[0]
    cdef int x_length = array.shape[1]
    cdef double largest_axis_length = fmax(y_length, x_length)

    # Get closest power of 2
    cdef int closest_power = 1
    while closest_power < largest_axis_length:
        closest_power *= 2
    
    cdef int y_pad_amount = closest_power - y_length
    cdef int x_pad_amount = closest_power - x_length
    return np.pad(array, [(0, int(y_pad_amount)), (0, int(x_pad_amount))], mode="constant", constant_values=0)

# Modified Laplacian (ML) at pixel
cdef modified_laplacian(np.ndarray[np.float32_t, ndim=2] array, int y, int x, int step):
    cdef double x_step = fabs(2 * array[y, x] - array[y, x - step] - array[y, x + step])
    cdef double y_step = fabs(2 * array[y, x] - array[y - step, x] - array[y + step, x])
    return x_step + y_step

# Compute gradient map for an image
cdef compute_image_gradient(np.ndarray[np.float32_t, ndim=2] array):
    cdef int step = 1
    cdef int h = array.shape[0]
    cdef int w = array.shape[1]

    for y in range(0, h):
        for x in range(0, w):
            array[y, x] = modified_laplacian(array, y, x, step)

    return array


# Compute sum of the weighted modified Laplacian (SWML) of array
@cython.cdivision(True)
cdef sum_of_weighted_modified_laplacian(np.ndarray[np.float32_t, ndim=2] array):
    # Tweaking of values possible
    cdef int step = 1
    cdef int threshhold = 5
    cdef int weighted_window_size = 8
    #cdef double modified_laplacian = 


# Padd all images in a list to be square, matching the closest power of 2
def pad_images(list images):
    # Pad arrays
    cdef int i = 0
    cdef int amount = len(images)
    for i in range(0, amount):
        images[i] = pad_array(images[i])
    return images

# Compute gradient maps for every image
def compute_gradient_maps(list images):
    cdef int i = 0
    cdef int amount = len(images)
    for i in range(0, amount):
        images[i] = compute_image_gradient(images[i])
    return images

def quadtree_decomposition(list images):
    cdef int current_level = 1
    # Maximum possible level of quadtree structure
    image_size = images[0].size
    cdef double max_possible_level = floor(log(fmax(image_size[0] - 1, image_size[1] - 1)) + 1)
    while current_level < max_possible_level:
        current_level += 1
        