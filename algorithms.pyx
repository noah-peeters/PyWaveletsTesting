# type: ignore
import numpy as np
import cython
cimport numpy as np
from libc.math cimport floor, sqrt, fabs, ceil, fmod

# Append zeros to array if not nicely divisible by kernel_size (y, x)
@cython.cdivision(True)
def pad_array(np.ndarray[np.float32_t, ndim=2] array, np.ndarray[np.int_t, ndim=1] kernel_size):
    # Must be floats for division later on
    cdef float img_height = array.shape[0]
    cdef float img_width = array.shape[1]
    cdef float tile_height = kernel_size[0]
    cdef float tile_width = kernel_size[1]

    cdef double y_padding_width = 0
    cdef double x_padding_width = 0
    cdef float intercalc

    cdef double y_remainder = fmod(img_height, tile_height)
    if y_remainder != 0:
        # Add padding on y-axis (equally on both sides)
        intercalc = ceil(img_height / tile_height)
        y_padding_width = (intercalc * tile_height) - img_height
    cdef double x_remainder = fmod(img_width, tile_width)
    if x_remainder != 0:
        # Add padding on x-axis (equally on both sides)
        intercalc = ceil(img_width / tile_width)
        x_padding_width = (intercalc * tile_width) - img_width
    
    # Pad array (if neccessary)
    return np.pad(array, [(0, int(y_padding_width)), (0, int(x_padding_width))], mode="constant", constant_values=0)

# Split array into smaller blocks (kernel_size)
# Source: https://towardsdatascience.com/efficiently-splitting-an-image-into-tiles-in-python-using-numpy-d1bf0dd7b6f7
@cython.cdivision(True)
def reshape_split_array(np.ndarray[np.float32_t, ndim=2] array, np.ndarray[np.int_t, ndim=1] kernel_size):

    # Pad array (if neccessary)
    array = pad_array(array, kernel_size)

    cdef int img_height = array.shape[0]
    cdef int img_width = array.shape[1]
    cdef int tile_height = kernel_size[0]
    cdef int tile_width = kernel_size[1]

    # Reshape array (tiling)
    cdef np.ndarray tiled_array = array.reshape(
        (
            int(floor(img_height / tile_height)), # "floor" returns a float
            int(tile_height),
            int(floor(img_width / tile_width)),
            int(tile_width),
        )
    )
    tiled_array = tiled_array.swapaxes(1, 2)
    return tiled_array

# Compute spatial frequency of array
@cython.cdivision(True)
def spatial_frequency(np.ndarray[np.float32_t, ndim=2] array):
    cdef int x, y
    cdef float calc

    cdef float row_frequency = 0
    cdef float column_frequency = 0
    # Floats for calculation
    cdef float h = array.shape[0]
    cdef float w = array.shape[1]

    # Row frequency
    for y in range(0, int(h)):
        for x in range(1, int(w)):  # Start one further
            calc = (array[y, x] - array[y - 1, x]) ** 2
            row_frequency += calc
    # Column frequency
    for x in range(0, int(w)):  # Start one further
        for y in range(1, int(h)):
            calc = (array[y, x] - array[y - 1, x]) ** 2
            column_frequency += calc

    cdef float mXn = 1 / (h * w)
    row_frequency = sqrt(mXn * row_frequency)
    column_frequency = sqrt(mXn * column_frequency)

    # Spatial frequency of array
    cdef float sf = sqrt((column_frequency ** 2) + (row_frequency ** 2))
    return sf

# Compute sum-modified-Laplacian (SML) of array
@cython.cdivision(True)
def sum_modified_laplacian(np.ndarray[np.float32_t, ndim=2] array, int threshhold):
    cdef int step = 1
    cdef int h = array.shape[0] - 1
    cdef int w = array.shape[1] - 1

    cdef float sum = 0
    cdef float x_step, y_step, delta
    cdef int x, y

    for y in range(1, h):
        for x in range(0, w):
            x_step = fabs(2 * array[y, x] - array[y, x - step] - array[y, x + step])
            y_step = fabs(2 * array[y, x] - array[y - step, x] - array[y + step, x])
            delta = x_step + y_step
            if delta >= threshhold:
                sum += delta
    return sum