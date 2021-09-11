# type: ignore
import numpy as np
cimport numpy as np
from libc.time cimport time, time_t
from libc.math cimport floor, sqrt, abs, ceil, fmod
cimport cython
from libc.stdio cimport printf
from posix.time cimport clock_gettime, timespec, CLOCK_REALTIME
import pywt

# Get accurate time
cdef get_time():
    cdef timespec ts
    cdef double current
    clock_gettime(CLOCK_REALTIME, &ts)
    current = ts.tv_sec + (ts.tv_nsec / 1000000000.)
    return current

# Append zeros to array if not nicely divisible by kernel_size (y, x)
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
def reshape_split_array(np.ndarray[np.float32_t, ndim=2] array, np.ndarray[np.int_t, ndim=1] kernel_size):
    cdef double start_time = get_time()

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

    cdef double end_time = get_time()
    printf("Array reshaping computation took: %f seconds.\n", (end_time - start_time))
    return tiled_array

# Compute spatial frequency of array
def spatial_frequency(np.ndarray[np.float32_t, ndim=2] ca):
    cdef double start_time = get_time()

    cdef int x, y
    cdef float calc

    cdef float row_frequency = 0
    cdef float column_frequency = 0

    cdef int h = ca.shape[0] - 1  # Python lists begin at index 0
    cdef int w = ca.shape[1] - 1

    # Row frequency
    for y in range(0, h):
        for x in range(1, w):  # Start one further
            calc = (ca[y, x] - ca[y - 1, x]) ** 2
            row_frequency += calc
    # Column frequency
    for x in range(0, w):  # Start one further
        for y in range(1, h):
            calc = (ca[y, x] - ca[y - 1, x]) ** 2
            column_frequency += calc

    cdef float mXn = 1 / ((h + 1) * (w + 1))
    row_frequency = sqrt(mXn * row_frequency)
    column_frequency = sqrt(mXn * column_frequency)

    # Spatial frequency of "ca"
    cdef float sf = sqrt((column_frequency ** 2) + (row_frequency ** 2))

    cdef double end_time = get_time()
    printf("Spatial frequency computation took: %f seconds.\n", (end_time - start_time))
    return sf

# Compute sum-modified-Laplacian (SML) of array
def sum_modified_laplacian(np.ndarray[np.float32_t, ndim=2] array, int threshhold):
    cdef double start_time = get_time()

    cdef int step = 1
    cdef int h = array.shape[0] - 1
    cdef int w = array.shape[1] - 1

    cdef float sum = 0
    cdef float x_step, y_step, delta
    cdef int x, y

    for y in range(1, h):
        for x in range(0, w):
            x_step = abs(2 * array[y, x] - array[y, x - step] - array[y, x + step])
            y_step = abs(2 * array[y, x] - array[y - step, x] - array[y + step, x])
            delta = x_step + y_step
            if delta >= threshhold:
                sum += delta
    
    cdef double end_time = get_time()
    printf("Sum-modified-Laplacian computation took: %f seconds.\n", (end_time - start_time))

    return sum