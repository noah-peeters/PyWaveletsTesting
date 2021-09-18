# type: ignore
import numpy as np
cimport numpy as np
from libc.math cimport fmax

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



# Padd all images in a list to be square, matching the closest power of 2
def pad_images(list images):
    # Pad arrays
    cdef int i = 0
    for i in range(0, len(images)):
        pad_array(images[i])
        