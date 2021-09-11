import time
import numpy as np
import math

def _time_it(f):
    def wrapper(*args):
        start = time()
        r = f(*args)
        end = time()
        print("timed %s: %f" % (f.__name__, end - start))
        return r

    return wrapper

# Compute spatial frequency of array
@_time_it
def spatial_frequency(ca: np.ndarray):

    cdef int x, y, w, h

    cdef float row_frequency = 0
    cdef float column_frequency = 0

    h = ca.shape[0] - 1  # Python lists begin at index 0
    w = ca.shape[1] - 1

    # Row frequency
    for y in range(0, h):
        for x in range(1, w):  # Start one further
            row_frequency += (ca[y, x] - ca[y - 1, x]) ** 2
    # Column frequency
    for x in range(0, w):  # Start one further
        for y in range(1, h):
            column_frequency += (ca[y, x] - ca[y - 1, x]) ** 2

    mXn = 1 / (ca.shape[0] * ca.shape[1])
    row_frequency = math.sqrt(mXn * row_frequency)
    column_frequency = math.sqrt(mXn * column_frequency)

    # Spatial frequency of "ca"
    sf = math.sqrt((column_frequency ** 2) + (row_frequency ** 2))
    return sf