import time
import numpy as np
import cython

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def spatial_frequency(ca):
    start_time = time.time()
    row_frequency = np.empty_like(ca)
    column_frequency = np.empty_like(ca)

    cdef int x, y
    cdef int h = ca.shape[0]
    cdef int w = ca.shape[1]

    # Row frequency
    for y in range(0, h):
        for x in range(1, w): # Start one further
            row_frequency[y, x] = (ca[y, x] - ca[y - 1, x]) ** 2
    # Column frequency
    for x in range(0, w): # Start one further
        for y in range(1, h):
            column_frequency[y, x] = (ca[y, x] - ca[y, x - 1]) ** 2

    mXn = 1 / (ca.shape[0] * ca.shape[1])
    row_frequency = np.sqrt(row_frequency * mXn)
    column_frequency = np.sqrt(column_frequency * mXn)

    # Spatial frequency of "ca"
    sf = np.sqrt(np.add(column_frequency ** 2, row_frequency ** 2))
    print("--- Spatial frequency computation took %s seconds ---" % (time.time() - start_time))
    return sf