import numpy as np
import cv2
import glob
from time import time

input_folder = "test_images/input/"
output_folder = "test_images/output/"
wavelet_to_use = "db4"
spatial_frequency_kernel_size = (
    5,
    5,
)  # Spatial frequency block size in pixels (y, x)
SML_threshold = 10

from algorithms import (
    spatial_frequency,
    sum_modified_laplacian,
    reshape_split_array,
    pad_array,
)


def _time_it(f):
    def wrapper(*args):
        start = time()
        r = f(*args)
        end = time()
        print("timed %s: %f" % (f.__name__, end - start))
        return r

    return wrapper


# Compute the focus measures of a list of arrays
@_time_it
def compute_focus_measures(
    src_arrays: list,
    kernel_size: tuple,
    focus_measure_function,
    function_argument=None,
):
    # Get image tiles
    tiles_per_array = []
    fm_per_array = []
    for i, ca in enumerate(src_arrays):
        # Split array in smaller tiles (acces to specific tile with: tiles[y_number][x_number])
        tiles = reshape_split_array(ca, np.array(kernel_size))
        tiles_per_array.insert(i, tiles)

        # Calculate focus measure per tile
        fm_per_tile = np.empty((tiles.shape[0], tiles.shape[1]), dtype=tiles.dtype)
        for y_index, _ in enumerate(tiles):
            for x_index, _ in enumerate(tiles[0]):
                focus_measure = 0
                if function_argument != None:
                    focus_measure = focus_measure_function(
                        tiles[y_index][x_index], function_argument
                    )
                else:
                    focus_measure = focus_measure_function(tiles[y_index][x_index])
                fm_per_tile[y_index, x_index] = focus_measure
        fm_per_array.insert(i, fm_per_tile)

    # Add zeros padding
    output_array = np.empty_like(pad_array(src_arrays[0], np.array(kernel_size)))

    # Get best image per tile (highest focus measure)
    for y_tile_index in range(0, len(tiles_per_array[0])):
        y_offset = y_tile_index * kernel_size[0]
        for x_tile_index in range(0, len(tiles_per_array[0][0])):
            best_fm_image_index = 0
            for array_index, fm_array in enumerate(fm_per_array):
                if (
                    fm_array[y_tile_index, x_tile_index]
                    > fm_per_array[best_fm_image_index][y_tile_index, x_tile_index]
                ):
                    best_fm_image_index = array_index

            x_offset = x_tile_index * kernel_size[1]
            if x_offset < 0:
                x_offset = 0
            if y_offset < 0:
                y_offset = 0

            output_array[
                y_offset : (y_offset + kernel_size[0]),
                x_offset : (x_offset + kernel_size[1]),
            ] = tiles_per_array[best_fm_image_index][y_tile_index, x_tile_index]

    return output_array


start_time = time()

images = []
for filename in glob.glob(input_folder + "*.jpg"):
    # Load image
    image = cv2.imread(filename)
    image = np.float32(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    images.append(image)

# Approximation coefficient
output_image = compute_focus_measures(
    images, spatial_frequency_kernel_size, spatial_frequency
)

# ll = np.maximum.reduce(ll)
# lh = pad_array(np.mean(lh, axis=0), np.array(spatial_frequency_kernel_size))
# hl = pad_array(np.mean(hl, axis=0), np.array(spatial_frequency_kernel_size))
# hh = pad_array(np.mean(hh, axis=0), np.array(spatial_frequency_kernel_size))

# IDWT and write output image
cv2.imwrite(output_folder + "output.jpg", output_image)

print("Wrote output image.")
print("--- Program execution took %s seconds ---" % (time() - start_time))
