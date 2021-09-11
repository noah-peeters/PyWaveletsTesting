import numpy as np
import cv2
import pywt
import pywt.data
import glob
from time import time

input_folder = "test_images/input/"
output_folder = "test_images/output/"
wavelet_to_use = "db4"
spatial_frequency_kernel_size = (10, 10)  # Spatial frequency block size in pixels
SML_threshold = 7

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


@_time_it
def compute_wavelet(image, wavelet_name):
    # Convert to float for more resolution for use with pywt
    # Format: LL, (LH, HL, HH)
    return pywt.dwt2(np.float32(image), wavelet_name)


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

    # for y_tile_index, _ in enumerate(tiles_per_array[0]):
    #     y_offset = y_tile_index * kernel_size[0]
    #     for x_tile_index, _ in enumerate(tiles_per_array[0][0]):
    #         # Get index of image with best tile (most in focus / highest FM)
    #         best_image_index = 0
    #         for ca_index, tiles in enumerate(tiles_per_array):
    #             # Is new image's tile more in focus? (higher FM)
    #             if (
    #                 fm_per_array[ca_index][y_tile_index][x_tile_index]
    #                 > fm_per_array[best_image_index][y_tile_index][x_tile_index]
    #             ):
    #                 best_image_index = ca_index

    #         x_offset = x_tile_index * kernel_size[1]
    #         if x_offset < 0:
    #             x_offset = 0
    #         if y_offset < 0:
    #             y_offset = 0

    #         output_array[
    #             y_offset : (y_offset + kernel_size[0]),
    #             x_offset : (x_offset + kernel_size[1]),
    #         ] = tiles_per_array[ca_index][y_tile_index][x_tile_index]

    return output_array


start_time = time()

wavelet_transforms = []
for filename in glob.glob(input_folder + "*.jpg"):
    # Load image
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    wavelet_transforms.append(compute_wavelet(image, wavelet_to_use))

# Sort wavelet coefficients together
ll, lh, hl, hh = [], [], [], []
for v in wavelet_transforms:
    # Low freq.
    ll.append(v[0])
    # High freq.
    lh.append(v[1][0])
    hl.append(v[1][1])
    hh.append(v[1][2])

# Approximation coefficient
ll = compute_focus_measures(ll, spatial_frequency_kernel_size, spatial_frequency)

lh = compute_focus_measures(
    lh, spatial_frequency_kernel_size, sum_modified_laplacian, SML_threshold
)
hl = compute_focus_measures(
    hl, spatial_frequency_kernel_size, sum_modified_laplacian, SML_threshold
)
hh = compute_focus_measures(
    hh, spatial_frequency_kernel_size, sum_modified_laplacian, SML_threshold
)

# ll = np.maximum.reduce(ll)
# lh = np.mean(lh, axis=0)
# hl = np.mean(hl, axis=0)
# hh = np.mean(hh, axis=0)
# Highest absolute value for high freq.
# lh = np.maximum.reduce(lh)
# hl = np.maximum.reduce(hl)
# hh = np.maximum.reduce(hh)

# IDWT and write output image
output_image = pywt.idwt2((ll, (lh, hl, hh)), wavelet_to_use)
cv2.imwrite(output_folder + "output.jpg", output_image)

print("Wrote output image.")
print("--- Program execution took %s seconds ---" % (time() - start_time))
