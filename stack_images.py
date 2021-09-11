import numpy as np
import cv2
import pywt
import pywt.data
import glob
from time import time

input_folder = "test_images/input/"
output_folder = "test_images/output/"
wavelet_to_use = "db2"
spatial_frequency_kernel_size = (87, 3001)  # Spatial frequency block size in pixels

from algorithms import spatial_frequency, sum_modified_laplacian, reshape_split_array

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


# # Compute spatial frequency of array
# @_time_it
# def spatial_frequency(ca):
#     row_frequency = 0
#     column_frequency = 0

#     h = ca.shape[0] - 1  # Python lists begin at index 0
#     w = ca.shape[1] - 1

#     # Row frequency
#     for y in range(0, h):
#         for x in range(1, w):  # Start one further
#             row_frequency += (ca[y, x] - ca[y - 1, x]) ** 2
#     # Column frequency
#     for x in range(0, w):  # Start one further
#         for y in range(1, h):
#             column_frequency += (ca[y, x] - ca[y - 1, x]) ** 2

#     mXn = 1 / (ca.shape[0] * ca.shape[1])
#     row_frequency = math.sqrt(mXn * row_frequency)
#     column_frequency = math.sqrt(mXn * column_frequency)

#     # Spatial frequency of "ca"
#     sf = math.sqrt((column_frequency ** 2) + (row_frequency ** 2))
#     return sf


# # Compute sum-modified-Laplacian (SML) of array
# @_time_it
# def sum_modified_laplacian(array, threshhold):
#     step = 1
#     h = array.shape[0] - 1
#     w = array.shape[1] - 1

#     sum = 0
#     for y in range(1, h):
#         for x in range(0, w):
#             x_step = abs(2 * array[y, x] - array[y, x - step] - array[y, x + step])
#             y_step = abs(2 * array[y, x] - array[y - step, x] - array[y + step, x])
#             delta = x_step + y_step
#             if delta >= threshhold:
#                 sum += delta
#     return sum


# Split array into smaller blocks (kernel_size)
# Source: https://towardsdatascience.com/efficiently-splitting-an-image-into-tiles-in-python-using-numpy-d1bf0dd7b6f7
# @_time_it
# def reshape_split_array(array: np.ndarray, kernel_size: tuple):
#     img_height, img_width = array.shape
#     tile_height, tile_width = kernel_size
#     print(array.shape)

#     tiled_array = array.reshape(
#         (
#             img_height // tile_height,
#             tile_height,
#             img_width // tile_width,
#             tile_width,
#         )
#     )
#     tiled_array = tiled_array.swapaxes(1, 2)
#     return tiled_array


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
        # Split ca in smaller tiles (acces to specific tile with: tiles[y_number][x_number])
        tiles = reshape_split_array(ca, np.array(kernel_size))
        tiles_per_array.insert(i, tiles)

        # Calculate focus measure per tile
        fm_per_tile = np.empty((tiles.shape[0], tiles.shape[1]), dtype=tiles.dtype)
        for y_index, _ in enumerate(tiles):
            for x_index, _ in enumerate(tiles[0]):
                focus_measure = 0
                if function_argument:
                    focus_measure = focus_measure_function(
                        tiles[y_index][x_index], function_argument
                    )
                else:
                    focus_measure = focus_measure_function(tiles[y_index][x_index])
                fm_per_tile[y_index, x_index] = focus_measure
        fm_per_array.insert(i, fm_per_tile)

    output_array = np.empty_like(src_arrays[0])
    # Loop through every tile
    for y_tile_index, _ in enumerate(tiles_per_array[0]):
        y_offset = y_tile_index * kernel_size[0]
        for x_tile_index, _ in enumerate(tiles_per_array[0][0]):
            # Get index of image with best tile (most in focus / highest SF)
            best_ca_index = 0
            for ca_index, tiles in enumerate(tiles_per_array):
                # Is new image's tile more in focus? (higher SF)
                if (
                    fm_per_array[ca_index][y_tile_index][x_tile_index]
                    > fm_per_array[best_ca_index][y_tile_index][x_tile_index]
                ):
                    best_ca_index = ca_index

            x_offset = x_tile_index * kernel_size[1]
            if x_offset < 0:
                x_offset = 0
            if y_offset < 0:
                y_offset = 0

            output_array[
                y_offset : (y_offset + kernel_size[0]),
                x_offset : (x_offset + kernel_size[1]),
            ] = tiles_per_array[ca_index][y_tile_index][x_tile_index]

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
    lh, spatial_frequency_kernel_size, sum_modified_laplacian, 6
)
hl = compute_focus_measures(
    hl, spatial_frequency_kernel_size, sum_modified_laplacian, 6
)
hh = compute_focus_measures(
    hh, spatial_frequency_kernel_size, sum_modified_laplacian, 6
)

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
