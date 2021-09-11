import numpy as np
import cv2
import pywt
import pywt.data
import os
import time


def compute_wavelet(image, wavelet_name):
    # Convert to float for more resolution for use with pywt
    image = np.float32(image)

    LL, (LH, HL, HH) = pywt.dwt2(image, wavelet_name)
    # Make directory
    os.mkdir("wavelet_images/" + wavelet_name)

    # Write image to disk
    titles = ["approximation", "horizontal", "vertical", "diagonal"]
    for index, wavelet_image in enumerate([LL, LH, HL, HH]):
        # Convert back to uint8 OpenCV format
        cv2.imwrite(
            "wavelet_images/" + wavelet_name + "/" "wavelet_" + titles[index] + ".jpg",
            np.uint8(wavelet_image),
        )

    # Do Inverse Wavelet transform
    inverse_transform = pywt.idwt2((LL, (LH, HL, HH)), wavelet_name)
    cv2.imwrite(
        "wavelet_images/" + wavelet_name + "/inverse_wavelet_transform.jpg",
        np.uint8(inverse_transform),
    )


start_time = time.time()

image = cv2.imread("images/out_of_focus.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imwrite("wavelet_images/grayscale_original.jpg", image)

for family in pywt.families():
    for wavelet_name in pywt.wavelist(family, kind="discrete"):
        compute_wavelet(image, wavelet_name)
# compute_wavelet(image, "db1")

print("Wrote output images.")
print("--- Program execution took %s seconds ---" % (time.time() - start_time))

# input("Press Enter to remove wavelet images:")

# cv2.imshow('image', np.uint8(LL))
# cv2.waitKey(0)
