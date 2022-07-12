import cv2 as cv
import numpy as np

__author__ = "Shahed E."

# This module provides functions that can be used to pre-process training and testing images

# Returns a list of each contour in the image. Each contour is an array of all the pixels i.e. the curve
# that is the boundary between white and black pixels
def get_contours(img):
    img_g = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # Convert to greyscale
    # Use Otsu's thresholding method to find the optimal threshold value
    # All pixels above the threshold value become white
    # All pixels below the threshold value become black
    threshold, img_array = cv.threshold(img_g, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    # Integer threshold can be used to determine optimal lighting; for example, avg. threshold for training images
    # should be the same for testing images
    contours_array = cv.findContours(img_array, 1, 2)[0]

    return contours_array

# Returns the bounding box and centroid coordinates that encloses the contour
def get_bounding_box(contour):
    min_x, min_y, width, height = cv.boundingRect(contour)  # Image only contains one contour
    offset = 5  # Offset to completely enclose the Lego piece in the image
    min_x -= offset
    min_y -= offset
    max_x = min_x + width + 2 * offset
    max_y = min_y + height + 2 * offset
    m = cv.moments(contour)
    c_x = int(m['m10'] / m['m00'])
    c_y = int(m['m01'] / m['m00'])

    return min_x, min_y, max_x, max_y, c_x, c_y

# Determines maximum border width
# Use inside of a loop
# !!Improve later to work recurisvely & without calling get_contours (to get bounding box information)
def find_max_border(max_border, min_x, min_y, max_x, max_y, c_x, c_y):
    if (c_y - min_y) > max_border: max_border = c_y - min_y
    if (max_y - c_y) > max_border: max_border = max_y - c_y
    if (c_x - min_x) > max_border: max_border = c_x - min_x
    if (max_x - c_x) > max_border: max_border = max_x - c_x
    return max_border

# Create a square, cropped image based off of the bounding box enclosing the contour
def smart_crop(img, max_border, dst_dir):
    max_border = int(max_border)
    contours_array = get_contours(img)
    for contour in contours_array:
        img_copy = img.copy() # Do not overwrite original image; crop the copies of the original image

        b_rect = cv.minAreaRect(contour)  # The center coordinates, width, height, and angle of rotation of b box
        rotation_matrix = cv.getRotationMatrix2D(center=b_rect[0], angle=b_rect[2], scale=1)
        rotated_img = cv.warpAffine(src=img_copy, M=rotation_matrix, dsize=(1280, 720))
        # Tuples are immutable
        (c_x, c_y) = b_rect[0]  # Used for rotated and un-rotated images
        c_x = int(c_x)
        c_y = int(c_y)
        (width, height) = b_rect[1]

        rotated_b_rect = (b_rect[0], b_rect[1], 0)  # Bounding box of piece after rotating image
        rotated_b_box = np.int0(cv.boxPoints(rotated_b_rect))  # The coordinates of the four corners of the bounding box
        # print(b_box)
        rotated_min_x = int(rotated_b_box[0, 0] - 2)
        rotated_max_x = int(rotated_b_box[2, 0] + 4)
        rotated_min_y = int(rotated_b_box[1, 1] - 2)
        rotated_max_y = int(rotated_b_box[3, 1] + 4)

        cropped_img = rotated_img[rotated_min_y:rotated_max_y, rotated_min_x:rotated_max_x]  # Return image within bounding box coordinates

        right = max_border - (rotated_max_x - c_x)
        top = max_border - (c_y - rotated_min_y)
        left = max_border - (c_x - rotated_min_x)
        bottom = max_border - (rotated_max_y - c_y)
        white = [255, 255, 255]
        square_img = cv.copyMakeBorder(cropped_img, top, bottom, left, right, cv.BORDER_CONSTANT, None, white)  # Add a white border
        if (len(contours_array) == 1):
            dst_dir_img = rf'{dst_dir}'
        else:
            down_width = 256
            down_height = 256
            down_points = (down_width, down_height)
            dst_dir_img = rf'{dst_dir}\{contours_array.index(contour)}.png'
            cv.resize(square_img, down_points, cv.INTER_LINEAR)
        cv.imwrite(dst_dir_img, square_img)


