import cv2 as cv
import numpy as np
import math

__author__ = "Shahed E."

# This module provides functions that can be used to pre-process training and testing images

# Returns a list of each contour in the image. Each contour is an array of all the pixels i.e. the curve
# that is the boundary between white and black pixels
def get_contours(img):
    img_g = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # Convert to greyscale
    # Use Otsu's thresholding method to find the optimal threshold value
    # All pixels above the threshold value become white
    # All pixels below the threshold value become black
    threshold, img_array = cv.threshold(img_g, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)  # The + means or
    # Integer threshold can be used to determine optimal lighting; for example, avg. threshold for training images
    # should be the same for testing images
    contours_array = cv.findContours(img_array, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]

    return contours_array

# Returns the bounding box and centroid coordinates that encloses the contour
def get_bounding_box(contour):
    min_x, min_y, width, height = cv.boundingRect(contour)  # Image only contains one contour
    offset = 2  # Offset to completely enclose the Lego piece in the image
    min_x -= offset
    min_y -= offset
    max_x = min_x + width + 2 * offset
    max_y = min_y + height + 2 * offset
    m = cv.moments(contour)
    if m['m00'] != 0: # to avoid %0 case
        c_x = int(m['m10'] / m['m00'])
        c_y = int(m['m01'] / m['m00'])
    else:
        c_x = 0
        c_y = 0

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
    max_border = math.ceil(max_border) # always round UP
    contours_array = get_contours(img)
    counter = 0


    for i in range(len(contours_array)):
        img_copy = img.copy()  # Do not overwrite original image; crop the copies of the original image

        contour = contours_array[i]
        min_x, min_y, max_x, max_y, c_x, c_y = get_bounding_box(contour)
        #print(f'min_x:{min_x}, min_y:{min_y}, max_x:{max_x}, max_y: {max_y}')
        # b_rect = cv.minAreaRect(contour)  # The center coordinates, width, height, and angle of rotation of b box
        # rotation_matrix = cv.getRotationMatrix2D(center=b_rect[0], angle=b_rect[2], scale=1)
        # b_box = np.int0(cv.boxPoints(b_rect))  # Un-rotated corner coordinates
        # rotated_img = cv.warpAffine(src=img_copy, M=rotation_matrix, dsize=(1280, 720))
        # # Tuples are immutable
        # (c_x, c_y) = b_rect[0]  # Used for rotated and un-rotated images
        # c_x = int(c_x)
        # c_y = int(c_y)
        # (width, height) = b_rect[1]
        if((max_x - min_x) > 20 and (max_y - min_y) > 20):

            # rotated_b_rect = (b_rect[0], b_rect[1], 0)  # Bounding box of piece after rotating image
            # rotated_b_box = np.int0(cv.boxPoints(rotated_b_rect))  # The coordinates of the four corners of the bounding box
            # rotated_min_x = int(rotated_b_box[0, 0])
            # rotated_max_x = int(rotated_b_box[2, 0])
            # rotated_min_y = int(rotated_b_box[1, 1])
            # rotated_max_y = int(rotated_b_box[3, 1])
            #
            # print(f'{rotated_min_x}, {rotated_min_y}, {rotated_max_x}, {rotated_max_y}')
            # cropped_img = rotated_img[rotated_min_y:rotated_max_y, rotated_min_x:rotated_max_x]  # Return image within bounding box coordinates
            #
            #
            # cv.drawContours(img_copy, [b_box], 0, (0, 255, 0), 2)
            # cv.imshow("img with no bg", img_copy)
            # cv.waitKey()
            # cv.destroyAllWindows()

            right = max_border - (max_x - c_x)
            top = max_border - (c_y - min_y)
            left = max_border - (c_x - min_x)
            bottom = max_border - (max_y - c_y)
            #print(f'right:{right}, top:{top}, left:{left}, bottom:{bottom}')
            white = [255, 255, 255]
            cropped_img = img_copy[min_y:max_y, min_x:max_x]
            square_img = cv.copyMakeBorder(cropped_img, top, bottom, left, right, cv.BORDER_CONSTANT, None, white)  # Add a white border

            # cv.imshow("img", cropped_img)
            # cv.waitKey()
            # cv.destroyAllWindows()

            if (len(contours_array) == 1):
                dst_dir_img = rf'{dst_dir}'
            else:
                dst_dir_img = rf'{dst_dir}\{counter}.png'

            counter += 1
            down_width = 256
            down_height = 256
            down_points = (down_width, down_height)
            square_img_resized = cv.resize(square_img, down_points, cv.INTER_LINEAR)
            cv.imwrite(dst_dir_img, square_img_resized)


