import cv2 as cv

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
    contours_array = get_contours(img)
    for i in range(len(contours_array)):
        img_copy = img.copy() # Do not overwrite original image; crop the copies of the original image
        min_x, min_y, max_x, max_y, c_x, c_y = get_bounding_box(contours_array[i])
        cropped_img = img_copy[min_y:max_y, min_x:max_x]  # Return image within bounding box coordinates
        right = max_border - (max_x - c_x)
        top = max_border - (c_y - min_y)
        left = max_border - (c_x - min_x)
        bottom = max_border - (max_y - c_y)
        white = [255, 255, 255]
        square_img = cv.copyMakeBorder(cropped_img, top, bottom, left, right, cv.BORDER_CONSTANT, None, white)  # Add a white border
        if (len(contours_array) == 1):
            dst_dir_img = rf'{dst_dir}'
        else:
            down_width = 150
            down_height = 150
            down_points = (down_width, down_height)
            dst_dir_img = rf'{dst_dir}\{i}.png'
            cv.resize(square_img, down_points, cv.INTER_LINEAR)
        cv.imwrite(dst_dir_img, square_img)


