import cv2 as cv
import numpy as np

__author__ = "Shahed E."
__date__ = "6/20/2022"
__status__ = "Development"

def get_contours(img):
    img_g = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Use Otsu's thresholding method to find the optimal threshold value
    # All pixels above the threshold value become white
    # All pixels below the threshold value become black
    threshold, img_array = cv.threshold(img_g, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    contours_array = cv.findContours(img_array, 1, 2)[0]
    return contours_array

def get_bounding_box(img):
    contours_array = get_contours(img)
    contour = contours_array[0]  # Return only one contour
    x_min, y_min, width, height = cv.boundingRect(contour)  # Image only contains one contour
    offset = 2  # Offset to completely enclose the Lego piece in the image
    x_min -= offset
    y_min -= offset
    x_max = x_min + width + 2 * offset
    y_max = y_min + height + 2 * offset
    m = cv.moments(contour)
    c_x = int(m['m10'] / m['m00'])
    c_y = int(m['m01'] / m['m00'])
    return x_min, y_min, x_max, y_max, c_x, c_y

def smart_crop(img):
    min_x, min_y, max_x, max_y, c_x, c_y = get_bounding_box(img)
    cropped_img = img[min_y:max_y, min_x:max_x]  # Return image within bounding box coordinates
    right = 75 - (max_x - c_x)
    top = 75 - (c_y - min_y)
    left = 75 - (c_x - min_x)
    bottom = 75 - (max_y - c_y)
    white = [255, 255, 255]
    square_img = cv.copyMakeBorder(cropped_img, top, bottom, left, right, cv.BORDER_CONSTANT, None, white)  # Add a white border
    return square_img

