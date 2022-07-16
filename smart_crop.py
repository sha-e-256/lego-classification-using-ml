import cv2 as cv
import numpy as np
import math


# This module provides functions that can be used to pre-process training and testing images

def get_contours_array(img):
    img_g = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # Convert image to greyscale
    img_g_blur = cv.GaussianBlur(src=img_g, ksize=(3, 3), sigmaX=0)
    threshold, img_mask = cv.threshold(src=img_g_blur, thresh=0, maxval=255,
                                       type=(cv.THRESH_BINARY_INV + cv.THRESH_OTSU))
    # cv.imshow('img', img_mask)  # Debug statment
    # cv.waitKey()
    # cv.destroyAllWindows()
    contours_array = cv.findContours(image=img_mask, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_SIMPLE)[0]
    # First element in tuple is an array of arrays i.e. an array of contours where each contour is an array of points

    return contours_array  # Image can contain one or more contours


def rotate_and_square_crop(img, dst_dir):
    max_border = 400  # Don't hardcode this
    contours_array = get_contours_array(img)
    num_pieces_index = 0  # Not all contours enclose a piece; num_pieces_index keeps track of
    # The index of a contour which does enclose a piece

    for contour in contours_array:
        b_box_rect = cv.minAreaRect(contour)
        b_box_c_x, b_box_c_y = b_box_rect[0]  # Coordinates of the center of the bounding box
        b_box_c_x = int(b_box_c_x)
        b_box_c_y = int(b_box_c_y)
        b_box_width, b_box_height = b_box_rect[1]  # Width and height of the bounding box
        b_box_angle = b_box_rect[2] # The angle of rotation of the bounding box
        rotation_matrix = cv.getRotationMatrix2D(center=(b_box_c_x, b_box_c_y), angle=b_box_angle, scale=1)
        img_height, img_width = img.shape[:2] # [0] and [1] are the image height and width, respectively
        rotated_img = cv.warpAffine(src=img, M=rotation_matrix, dsize=(img_height, img_width))

        if(b_box_width > 20 and b_box_height > 20):
            # If the bounding box is any smaller, it's probably not enclosing a piece
            # The glare can generate a contour
            rotated_b_box_rect = ((b_box_c_x, b_box_c_y),(b_box_width, b_box_height), 0)  # Change the angle to 0 (upright bounding box)
            rotated_b_box = np.int0(cv.boxPoints(rotated_b_box_rect)) # The coordinates of the four corners of the bounding box rect
            rotated_min_x = int(rotated_b_box[0, 0])
            rotated_max_x = int(rotated_b_box[2, 0])
            rotated_min_y = int(rotated_b_box[1, 1])
            rotated_max_y = int(rotated_b_box[3, 1])

            right_border_width = max_border - (rotated_max_x - b_box_c_x)
            top_border_width = max_border - (b_box_c_y - rotated_min_y)
            left_border_width = max_border - (b_box_c_x - rotated_min_x)
            bottom_border_width = max_border - (rotated_max_y - b_box_c_y)

            white = [255, 255, 255]
            cropped_img = rotated_img[rotated_min_y:rotated_max_y, rotated_min_x:rotated_max_x]
            square_and_cropped_img = cv.copyMakeBorder(src=cropped_img, top=top_border_width, bottom=bottom_border_width, right=right_border_width, left=left_border_width, borderType=cv.BORDER_CONSTANT, value=white)

            down_points = (256, 256)
            img_downsized = cv.resize(src=square_and_cropped_img, dsize=down_points, interpolation=cv.INTER_LINEAR)
            cv.imwrite(dst_dir, img_downsized)
            num_pieces_index += 1


