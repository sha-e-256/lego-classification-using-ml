import cv2 as cv
import numpy as np
import math

white = [255, 255, 255]

# This module provides functions that can be used to pre-process
# training and testing images

def get_contours_array(img):

    threshold, img_mask = cv.threshold(src=img, thresh=0, maxval=255,
                                       type=(cv.THRESH_BINARY_INV + cv.THRESH_OTSU))
    # cv.imshow('img', img_mask)  # Debug statment
    # cv.waitKey()
    # cv.destroyAllWindows()
    contours_array = cv.findContours(image=img_mask, mode=cv.RETR_EXTERNAL,
                                     method=cv.CHAIN_APPROX_SIMPLE)[0]
    # First element in tuple is an array of arrays i.e. an array of contours
    # where each contour is an array of points

    return contours_array, threshold  # Image can contain one or more contours

# Rotate the image with respect to the center of the bounding box
# and increase the width and height of the image so that
# the corners of the image do not get cut off after rotation
def rotate_img(img, b_box_rect):
    img_height = int(img.shape[0])
    img_width = int(img.shape[1])
    b_box_c_x = int(b_box_rect[0][0])  # Coordinates of the center of
    b_box_c_y = int(b_box_rect[0][1])  # the bounding box
    b_box_angle = b_box_rect[2]  # The angle of the bounding box
    b_box_width = int(b_box_rect[1][0])  # Width and height of the
    b_box_height = int(b_box_rect[1][1])  # bounding box

    img_c_x = int(img_width / 2)
    img_c_y = int(img_height / 2)
    rotation_matrix = cv.getRotationMatrix2D(center=(b_box_c_x, b_box_c_y),
                                             angle=b_box_angle, scale=1)

    rotated_img = cv.warpAffine(src=img, M=rotation_matrix,
                                dsize=(img_width, img_height),
                                borderMode=cv.BORDER_CONSTANT, borderValue=(0, 0, 0))
    z = cv.getRectSubPix(image=rotated_img, patchSize=(b_box_width, b_box_height), center=(b_box_c_x, b_box_c_y))
    return z


    rotated_img_width = int(img_height * sin_rotation_matrix +
                            img_width * cos_rotation_matrix)
    rotated_img_height = int(img_height * cos_rotation_matrix +
                             img_width * sin_rotation_matrix)

    rotated_b_box_c_x = img_c_x + cos_rotation_matrix*(b_box_c_x - img_c_x) - sin_rotation_matrix*(b_box_c_y - img_c_y)
    rotated_b_box_c_y = img_c_y + sin_rotation_matrix*(b_box_c_x - img_c_x) + cos_rotation_matrix*(b_box_c_y - img_c_y)

    # Change the angle to 0 (upright bounding box)
    rotated_b_box_rect = ((rotated_b_box_c_x, rotated_b_box_c_y),
                          (b_box_width, b_box_height), 0)
    return rotated_b_box_rect

def rotate_and_square_crop(img, dst_dir):
    max_border = 220  # Don't hardcode this
    img_g = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # Convert image to greyscale
    img_g_blur = cv.GaussianBlur(src=img_g, ksize=(3, 3), sigmaX=0)

    contours_array = get_contours_array(img_g_blur)[0]
    threshold = get_contours_array(img_g_blur)[1]

    background = cv.inRange(src=img_g, lowerb=threshold, upperb=255)
    img[background > threshold] = white  # Change colour of all background pixels to white

    num_pieces_index = 0  # Not all contours enclose a piece; num_pieces_index
    # keeps track of the index of a contour which does enclose a piece

    for contour in contours_array:
        b_box_rect = cv.minAreaRect(contour)

        b_box_width = int(b_box_rect[1][0])  # Rows (width)
        b_box_height = int(b_box_rect[1][1])  # Cols (height)


        # If the bounding box is very small, it's not enclosing a piece
        if b_box_width > 80 and b_box_height > 80:
            b_box_c_x = int(b_box_rect[0][0])  # Coordinates of the center of
            b_box_c_y = int(b_box_rect[0][1])  # the bounding box
            b_box_angle = b_box_rect[2]  # The angle of the bounding box
            rotated_img = rotate_img(img, b_box_rect)  # Rotate the image with respect to the angle of
            # each bounding box in the image

            rotated_b_box_rect = ((b_box_c_x, b_box_c_y),
                                  (b_box_width, b_box_height), 0)
            rotated_b_box_c_x = int(rotated_b_box_rect[0][0])  # Coordinates of the center of
            rotated_b_box_c_y = int(rotated_b_box_rect[0][1])  # the bounding box

            b_box = np.int0(cv.boxPoints(rotated_b_box_rect))  # The coordinates
            rotated_b_box = np.int0(cv.boxPoints(rotated_b_box_rect))  # The coordinates
            # of the four corners of the bounding box rect

            cv.drawContours(rotated_img, [rotated_b_box], 0, (0, 0, 0), 1)
            cv.imshow('img', rotated_img)  # Debug statement
            cv.waitKey()
            cv.destroyAllWindows()

            # rotated_min_x = int(rotated_b_box[0, 0])
            # rotated_max_x = int(rotated_b_box[2, 0])
            # rotated_min_y = int(rotated_b_box[1, 1])
            # rotated_max_y = int(rotated_b_box[3, 1])
            #
            # right_border_width = max_border - (rotated_max_x - rotated_b_box_c_x)
            # top_border_width = max_border - (rotated_b_box_c_y - rotated_min_y)
            # left_border_width = max_border - (rotated_b_box_c_x - rotated_min_x)
            # bottom_border_width = max_border - (rotated_max_y - rotated_b_box_c_y)

            # cropped_img = rotated_img[rotated_min_y:rotated_max_y,
            #                           rotated_min_x:rotated_max_x]
            #
            #
            # square_and_cropped_img = cv.copyMakeBorder(src=cropped_img,
            #                                            top=top_border_width,
            #                                            bottom=bottom_border_width,
            #                                            right=right_border_width,
            #                                            left=left_border_width,
            #                                            borderType=cv.BORDER_CONSTANT,
            #                                            value=white)
            #
            #
            # down_points = (256, 256)
            # img_downsized = cv.resize(src=square_and_cropped_img,
            #                           dsize=down_points,
            #                           interpolation=cv.INTER_LINEAR)
            #
            # cv.imwrite(dst_dir, img_downsized)
            num_pieces_index += 1
