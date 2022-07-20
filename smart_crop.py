import cv2 as cv
import numpy as np

WHITE = [255, 255, 255]
BLACK = [0, 0, 0]
true_contours = []

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

# Return the length that extends the images border so that the image
# has a square aspect ratio but the image remains centered in the image
def get_max_border(max_border, min_x, min_y, max_x, max_y, c_x, c_y):
    if (c_y - min_y) > max_border: max_border = c_y - min_y
    if (max_y - c_y) > max_border: max_border = max_y - c_y
    if (c_x - min_x) > max_border: max_border = c_x - min_x
    if (max_x - c_x) > max_border: max_border = max_x - c_x
    return max_border


# Rotate the image with respect to the center of the bounding box
# and increase the width and height of the image so that
# the corners of the image do not get cut off after rotation
def rotate_img(img, b_box_rect):
    img_height = int(img.shape[0])
    img_width = int(img.shape[1])
    (b_box_width, b_box_height) = get_bounding_box_info(b_box_rect)[0]
    (b_box_c_x, b_box_c_y) = get_bounding_box_info(b_box_rect)[1]
    b_box_angle = get_bounding_box_info(b_box_rect)[2]

    # Rotate image about center of bounding box rectangle
    rotation_matrix = cv.getRotationMatrix2D(center=(b_box_c_x, b_box_c_y),
                                             angle=b_box_angle, scale=1)

    rotated_img = cv.warpAffine(src=img, M=rotation_matrix,
                                dsize=(img_width, img_height),
                                borderMode=cv.BORDER_CONSTANT, borderValue=WHITE)

    cropped_img = cv.getRectSubPix(image=rotated_img, patchSize=(b_box_width, b_box_height),
                                   center=(b_box_c_x, b_box_c_y))
    return cropped_img


# Return information on the bounding box
def get_bounding_box_info(b_box_rect):
    b_box_width = int(b_box_rect[1][0])  # Rows (width)
    b_box_height = int(b_box_rect[1][1])  # Cols (height)
    b_box_c_x = int(b_box_rect[0][0])  # Coordinates of the center of
    b_box_c_y = int(b_box_rect[0][1])  # the bounding box
    b_box_angle = b_box_rect[2]
    b_box = np.int0(cv.boxPoints(b_box_rect))
    min_x = int(b_box[0, 0])  # The coordinates of the four corners
    max_x = int(b_box[2, 0])
    min_y = int(b_box[1, 1])
    max_y = int(b_box[3, 1])
    return (b_box_width, b_box_height), \
           (b_box_c_x, b_box_c_y), \
           b_box_angle, \
           (min_x, min_y, max_x, max_y)


# Segment an image into 1 more sub-image, where each
# image contains a Lego piece
def rotate_and_square_crop(img, dst_dir, isTest):
    #max_border = 399 + 10 # For real images
    max_border = 420 + 10  # For 3D CAD images

    img_g = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # Convert image to greyscale
    img_g_blur = cv.GaussianBlur(src=img_g, ksize=(3, 3), sigmaX=0)

    contours_array = get_contours_array(img_g_blur)[0]
    threshold = get_contours_array(img_g_blur)[1]
    background = cv.inRange(src=img_g, lowerb=threshold, upperb=255)
    img[background > threshold] = WHITE  # Change colour of all background pixels to white

    num_pieces_index = 0  # Not all contours enclose a piece; num_pieces_index
    # keeps track of the index of a contour which does enclose a piece

    for contour in contours_array:
        b_box_rect = cv.minAreaRect(contour)
        (b_box_width, b_box_height) = get_bounding_box_info(b_box_rect)[0]

        # If the bounding box is very small, it's not enclosing a piece
        if b_box_width > 200 or b_box_height > 200:
            # keep at 250 for real pics, 200 for 3d
            true_contours.append(contour)

            cropped_img = rotate_img(img, b_box_rect)  # Rotate the image with
            # respect to the angle of each bounding box in the image
            (b_box_c_x, b_box_c_y) = get_bounding_box_info(b_box_rect)[1]
            b_box_angle = 0
            rotated_b_box_rect = ((b_box_c_x, b_box_c_y),  # Align bounding box
                                  (b_box_width, b_box_height), b_box_angle)  # parallel to image borders

            (rotated_b_box_c_x, rotated_b_box_c_y) = get_bounding_box_info(rotated_b_box_rect)[1]
            (rotated_min_x, rotated_min_y,
             rotated_max_x, rotated_max_y) = get_bounding_box_info(rotated_b_box_rect)[3]

            # Center object in image by expanding border so all images
            # are scaled relative to each other
            # Determine lengths needed to expand image border
            right_border_width = max_border - (rotated_max_x - rotated_b_box_c_x)
            top_border_width = max_border - (rotated_b_box_c_y - rotated_min_y)
            left_border_width = max_border - (rotated_b_box_c_x - rotated_min_x)
            bottom_border_width = max_border - (rotated_max_y - rotated_b_box_c_y)

            # Expand image border using lengths
            square_and_cropped_img = cv.copyMakeBorder(src=cropped_img,
                                                       top=top_border_width,
                                                       bottom=bottom_border_width,
                                                       right=right_border_width,
                                                       left=left_border_width,
                                                       borderType=cv.BORDER_CONSTANT,
                                                       value=WHITE)
            # Scale down images to 256x256
            down_points = (256, 256)
            img_downsized = cv.resize(src=square_and_cropped_img,
                                      dsize=down_points,
                                      interpolation=cv.INTER_LINEAR)
            if isTest:
                img_dst_dir = rf'{dst_dir}\{str(num_pieces_index)}'
            else:
                img_dst_dir = rf'{dst_dir}.png'
            cv.imwrite(img_dst_dir, img_downsized)
            num_pieces_index += 1


def get_true_contours():
    return true_contours
