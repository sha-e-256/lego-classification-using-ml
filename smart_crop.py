import cv2 as cv
import numpy as np

# The purpose of the smart_crop module is to provide functions that can be used to
# pre-process both training and testing

WHITE = [255, 255, 255]  # OpenCV uses GBR
BLACK = [0, 0, 0]

# Return width and height of a bounding box
def get_bounding_box_size(b_box_rect):
    b_box_width = int(b_box_rect[1][0])  # Rows (width)
    b_box_height = int(b_box_rect[1][1])  # Cols (height)
    return b_box_width, b_box_height


# Return the coordinates of the center of the bounding box
def get_bounding_box_center(b_box_rect):
    b_box_c_x = int(b_box_rect[0][0])  # Coordinates of the center of
    b_box_c_y = int(b_box_rect[0][1])  # the bounding box
    return b_box_c_x, b_box_c_y


# Return coordinates of the four corners of the bounding box
def get_bounding_box_points(np_array_b_box):
    points = np.int0(np_array_b_box)[0]  # Array of points of the four corners of the bounding box
    points[points < 0] = 0  # Set negative points to be equal to zero
    min_x = points[1][1]
    max_x = points[0][1]
    min_y = points[1][0]
    max_y = points[2][0]
    return min_x, max_x, min_y, max_y


# Return width and height of an image
def get_img_size(img):
    img_height = int(img.shape[0])  # Rows
    img_width = int(img.shape[1])  # Cols
    return img_width, img_height


# Return center coordinates of img
def get_img_center(img):
    img_width, img_height = get_img_size(img)
    img_center_x = img_width // 2
    img_center_y = img_height // 2
    return img_center_x, img_center_y


# Sorts the contours based off of the size of the minimum bounding box enclosed by the contour
# And returns an array of the index of the contours in order of descending bounding box size
def get_contours_sorted_by_descending_size(contours):
    b_boxes = []
    for contour in contours:
        b_box_rect = cv.minAreaRect(contour)
        (b_box_width, b_box_height) = get_bounding_box_size(b_box_rect)
        area = b_box_width * b_box_height
        b_boxes.append(area)
    sorted_b_boxes = sorted(list(enumerate(b_boxes)), key=lambda x: x[1], reverse=True)
    # The lambda function allows for the list
    # to be sorted by the size of the bounding box
    # while maintaining the index
    # of the contour in the initial unsorted list
    # for box in sorted_b_boxes:
    #     print(box)
    contours_descending_size = []
    for i in range(len(sorted_b_boxes)):
        contours_descending_size.append(sorted_b_boxes[i][0])
    return contours_descending_size


# Determine the maximum border needed to enclose all pieces
# such that each image
# has a square aspect ratio, the piece is centered in the image,
# and the pieces are all scaled relative to each other
def get_max_border(max_border, min_x, min_y, max_x, max_y, c_x, c_y):
    if (c_y - min_y) > max_border: max_border = c_y - min_y
    if (max_y - c_y) > max_border: max_border = max_y - c_y
    if (c_x - min_x) > max_border: max_border = c_x - min_x
    if (max_x - c_x) > max_border: max_border = max_x - c_x
    return max_border


# Rotate the unsegmented image with respect to its center
# and increase the width and height of the image so that
# the corners of the image do not get cut off after rotation
def rotate_and_crop_img(img, contour):
    img_width, img_height = get_img_size(img)
    img_center_x, img_center_y = get_img_center(img)
    b_box_rect = cv.minAreaRect(contour)
    b_box_angle = b_box_rect[2]

    # Determine rotation matrix needed to rotate
    # unsegmented image about its center
    rotation_matrix = cv.getRotationMatrix2D(center=(img_center_x, img_center_y),
                                             angle=b_box_angle, scale=1)
    cos_rotation_matrix = np.abs(rotation_matrix[0][0])  # X component of the rotation matrix
    sin_rotation_matrix = np.abs(rotation_matrix[0][1])  # Y component of the rotation matrix
    # New width/heights of unsegmented image after rotation
    rotated_img_width = int((img_height * cos_rotation_matrix) +
                            (img_width * sin_rotation_matrix))
    rotated_img_height = int((img_height * sin_rotation_matrix) +
                             (img_width * cos_rotation_matrix))
    # Update components in rotation matrix to include translation
    rotation_matrix[0][2] += (rotated_img_width // 2) - img_center_x
    rotation_matrix[1][2] += (rotated_img_height // 2) - img_center_y

    # Rotate/translate the image using the rotation matrix
    rotated_img = cv.warpAffine(src=img, M=rotation_matrix,
                                dsize=(rotated_img_width, rotated_img_height),
                                borderMode=cv.BORDER_CONSTANT, borderValue=WHITE)

    b_box = cv.boxPoints(b_box_rect)
    # Rotate/translate bounding box using the rotation matrix
    rotated_b_box = cv.transform(np.array([b_box]), rotation_matrix)
    min_x, max_x, min_y, max_y = get_bounding_box_points(rotated_b_box)
    cropped_img = rotated_img[min_x:max_x, min_y:max_y]
    return cropped_img


# Set background colour to white
def clear_background(img):
    dark_colour = np.array([255, 70, 255])  # HSV for all
    light_colour = np.array([0, 0, 0])  # washed out colours
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    img_mask = cv.inRange(src=img_hsv,
                          lowerb=light_colour,
                          upperb=dark_colour)

    img[img_mask == 255] = WHITE
    return img_mask


# Find border of image that results in a segmented image with
# a square aspect ratio
def get_border_widths(max_border, min_x, max_x, min_y, max_y, b_box_c_x, b_box_c_y):

    right_border_width = max_border - (max_x - b_box_c_x)
    top_border_width = max_border - (b_box_c_y - min_y)
    left_border_width = max_border - (b_box_c_x - min_x)
    bottom_border_width = max_border - (max_y - b_box_c_y)
    return right_border_width, top_border_width, left_border_width, bottom_border_width

# Expand border of image so that segmented image is a square
# (it has a 1:1 aspect ratio)
def get_segmented_img(cropped_img, right_border_width, top_border_width, left_border_width, bottom_border_width):
    segmented_img = cv.copyMakeBorder(src=cropped_img,
                      top=top_border_width,
                      bottom=bottom_border_width,
                      right=right_border_width,
                      left=left_border_width,
                      borderType=cv.BORDER_CONSTANT,
                      value=WHITE)

    # Scale down image to 256x256
    down_points = (256, 256)
    downsized_img = cv.resize(src=segmented_img,
                              dsize=down_points,
                              interpolation=cv.INTER_LINEAR)
    return downsized_img


# Segment an image into 1 more sub-image, where each
# image contains a Lego piece
def smart_crop(img, dst_dir, is_test_flag):
    true_contours = []  # An array of contours that are enclosing a piece (and not just glare)
    max_border = 399 + 10  # Value determined experimentally
    img_mask = clear_background(img)
    img_mask_inv = cv.bitwise_not(img_mask)  # Create a mask of the background pixels
    contours = cv.findContours(image=img_mask_inv, mode=cv.RETR_EXTERNAL,
                               method=cv.CHAIN_APPROX_SIMPLE)[0]
    contours_descending_size = get_contours_sorted_by_descending_size(contours)
    i = 0
    file_name = 0
    while i < len(contours_descending_size):
        contour_index = contours_descending_size[i]
        contour = contours[contour_index]
        b_box_rect = cv.minAreaRect(contour)
        cropped_img = rotate_and_crop_img(img, contour)
        b_box_c_x, b_box_c_y = get_img_center(cropped_img)
        min_x, min_y = 0, 0
        b_box_width, b_box_height = get_bounding_box_size(b_box_rect)
        max_x, max_y = b_box_width, b_box_height
        area = b_box_width * b_box_height

        right_border_width, top_border_width, left_border_width, bottom_border_width = get_border_widths(max_border,
                                                                                                         min_x,
                                                                                                         max_x,
                                                                                                         min_y,
                                                                                                         max_y,
                                                                                                         b_box_c_x,
                                                                                                         b_box_c_y)

        if area < 21083:  # 380 x 90 (4x1 plate: smallest piece)
            break  # If the piece is so too small, skip the current contour and the rest
        # If there is a negative border, the piece its enclosing is a bunch of touching pieces
        # so break the loop and move on to the next piece
        if right_border_width > 0 \
                and top_border_width > 0 \
                and left_border_width > 0 \
                and bottom_border_width > 0 \
                and area < 218625:  # 795 x 275 (1x8x2 arch: biggest piece)
            true_contours.append(contour)  # Only if these two conditions are satisfied
            # is the piece enclosed truly a contour

            segmented_img = get_segmented_img(cropped_img,
                                              right_border_width,
                                              top_border_width,
                                              left_border_width,
                                              bottom_border_width)
            if is_test_flag:
                # Name using number
                img_dst_dir = rf'{dst_dir}/{str(file_name)}.png'
            else:
                # Name file depending on source filename
                img_dst_dir = rf'{dst_dir}-{str(file_name)}.png'
            cv.imwrite(img_dst_dir, segmented_img)
        else:
            file_name -= 1
        i += 1
        file_name += 1
    return true_contours
