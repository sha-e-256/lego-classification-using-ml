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


# Sorts the contours based off of the size of the minimum bounding box enclosed by the contour
# And returns an array of the index of the contours in order of descending bounding box size
def get_index_largest_contours(contours_array):
    b_box_array = []
    for contour in contours_array:
        b_box_rect = cv.minAreaRect(contour)
        (b_box_width, b_box_height) = get_bounding_box_info(b_box_rect)[0]
        b_box_array.append((b_box_width, b_box_height))
    b_box_list = list(enumerate(b_box_array))
    sorted_b_box_list = sorted(b_box_list, key=lambda x: x[1], reverse=True)  # The lambda function allows for the list
    # to be sorted by the tuple with the largest
    # width and height, while keeping the index
    # of the the contour in the initial list

    contours_descending_order = []
    for i in range(len(sorted_b_box_list)):
        contours_descending_order.append(sorted_b_box_list[i][0])
    return contours_descending_order


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

    img_height = int(img.shape[0])   # Rows
    img_width = int(img.shape[1])  # Cols
    (b_box_width, b_box_height) = get_bounding_box_info(b_box_rect)[0]
    (b_box_c_x, b_box_c_y) = get_bounding_box_info(b_box_rect)[1]
    b_box_angle = get_bounding_box_info(b_box_rect)[2]

    # Rotate image about center of bounding box rectangle
    rotation_matrix = cv.getRotationMatrix2D(center=(img_width // 2,
                                                     img_height // 2),
                                             angle=b_box_angle, scale=1)
    # # Expand image border using lengths
    # img = cv.copyMakeBorder(src=img,
    #                                   top=img_height//2,
    #                                   bottom=img_height//2,
    #                                   right=img_width//2,
    #                                   left=img_width//2,
    #                                   borderType=cv.BORDER_CONSTANT,
    #                                   value=WHITE)

    rotated_img = cv.warpAffine(src=img, M=rotation_matrix,
                                dsize=(img_width, img_height),
                                borderMode=cv.BORDER_CONSTANT, borderValue=WHITE)

    test_img = rotated_img.copy()
    #cv.drawContours(test_img, true_contours, -1, [255, 0, 0], 5)  # Draw all contours
    test_img = cv.resize(src=test_img,
                    dsize=(1920 // 2, 1080 // 2),
                    interpolation=cv.INTER_LINEAR)
    cv.imshow('img', test_img)
    cv.waitKey()
    cv.destroyAllWindows()

    rotated_b_box_rect = (b_box_rect[0], b_box_rect[1], 0)
    b_box = cv.boxPoints(b_box_rect)
    pts = np.int0(cv.transform(np.array([b_box]), rotation_matrix))[0]
    pts[pts < 0] = 0

    # cropped_img = cv.getRectSubPix(image=rotated_img, patchSize=(b_box_width, b_box_height),
    #                                center=(b_box_c_x, b_box_c_y))

    cropped_img = rotated_img[pts[1][1]:pts[0][1], pts[1][0]: pts[2][0]]

    cv.imshow('img', cropped_img)
    cv.waitKey()
    cv.destroyAllWindows()


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
    img_g = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # Convert image to greyscale
    img_g_blur = cv.GaussianBlur(src=img_g, ksize=(13, 13), sigmaX=0)  # Kernel size has to be an odd number

    contours_array = get_contours_array(img_g_blur)[0]
    threshold = get_contours_array(img_g_blur)[1]
    num_pieces_index = 0  # Not all contours enclose a piece; num_pieces_index
    # keeps track of the index of a contour which does enclose a piece

    sorted_contour_indices = get_index_largest_contours(contours_array)
    for i, contour_index in enumerate(sorted_contour_indices):  # contours_array should be sorted prior to doing this
        b_box_rect = cv.minAreaRect(contours_array[contour_index])
        (b_box_width, b_box_height) = get_bounding_box_info(b_box_rect)[0]
        try:
            next_b_box_rect = cv.minAreaRect(contours_array[sorted_contour_indices[i + 1]])
            (next_b_box_width, next_b_box_height) = get_bounding_box_info(next_b_box_rect)[0]

            if b_box_width > 10 * next_b_box_width or b_box_height > 10 * next_b_box_height:
                true_contours.append(contours_array[  # Add the last true piece
                                         contour_index])
                break  # And ignore the rest of the contours
            else:
                true_contours.append(contours_array[
                                         contour_index])  # If the bounding box of the next piece
            # is significantly smaller than the current piece
        except IndexError as error:
            true_contours.append(contours_array[contour_index])  # If there are no more contours
            # to check against, then just dont check next contour

    # test_img = img.copy()
    # cv.drawContours(test_img, true_contours, -1, [255, 0, 0], 5)  # Draw all contours
    # test_img = cv.resize(src=test_img,
    #                 dsize=(1366 // 2, 768 // 2),
    #                 interpolation=cv.INTER_LINEAR)
    # cv.imshow('img', test_img)
    # cv.waitKey()
    # cv.destroyAllWindows()
    print(len(true_contours))
    for contour in true_contours:
        b_box_rect = cv.minAreaRect(contour)
        (b_box_width, b_box_height) = get_bounding_box_info(b_box_rect)[0]
        cropped_img = rotate_img(img, b_box_rect)  # Rotate the image with
        # cv.imshow('cropped_img', cropped_img)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
        # respect to the angle of each bounding box in the image
        (b_box_c_x, b_box_c_y) = get_bounding_box_info(b_box_rect)[1]
        b_box_angle = 0
        rotated_b_box_rect = ((b_box_c_x, b_box_c_y),  # Align bounding box
                              (b_box_width, b_box_height), b_box_angle)  # parallel to image borders
        max_length = b_box_width if b_box_width >= b_box_height else b_box_height  # The maximum length to extend the border

        # so that the image becomes square is the maximum of the width and the height

        (rotated_b_box_c_x, rotated_b_box_c_y) = get_bounding_box_info(rotated_b_box_rect)[1]
        (rotated_min_x, rotated_min_y,
         rotated_max_x, rotated_max_y) = get_bounding_box_info(rotated_b_box_rect)[3]

        # Center object in image by expanding border so all images
        # are scaled relative to each other
        # Determine lengths needed to expand image border
        if max_length == b_box_width:
            # Only add from the top
            top_border_width = max_length // 2 - b_box_height // 2 + 2
            bottom_border_width = max_length // 2 - b_box_height // 2 + 2
            left_border_width = 2  # Only add offset
            right_border_width = 2
        if max_length == b_box_height:
            # Only add from the sides
            top_border_width = 2
            bottom_border_width = 2
            left_border_width = max_length // 2 - b_box_width // 2 + 2
            right_border_width = max_length // 2 - b_box_width // 2 + 2

        # right_border_width = max_border - (rotated_max_x - rotated_b_box_c_x)
        # top_border_width = max_border - (rotated_b_box_c_y - rotated_min_y)
        # left_border_width = max_border - (rotated_b_box_c_x - rotated_min_x)
        # bottom_border_width = max_border - (rotated_max_y - rotated_b_box_c_y)

        # Expand image border using lengths
        segmented_img = cv.copyMakeBorder(src=cropped_img,
                                          top=top_border_width,
                                          bottom=bottom_border_width,
                                          right=right_border_width,
                                          left=left_border_width,
                                          borderType=cv.BORDER_CONSTANT,
                                          value=WHITE)
        # Determine threshold for each segmented image
        segmented_img_grey = cv.cvtColor(segmented_img, cv.COLOR_BGR2GRAY)
        segmented_threshold, _ = cv.threshold(src=segmented_img_grey, thresh=0, maxval=255,
                                              type=(cv.THRESH_BINARY_INV + cv.THRESH_OTSU))
        background = cv.inRange(src=segmented_img_grey, lowerb=threshold, upperb=255)
        segmented_img[background > segmented_threshold] = WHITE  # Change colour of all background pixels to white

        # Scale down images to 256x256
        down_points = (256, 256)
        img_downsized = cv.resize(src=segmented_img,
                                  dsize=down_points,
                                  interpolation=cv.INTER_LINEAR)

        if isTest:
            img_dst_dir = rf'{dst_dir}\{str(num_pieces_index)}.png'
        else:
            img_dst_dir = rf'{dst_dir}.png'
        cv.imwrite(img_dst_dir, img_downsized)
        num_pieces_index += 1
