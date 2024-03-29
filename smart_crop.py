import cv2 as cv
import numpy as np
import typing  # For type hinting

"""
A module that is used to pre-process testing and training images.

Attributes
----------
WHITE: list
    White colour in GBR colour model.
BLACK: list
    Black colour in GBR colour model. 


Methods
-------
get_bounding_box_size(b_box_rect):
    Returns the width and height of the bounding box.
    
get_bounding_box_center(b_box_rect):
    Returns the coordinates of the center of the bounding box.

get_bounding_box_corners(b_box):]
    Returns the minimum x, minimum y, maximum x, and maximum y value of
    the bounding box's coordinates.
    
get_img_size(img):
    Returns the width and height of the image.

get_img_center(img):
    Returns the coordinates of the center of the image. 
    
get_contours_sorted_by_descending_size(contours):
    Returns a list of the index of the contours in the initial unsorted tuple of contours
    after they have been sorted in descending bounding box size. 

get_rotated_and_cropped_image(img, b_box_rect):
     Rotates the input image so that the piece of interest is upright and then crops the piece of
     interest out of the rotated image and returns the cropped image. 

clear_background(img):
    Sets the background pixels of the image to white using HSV thresholding.
    
get_border_widths(max_border, b_box_max_corner_vals, b_box_c_x, b_box_c_y):
    Returns the widths of the borders needed to expand the borders of the cropped image
    so that it has a square aspect ratio.

get_segmented_img(img, border_widths):
    Returns the cropped image after it has been scaled to the largest piece and resized to be 256 x 256.

get_segmented_imgs(img, max_border, dst_dir, is_test_img):
    Saves a segmented image of each piece in an unsegmented image to file; Returns a list of contours of each
    piece in the unsegmented image.
"""


WHITE = [255, 255, 255]  # GBR colours
BLACK = [0, 0, 0]


def get_bounding_box_size(b_box_rect: tuple[tuple[float, float], tuple[float, float], float]) -> tuple[int, int]:
    """
    Returns the width and height of the bounding box.

    Parameters
    ----------
    b_box_rect : b_box_rect: tuple[tuple[float, float], tuple[float, float], float]
        Bounding box enclosed by the contour.

    Returns
    -------
    b_box_width, b_box_height: tuple[int, int]
        The width and height of the bounding box, respectively.
    """
    b_box_width = int(b_box_rect[1][0])  # Cols (width)
    b_box_height = int(b_box_rect[1][1])  # Rows (height)
    return b_box_width, b_box_height


def get_bounding_box_center(b_box_rect: tuple[tuple[float, float], tuple[float, float], float]) -> tuple[int, int]:
    """
     Returns the coordinates of the center of the bounding box.

     Parameters
     ----------
     b_box_rect : b_box_rect: tuple[tuple[float, float], tuple[float, float], float]
         Bounding box enclosed by the contour.

     Returns
     -------
     b_box_c_x, b_box_c_y: tuple[int, int]
        The x-coordinate and y-coordinate of the center of the bounding box, respectively.
     """
    b_box_c_x = int(b_box_rect[0][0])
    b_box_c_y = int(b_box_rect[0][1])
    return b_box_c_x, b_box_c_y


def get_bounding_box_corners(b_box: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> tuple[int, int, int, int]:
    """
     Returns the minimum x, minimum y, maximum x, and maximum y value of
     the bounding box's coordinates.

     Parameters
     ----------
     b_box : tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
         A numpy array of each coordinate of the four corners of the bounding box.

     Returns
     -------
     min_x, min_y, max_x, max_y: tuple[int, int]
        The minimum x, minimum y, maximum x, and maximum y value of
        the coordinates of the corners of the bounding box respectively.
     """
    corner_coords = np.int0(b_box)[0]  # Array of coordinates of the four corners of the bounding box
    corner_coords[corner_coords < 0] = 0  # Set negative points to be equal to zero
    min_x = corner_coords[1][1]
    max_x = corner_coords[0][1]
    min_y = corner_coords[1][0]
    max_y = corner_coords[2][0]
    return min_x, max_x, min_y, max_y


def get_img_size(img: np.ndarray) -> tuple[int, int]:
    """
     Returns the width and height of an image.

     Parameters
     ----------
     img : np.ndarray
         A numpy array of the image.

     Returns
     -------
     img_width, img_height: tuple[int, int]
        Returns the width and height of an image, respectively.
     """
    img_height = int(img.shape[0])  # Rows (height)
    img_width = int(img.shape[1])  # Cols (width)
    return img_width, img_height


def get_img_center(img: np.ndarray) -> tuple[int, int]:
    """
     Returns the coordinates of the center of the image.

     Parameters
     ----------
     img : np.ndarray
         A numpy array of the image.

     Returns
     -------
     img_c_x, img_c_y: tuple[int, int]
        The x-coordinate and y-coordinate of the center of the image, respectively.
     """
    img_width, img_height = get_img_size(img)
    img_c_x = img_width // 2  # // is Integer division
    img_c_y = img_height // 2
    return img_c_x, img_c_y


def get_contours_sorted_by_descending_size(contours: tuple[np.ndarray, ...]) -> list:
    """
    Returns a list of the index of the contours in the initial unsorted tuple of contours
    after they have been sorted in descending bounding box size.

    Parameters
    ----------
    contours: tuple[np.ndarray, ...]
        A tuple of numpy arrays, where each array is an array of points making up the contour.

    Returns
    -------
    contours_descending_size: list
        A list of the index of the contours in the initial unsorted tuple of contours
        after they have been sorted in descending bounding box size.
    """
    b_box_areas = []
    for contour in contours:
        b_box_rect = cv.minAreaRect(contour)
        (b_box_width, b_box_height) = get_bounding_box_size(b_box_rect)
        area = b_box_width * b_box_height
        b_box_areas.append(area)
    sorted_b_boxes = sorted(list(enumerate(b_box_areas)),
                            key=lambda x: x[1],
                            reverse=True)
    # The lambda function allows for the sorted_b_boxes list
    # to be sorted by the size of the bounding box
    # while maintaining the index
    # of the contour in the initial unsorted tuple

    contours_descending_size = []  # A list of the index of the contours in the initial list
    for i in range(len(sorted_b_boxes)):
        contours_descending_size.append(sorted_b_boxes[i][0])  # Only need to grab the index, not the area
    return contours_descending_size


def get_rotated_and_cropped_img(img: np.ndarray,
                        b_box_rect: tuple[tuple[float, float], tuple[float, float], float]) -> np.ndarray:
    """
    Rotates the input image so that the piece of interest is upright, and then crops the piece of
    interest out of the rotated image.

    Parameters
    ----------
    img: np.ndarray
        A numpy array of the image.
    b_box_rect: b_box_rect: tuple[tuple[float, float], tuple[float, float], float]
        Bounding box enclosed by a contour.

    Returns
    -------
    cropped_img: np.ndarray
        A numpy array of the cropped image.
        """
    img_width, img_height = get_img_size(img)
    img_c_x, img_c_y = get_img_center(img)
    b_box_angle = b_box_rect[2]  # The angle that the minimum bounding box is rotated at

    # Determine the rotation matrix needed to rotate
    # unsegmented image about its center
    rotation_matrix = cv.getRotationMatrix2D(center=(img_c_x, img_c_y),
                                             angle=b_box_angle,
                                             scale=1)
    cos_rotation_matrix = np.abs(rotation_matrix[0][0])  # X component of the rotation matrix
    sin_rotation_matrix = np.abs(rotation_matrix[0][1])  # Y component of the rotation matrix

    # New width/height of unsegmented image after rotation
    rotated_img_width = int((img_height * cos_rotation_matrix) +
                            (img_width * sin_rotation_matrix))
    rotated_img_height = int((img_height * sin_rotation_matrix) +
                             (img_width * cos_rotation_matrix))

    # Update components in the rotation matrix to include translation
    rotation_matrix[0][2] += (rotated_img_width // 2) - img_c_x
    rotation_matrix[1][2] += (rotated_img_height // 2) - img_c_y

    # Rotate/translate the image using the rotation matrix
    rotated_img = cv.warpAffine(src=img,
                                M=rotation_matrix,
                                dsize=(rotated_img_width, rotated_img_height),
                                borderMode=cv.BORDER_CONSTANT,
                                borderValue=WHITE)

    b_box = cv.boxPoints(b_box_rect)  # Obtain the coordinates of the corners of the bounding box

    # Rotate/translate the bounding box using the rotation matrix
    rotated_b_box = cv.transform(np.array([b_box]),
                                 rotation_matrix)
    min_x, max_x, min_y, max_y = get_bounding_box_corners(rotated_b_box)
    cropped_img = rotated_img[min_x:max_x,
                              min_y:max_y]  # Crop the rotated image so that
                                            # The cropped image only contains the piece of interest

    return cropped_img


# Set background colour to white
def clear_background(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Sets the background pixels of the image to white using HSV thresholding.

    Parameters
    ----------
    img: np.ndarray
        A numpy array of the image.

    Returns
    -------
    img_copy: np.ndarray
        A numpy array of the image with a white background.

    img_mask: np.ndarray
        A mask where the background pixels are black and the foreground pixels are white.
    """
    light_bg = np.array([179, 70, 255])  # HSV range of
    dark_bg = np.array([0, 0, 0])        # the bg (low saturation pixels)
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    img_mask = cv.inRange(src=img_hsv,
                          lowerb=dark_bg,
                          upperb=light_bg)
    img_mask = cv.bitwise_not(img_mask)  # Create a mask and set the background pixels to black
    img_copy = img.copy()  # To prevent background removal from overwriting original image
    img_copy[img_mask == 0] = WHITE
    return img_copy, img_mask


def get_border_widths(max_border: int,
                      b_box_max_corner_vals: tuple[int, int, int, int],
                      b_box_c_x: int,
                      b_box_c_y: int) -> tuple[int, int, int, int]:
    """
    Returns the widths of the borders needed to expand the borders of the cropped image
    so that it has a square aspect ratio.

    Parameters
    ----------
    max_border: int
        The border width of the largest piece; this is used to scale all the pieces relative
        to each other.
    min_x, max_x, min_y, max_y: tuple[int, int, int, int]
        The minimum x, minimum y, maximum x, and maximum y value of
        the coordinates of the corners of the bounding box respectively.

    Returns
    -------
    img_copy: np.ndarray
        A numpy array of the image with a white background.

    img_mask: np.ndarray
        A mask where the background pixels are black and the foreground pixels are white.
    """
    min_x, min_y, max_x, max_y = b_box_max_corner_vals
    right_border = max_border - (max_x - b_box_c_x)
    top_border = max_border - (b_box_c_y - min_y)
    left_border = max_border - (b_box_c_x - min_x)
    bottom_border = max_border - (max_y - b_box_c_y)
    return right_border, top_border, left_border, bottom_border


def get_segmented_img(cropped_img: np.ndarray,
                      border_widths: tuple[int, int, int, int]) -> np.ndarray:
    """
    Returns the cropped image after it has been scaled to the largest piece and resized to be 256 x 256.

    Parameters
    ----------
    cropped_img: np.ndarray
        A numpy array of the image after the piece has been cropped out of the initial image.
    border_widths: tuple[int, int, int, int]
        The minimum x, minimum y, maximum x, and maximum y value of
        the coordinates of the corners of the bounding box respectively.

    Returns
    -------
    segmented_img: np.ndarray
        A numpy array of the image after it has been scaled to the largest piece and resized to be 256 x 256.
    """
    right_border, top_border, left_border, bottom_border = border_widths
    segmented_img = cv.copyMakeBorder(src=cropped_img,
                                      top=top_border,
                                      bottom=bottom_border,
                                      right=right_border,
                                      left=left_border,
                                      borderType=cv.BORDER_CONSTANT,
                                      value=WHITE)

    # Scale down image to 256x256
    down_points = (256, 256)
    downsized_img = cv.resize(src=segmented_img,
                              dsize=down_points,
                              interpolation=cv.INTER_LINEAR)
    return downsized_img


def get_segmented_imgs(img : np.ndarray,
                max_border: int,
                dst_dir: str,
                is_test_flag: bool) -> list:
    """
    Saves a segmented image of each piece in an unsegmented image to file; Returns a list of contours of each
    piece in the unsegmented image.

    Parameters
    ----------
    img: np.ndarray
        A numpy array of the unsegmented image.
    max_border: int
        The border width of the largest piece.
    dst_dir: str
        The directory where the segmented images will be saved.
    is_test_flag: bool
        A flag used to determine if the input image is a training image or a testing image; if it is a training image,
        then name the processed image according to the name of the source image; otherwise, name it using a number in
        chronological order.

    Returns
    -------
    true_contours: list
        A list of all the contours that are enclosing a piece. 
    """
    true_contours = []  # A list of the contours that are actually enclosing a piece
                        # Sometimes; a bounding box is generated for glare; ignore these
    img, img_mask = clear_background(img)   # Change the background to white
    contours = cv.findContours(image=img_mask, mode=cv.RETR_EXTERNAL,
                               method=cv.CHAIN_APPROX_SIMPLE)[0]
    contours_descending_size = get_contours_sorted_by_descending_size(contours)  # Sort the contour indices in order of
                                                                                 # descending size
    file_name = 0
    for contour_index in contours_descending_size:
        contour = contours[contour_index]
        b_box_rect = cv.minAreaRect(contour)
        cropped_img = get_rotated_and_cropped_img(img, b_box_rect)  # Crop the piece out of the image (no bg)
        b_box_c_x, b_box_c_y = get_img_center(cropped_img)
        b_box_width, b_box_height = get_bounding_box_size(b_box_rect)
        area = b_box_width * b_box_height
        b_box_max_corner_vals = (0, 0, b_box_width, b_box_height)  # The coordinates of the corners of the bounding box
        right_border, top_border, left_border, bottom_border = get_border_widths(max_border,
                                                                                 b_box_max_corner_vals,
                                                                                 b_box_c_x,
                                                                                 b_box_c_y)

        if area < 21083:  # 21083 = 380 x 90 (4x1 plate: smallest piece)
            break  # If the piece is so too small, skip the current contour and the rest of the contours

        # If there is a negative border, the piece its enclosing is a bunch of touching pieces
        # so break the loop and move on to the next piece
        if right_border > 0 and top_border > 0 and left_border > 0 and bottom_border > 0 and area < 218625:
            # 218625 = 795 x 275 (1x8x2 arch: biggest piece)
            true_contours.append(contour)  # Only if these two conditions are satisfied
                                           # is the piece enclosed truly a contour

            # Increase the border of the image so that all images have a square aspect ratio
            border_widths = right_border, top_border, left_border, bottom_border
            segmented_img = get_segmented_img(cropped_img,
                                              border_widths)
            if is_test_flag:
                # Name using number (0, 1, 2...)
                img_dst_dir = rf'{dst_dir}/{str(file_name)}.png'
            else:
                # Name file depending on source filename
                img_dst_dir = rf'{dst_dir}-{str(file_name)}.png'
            cv.imwrite(img_dst_dir, segmented_img)
        else:  # If the area of the bounding box is too big, it's enclosing several pieces, so skip
            file_name -= 1  # Since a piece was skipped, reduce the contour so that the saved images are numbered
                            # chronologically
        file_name += 1
    return true_contours
