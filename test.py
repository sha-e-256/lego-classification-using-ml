import os
import time
import cv2 as cv
import smart_crop as sc
import numpy as np
import random
import tensorflow as tf
import math
from tensorflow import keras
import subprocess
import typing

"""
A module that is used to perform the demonstration. This module takes an image using the Raspberry Pi camera,
segments that image, and then labels the initial unsegmented image with the predicted label, the probability of that
label, and then draws bounding boxes. 

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

"""


BLACK = [0, 0, 0]
WHITE = [255, 255, 255]


def get_prediction_and_probability(img: np.ndarray,
                                   model) -> tuple[str, float]:
    """
        Returns the name of the predicted piece and the probability of that prediction

        Parameters
        ----------
        img : np.ndarray
            Numpy array of the segmented image.
        model:  keras.engine.sequential.Sequential
            Model used to generated predictions.

        Returns
        -------
        predicted_name, probability: tuple[str, float]
            The predicted name of the piece and the probability of that prediction, respectively.
        """
    probability_array = model.predict(img)[0]  # 1 x 8 array of the predictions
    max_probability = np.max(probability_array)
    piece_names = np.array(["Plate 4 x 4 Corner", "Brick 1 x 6", "Plate 2 x 3", "Plate 2 x 8",
                   "Brick Arch 1 x 8 x 2", "Plate 1 x 6", "Plate 1 x 4", "Wedge Plate 4 x 2"])
    predicted_name = piece_names[probability_array == max_probability][0]
    return predicted_name, max_probability


# Draws text at the center of each box; used to display the prediction & probabilities
def draw_text_with_outline(colour, img, center, text):
    outline_thickness = 13
    text_thickness = 3
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    text_width, text_height = cv.getTextSize(text=text,
                                             fontFace=font,
                                             fontScale=font_scale,
                                             thickness=outline_thickness)[0]
    c_x, c_y = center
    text_min_x = c_x - text_width // 2
    text_min_y = c_y + text_height // 2
    # Outline
    cv.putText(img=img, text=text,
               org=(text_min_x, text_min_y),
               fontFace=font,
               fontScale=font_scale, color=WHITE,
               thickness=outline_thickness,
               lineType=cv.LINE_AA)
    # Text
    cv.putText(img=img, text=text,
               org=(text_min_x, text_min_y),
               fontFace=cv.FONT_HERSHEY_SIMPLEX,
               fontScale=font_scale, color=BLACK,
               thickness=text_thickness,
               lineType=cv.LINE_AA)


def clear_dir(dst_dir):
    for file_name in os.listdir(dst_dir):
        os.remove(os.path.join(dst_dir, file_name))


def update_dictionary(results, i, prediction, probability, segmented_img_b_box, segmented_img_b_box_center):
    results[i] = {}
    results[i]['label'] = prediction
    results[i]['probability'] = probability
    results[i]['bounding_box'] = segmented_img_b_box
    (segmented_img_b_box_c_x, segmented_img_b_box_c_y) = segmented_img_b_box_center
    results[i]['c_x'] = segmented_img_b_box_c_x
    results[i]['c_y'] = segmented_img_b_box_c_y
    return results


def normalize_img(img_dir):
    segmented_img = cv.imread(img_dir, cv.IMREAD_GRAYSCALE)  # NumPy array of the segmented image
    display_fullscreen_img(segmented_img)
    segmented_img_tensor = np.array(segmented_img).reshape(-1, 256, 256, 1)
    segmented_img_normalized = segmented_img_tensor / 255.0
    return segmented_img_normalized


def display_fullscreen_img(img):
    img_w_border, _, _= add_border(img=img, lcd_width=1920, lcd_height=1080, scale_factor=1)
    cv.namedWindow('img', cv.WND_PROP_FULLSCREEN)
    cv.setWindowProperty('img', cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
    cv.imshow('img', img_w_border)
    cv.waitKey()


def add_border(img, lcd_width, lcd_height, scale_factor):
    img_height = int(img.shape[0])
    img_width = int(img.shape[1])

    # Border must be positive; increase scale factor until its positive
    # This is just to prevent an exception from being thrown
    while True:
        top = (lcd_height - (img_height // scale_factor)) // 2 * scale_factor
        bottom = (lcd_height - (img_height // scale_factor)) // 2 * scale_factor
        right = (lcd_width - (img_width // scale_factor)) // 2 * scale_factor
        left = (lcd_width - (img_width // scale_factor)) // 2 * scale_factor
        if top < 0 or bottom < 0 or right < 0 or left < 0:
            scale_factor += 1
        else:
            break

    unsegmented_img_w_border = cv.copyMakeBorder(src=img,
                                                 top=top,
                                                 bottom=bottom,
                                                 right=right,
                                                 left=left,
                                                 borderType=cv.BORDER_CONSTANT,
                                                 value=WHITE)
    border = top, bottom, right, left
    return unsegmented_img_w_border, border, scale_factor

def draw_on_unsegmented_img(unsegmented_img_w_border, right_border, top_border, results):

    for j in range(len(results)):
        segmented_img_b_box = results[j]['bounding_box']
        segmented_img_b_box_c_x = results[j]['c_x']
        segmented_img_b_box_c_y = results[j]['c_y']
        segmented_img_label = results[j]['label']
        segmented_img_probability = '{0:.2%}'.format(results[j]['probability'])  # ADDED IN!!! JUL21-3AM
        # Adding a border causes the bounding boxes to be in the incorrect position
        # Therefore they need to be translated to the correct position by adding the offset
        # generated by the border
        translated_segmented_img_b_box_c_x = segmented_img_b_box_c_x + right_border  # right = left
        translated_segmented_img_b_box_c_y = segmented_img_b_box_c_y + top_border  # top = bottom
        translated_segmented_img_b_box = segmented_img_b_box.copy()
        for k in range(4):
            translated_segmented_img_b_box[k] = translated_segmented_img_b_box[k] + (right_border, top_border)

        color_vals = [0, 256]
        colour = (color_vals[random.randint(0, 1)],
                  color_vals[random.randint(0, 1)],
                  color_vals[random.randint(0, 1)])
        # Draw contour of bounding box on unsegmented image in a random colour
        cv.drawContours(unsegmented_img_w_border, [translated_segmented_img_b_box], 0, colour, 8)

        draw_text_with_outline(colour=colour,
                               img=unsegmented_img_w_border,
                               text=segmented_img_label,
                               center=(translated_segmented_img_b_box_c_x,
                                       translated_segmented_img_b_box_c_y - 30))

        draw_text_with_outline(colour=colour,
                               img=unsegmented_img_w_border,
                               text=segmented_img_probability,
                               center=(translated_segmented_img_b_box_c_x,
                                       translated_segmented_img_b_box_c_y + 30))
    return unsegmented_img_w_border

def run_test(src_dir, dst_dir, model):
    max_border = 399 + 10  # Value determined experimentally
    clear_dir(dst_dir)  # Clear the directory containing segmented testing images
    results = {}  # A dict containing the prediction, probability of prediction, coordinates
    # of the four corners of the bounding box of each piece,
    # and the center coordinates of each piece

    unsegmented_img_src_dir = os.path.join(src_dir, '0.png')
    unsegmented_img = cv.imread(unsegmented_img_src_dir)  # The image of the scattered Lego pieces
                                                          # taken by the camera
    # Generate segmented images and place them in the destination directory
    # and return a list of the contours enclosing a piece (the 'true contours') in the unsegmented image
    true_contours = sc.get_segmented_imgs(unsegmented_img, max_border, dst_dir, is_test_flag=True)

    segmented_img_names = sorted(os.listdir(dst_dir))
    for i in range(len(segmented_img_names)):
        file_name = str(i) + '.png'

        segmented_img_src_dir = os.path.join(dst_dir, file_name)  # Path of the segmented image
        segmented_img = normalize_img(img_dir=segmented_img_src_dir)


        prediction, probability = get_prediction_and_probability(segmented_img, model)

        # Once a prediction has been determined, update the dictionary
        segmented_img_rect = cv.minAreaRect(true_contours[i])
        (segmented_img_b_box_center) = sc.get_bounding_box_center(segmented_img_rect)
        segmented_img_b_box = np.int0(cv.boxPoints(segmented_img_rect))  # Coordinates of the four corners

        results = update_dictionary(results, i, prediction, probability,
                                    segmented_img_b_box,
                                    segmented_img_b_box_center)

        unsegmented_img_w_border, border, scale_factor = add_border(img=unsegmented_img,
                                                                    lcd_width=1366,
                                                                    lcd_height=768,
                                                                    scale_factor=2)
        top_border, bottom_border, left_border, right_border = border

    unsegmented_img_w_border = draw_on_unsegmented_img(unsegmented_img_w_border=unsegmented_img_w_border,
                                                       right_border=right_border,
                                                       top_border=top_border,
                                                       results=results)
    img_height = int(unsegmented_img_w_border.shape[0])
    img_width = int(unsegmented_img_w_border.shape[1])

    # Resize image so that it fits on the monitor
    down_points = (img_width // scale_factor, img_height // scale_factor)
    unsegmented_img_downsized = cv.resize(src=unsegmented_img_w_border,
                                          dsize=down_points,
                                          interpolation=cv.INTER_LINEAR)

    # Display image with bounding box information
    display_fullscreen_img(unsegmented_img_downsized)


def main():

    dst_dir = rf'D:\lego-classification-using-ml\segmented-testing-images'
    src_dir = rf'D:\lego-classification-using-ml\testing-images'
    model = tf.keras.models.load_model(rf'D:\lego-classification-using-ml\neural_net')

    # while True:
    #     cap = cv.VideoCapture(0)
    #     cap.set(3, 2048)  # width = 2048
    #     cap.set(4, 2048)  # height = 2048
    #     ret, frame = cap.read()
    #     frame = cv.rotate(frame, cv.ROTATE_90_COUNTERCLOCKWISE)
    #     display_fullscreen_img(frame)
    #     if cv.waitKey(1) & 0xFF == ord('y'):
    #         break
    #
    # cap.release()
    # cv.destroyAllWindows()
    #
    # cap = cv.VideoCapture(0)
    # cap.set(3, 2048)  # width=2048
    # cap.set(4, 2048)  # height=2048
    #
    # if cap.isOpened():
    #     _, frame = cap.read()
    #     frame = cv.rotate(frame, cv.ROTATE_90_COUNTERCLOCKWISE)
    #     cap.release()  # releasing camera immediately after capturing picture
    #     if _ and frame is not None:
    #         cv.imwrite(rf'D:\lego-classification-using-ml\testing-images/testing-images/0.png', frame)
    run_test(src_dir, dst_dir, model)


if __name__ == '__main__':
    main()