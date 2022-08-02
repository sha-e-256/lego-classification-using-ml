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


def get_prediction_and_probability(img, model):
    probability_array = model.predict(img)[0]  # 1 x 8 array of the predictions
    max_probability = np.max(probability_array)
    piece_names = np.array(["Plate 4 x 4 Corner", "Brick 1 x 6", "Plate 2 x 3", "Plate 2 x 8",
                   "Brick Arch 1 x 8 x 2", "Plate 1 x 6", "Plate 1 x 4", "Wedge Plate 4 x 2"])
    piece_name = piece_names[probability_array == max_probability]
    return piece_name[0], max_probability

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


# Create a dictionary that contains information on the bounding box coordinates
# and the model prediction & probabilities of each Lego piece so that this information
# can be drawn on the unsegmented image taken by the camera
def run_test(src_dir, dst_dir, model):
    max_border = 399 + 10  # Value determined experimentally

    for file_name in os.listdir(dst_dir):
        os.remove(os.path.join(dst_dir, file_name))  # Clear the directory containing segmented testing images
    results = {}  # A dict containing the prediction, probability of prediction, coordinates
    # of the four corners of the bounding box of each piece,
    # and the center coordinates of each piece
    unsegmented_img_src_dir = os.path.join(src_dir, '14.png')
    # Add exception, this image must exist
    unsegmented_img = cv.imread(unsegmented_img_src_dir)  # The image of the scattered Lego pieces
    # cv.imwrite(rf'E:\InterSummer 2022\ELEC-4000 Capstone\Final Report\report\0.png', unsegmented_img)

    # taken by the camera
    unsegmented_img_copy = unsegmented_img.copy()

    true_contours = sc.get_segmented_imgs(unsegmented_img, max_border, dst_dir, is_test_flag=True)  # Generate segmented images

    # and return a list of the contours enclosing a piece (the 'true contours') in the unsegmented_img
    segmented_img_names = sorted(os.listdir(dst_dir))
    for i in range(len(segmented_img_names)):
        file_name = str(i) + '.png'

        segmented_img_src_dir = os.path.join(dst_dir, file_name)  # Path of the segmented image
        segmented_img = cv.imread(segmented_img_src_dir, cv.IMREAD_GRAYSCALE)  # NumPy array of the segmented image

        segmented_img = np.array(segmented_img).reshape(-1, 256, 256, 1)
        segmented_img = segmented_img / 255.0

        # segmented_img: input to the model

        prediction, probability = get_prediction_and_probability(segmented_img, model)

        # Once prediction has been determined, update the dictionary
        segmented_img_rect = cv.minAreaRect(true_contours[i])
        (segmented_img_b_box_c_x, segmented_img_b_box_c_y) = sc.get_bounding_box_center(segmented_img_rect)
        segmented_img_b_box = np.int0(cv.boxPoints(segmented_img_rect))  # Coordinates of the four corners

        results[i] = {}
        results[i]['label'] = prediction
        results[i]['probability'] = probability
        results[i]['bounding_box'] = segmented_img_b_box
        results[i]['c_x'] = segmented_img_b_box_c_x
        results[i]['c_y'] = segmented_img_b_box_c_y

    img_height = int(unsegmented_img_copy.shape[0])
    img_width = int(unsegmented_img.shape[1])
    lcd_width = 1366
    lcd_height = 768
    scale_factor = 2
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

    unsegmented_img_copy_w_border = cv.copyMakeBorder(src=unsegmented_img_copy,
                                                      top=top,
                                                      bottom=bottom,
                                                      right=right,
                                                      left=left,
                                                      borderType=cv.BORDER_CONSTANT,
                                                      value=WHITE)

    # Scale down images to 256x256
    for j in range(len(results)):
        segmented_img_b_box = results[j]['bounding_box']
        segmented_img_b_box_c_x = results[j]['c_x']
        segmented_img_b_box_c_y = results[j]['c_y']
        segmented_img_label = results[j]['label']
        segmented_img_probability = '{0:.2%}'.format(results[j]['probability'])  # ADDED IN!!! JUL21-3AM
        # Adding a border causes the bounding boxes to be in the incorrect position
        # Therefore they need to be translated to the correct position by adding the offset
        # generated by the border
        translated_segmented_img_b_box_c_x = segmented_img_b_box_c_x + right  # right = left
        translated_segmented_img_b_box_c_y = segmented_img_b_box_c_y + top  # top = bottom
        translated_segmented_img_b_box = segmented_img_b_box.copy()
        for k in range(4):
            translated_segmented_img_b_box[k] = translated_segmented_img_b_box[k] + (right, top)

        color_vals = [0, 256]
        colour = (color_vals[random.randint(0, 1)],
                  color_vals[random.randint(0, 1)],
                  color_vals[random.randint(0, 1)])
        # Draw contour of bounding box on unsegmented image in a random colour
        cv.drawContours(unsegmented_img_copy_w_border, [translated_segmented_img_b_box], 0, colour, 8)

        draw_text_with_outline(colour=colour,
                               img=unsegmented_img_copy_w_border,
                               text=segmented_img_label,
                               center=(translated_segmented_img_b_box_c_x,
                                       translated_segmented_img_b_box_c_y - 30))

        draw_text_with_outline(colour=colour,
                               img=unsegmented_img_copy_w_border,
                               text=segmented_img_probability,
                               center=(translated_segmented_img_b_box_c_x,
                                       translated_segmented_img_b_box_c_y + 30))

    img_height = int(unsegmented_img_copy_w_border.shape[0])
    img_width = int(unsegmented_img_copy_w_border.shape[1])
    # Resize image so it fits on the monitor
    down_points = (img_width // scale_factor, img_height // scale_factor)
    print(f'display resolution: {down_points}')
    unsegmented_img_downsized = cv.resize(src=unsegmented_img_copy_w_border,
                                          dsize=down_points,
                                          interpolation=cv.INTER_LINEAR)
    cv.imwrite(rf'E:\InterSummer 2022\ELEC-4000 Capstone\Final Report\report\2.png', unsegmented_img_copy_w_border)

    # Display image with bounding box information
    window_name = "unsegmented img with bounding boxes"
    cv.imshow(window_name, unsegmented_img_downsized)
    cv.waitKey()
    cv.destroyAllWindows()


def main():

    dst_dir = rf'E:\lego-classification-using-ml\segmented-testing-images'
    src_dir = rf'E:\lego-classification-using-ml\testing-images'
    # subprocess.run(["D:\lego-classification-using-ml/livefeed.sh"])
    model = tf.keras.models.load_model(rf'E:\lego-classification-using-ml\neural_net')
    #subprocess.run(["D:\lego-classification-using-ml/take_pic.sh"])
    start_time = time.time()

    run_test(src_dir, dst_dir, model)

    end_time = time.time()
    #print(f'total time:{end_time - start_time}')

if __name__ == '__main__':
    main()