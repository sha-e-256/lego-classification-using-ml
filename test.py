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

# The objective of this program is to take an image of a pile of scattered pieces
# and segment it into several images
# The coordinates of the bounding box and the centroid of each Lego piece
# will be saved in a JSON file

BLACK = [0, 0, 0]
WHITE = [255, 255, 255]


def main():

    def bounding_accuracy(image, model):
        # img = (np.expand_dims(image,0))
        my_prediction = model.predict(image)
        num_pred = my_prediction[0][0]
        length = len(my_prediction[0])
        for i in range(len(my_prediction[0])):
            if my_prediction[0][i] > num_pred:
                num_pred = my_prediction[0][i]

        return num_pred

    def predicted_class(image, model):
        # img = (np.expand_dims(image,0))
        my_prediction = model.predict(image)
        int_range = np.argmax(my_prediction)
        names = ["Plate 4 x 4 Corner", "Brick 1 x 6", "Plate 2 x 3", "Plate 2 x 8",
                 "Brick Arch 1 x 8 x 2", "Plate 1 x 6", "Plate 1 x 4", "Wedge Plate 4 x 2 Right"]
        label = names[int_range]

        return label

    # Draws text at the center of each box; used to display the prediction & probabilities
    def draw_text_with_outline(img, center, text):
        # Outline
        cv.putText(img=img, text=text,
                   org=center,
                   fontFace=cv.FONT_HERSHEY_SIMPLEX,
                   fontScale=1.5, color=BLACK, thickness=13,
                   lineType=cv.LINE_AA)
        # Text
        cv.putText(img=img, text=text,
                   org=center,
                   fontFace=cv.FONT_HERSHEY_SIMPLEX,
                   fontScale=1.5, color=WHITE, thickness=3,
                   lineType=cv.LINE_AA)

    # Create a dictionary that contains information on the bounding box coordinates
    # and the model prediction & probabilities of each Lego piece so that this information
    # can be drawn on the unsegmented image taken by the camera
    def run_test(src_dir, dst_dir, model):
        results = {}  # A dict containing the prediction, probability of prediction, coordinates
        # of the four corners of the bounding box of each piece,
        # and the center coordinates of each piece
        unsegmented_img_src_dir = os.path.join(src_dir, '0.png')
        unsegmented_img = cv.imread(unsegmented_img_src_dir)  # The image of the scattered Lego pieces
        # taken by the camera
        unsegmented_img_copy = unsegmented_img.copy()
        sc.rotate_and_square_crop(unsegmented_img, dst_dir, isTest=True)  # Generate segmented images
        true_contours = sc.true_contours
        # and return a list of the contours enclosing a piece (the 'true contours') in the unsegmented_img
        segmented_img_names = os.listdir(dst_dir)
        segmented_img_names_enum = enumerate(segmented_img_names)
        for i, segmented_img_name in segmented_img_names_enum:
            pathed = str(i) + ".png"
            # print(segmented_img_name)

            segmented_img_src_dir = os.path.join(dst_dir, pathed)  # Path of the segmented image
            segmented_img = cv.imread(segmented_img_src_dir, cv.IMREAD_GRAYSCALE)  # NumPy array of the segmented image

            segmented_img = np.array(segmented_img).reshape(-1, 256, 256, 1)
            segmented_img = segmented_img / 255.0

            # segmented_img: input to the model
            prediction = predicted_class(segmented_img, model)  # prediction: output from model
            probability = bounding_accuracy(segmented_img, model)  # 3 decimals  # probability: output from model

            # Once prediction has been determined, update the dictionary
            segmented_img_rect = cv.minAreaRect(true_contours[i])
            (segmented_img_b_box_c_x, segmented_img_b_box_c_y) = sc.get_bounding_box_info(segmented_img_rect)[1]
            segmented_img_b_box = np.int0(cv.boxPoints(segmented_img_rect))  # Coordinates of the four corners

            results[i] = {}
            results[i]['label'] = prediction
            results[i]['probability'] = probability
            results[i]['bounding_box'] = segmented_img_b_box
            results[i]['c_x'] = segmented_img_b_box_c_x
            results[i]['c_y'] = segmented_img_b_box_c_y

        img_height = int(unsegmented_img_copy.shape[0])
        img_width = int(unsegmented_img.shape[1])
        lcd_width = 1920
        lcd_height = 1080
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
                                                          borderType=cv.BORDER_REPLICATE,
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
            translated_segmented_img_b_box_c_x = segmented_img_b_box_c_x + right  #  right = left
            translated_segmented_img_b_box_c_y = segmented_img_b_box_c_y + top    # top = bottom
            translated_segmented_img_b_box = segmented_img_b_box.copy()
            for k in range(4):
                translated_segmented_img_b_box[k] = translated_segmented_img_b_box[k] + (right, top)

            colour = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))
            # Draw contour of bounding box on unsegmented image in a random colour
            cv.drawContours(unsegmented_img_copy_w_border, [translated_segmented_img_b_box], 0, colour, 10)

            draw_text_with_outline(img=unsegmented_img_copy_w_border,
                                   text=segmented_img_label,
                                   center=(translated_segmented_img_b_box_c_x,
                                           translated_segmented_img_b_box_c_y))

            draw_text_with_outline(img=unsegmented_img_copy_w_border,
                                   text=segmented_img_probability,
                                   center=(translated_segmented_img_b_box_c_x,
                                           translated_segmented_img_b_box_c_y + 60))

        img_height = int(unsegmented_img_copy_w_border.shape[0])
        img_width = int(unsegmented_img_copy_w_border.shape[1])
        # Resize image so it fits on the monitor
        down_points = (img_width // scale_factor, img_height // scale_factor)
        unsegmented_img_downsized = cv.resize(src=unsegmented_img_copy_w_border,
                                              dsize=down_points,
                                              interpolation=cv.INTER_LINEAR)
        sc.true_contours.clear()
        # Display image with bounding box information
        window_name = "unsegmented img with bounding boxes"
        cv.imshow(window_name, unsegmented_img_downsized)
        cv.waitKey()
        cv.destroyAllWindows()

    dst_dir = rf'D:\lego-classification-using-ml\segmented-testing-images'
    src_dir = rf'D:\lego-classification-using-ml\testing-images'
    # subprocess.run(["D:\lego-classification-using-ml/livefeed.sh"])
    model = tf.keras.models.load_model(rf'D:\lego-classification-using-ml\neural_net')
    #subprocess.run(["D:\lego-classification-using-ml/take_pic.sh"])
    start_time = time.time()

    run_test(src_dir, dst_dir, model)

    end_time = time.time()
    print(f'total time:{end_time - start_time}')

if __name__ == '__main__':
    main()