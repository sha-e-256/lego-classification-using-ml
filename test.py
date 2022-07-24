import os
import cv2 as cv
import smart_crop as sc
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
import subprocess

# The objective of this program is to take an image of a pile of scattered pieces
# and segment it into several images
# The coordinates of the bounding box and the centroid of each Lego piece
# will be saved in a JSON file

BLACK = [0, 0, 0]
WHITE = [255, 255, 255]

# subprocess.run(["/home/lewocp/livefeed.sh"])
model = tf.keras.models.load_model(rf'D:\lego-classification-using-ml\neural_net')

# model.summary()

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


def main():
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
    def run_test(src_dir, dst_dir):
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

            # cv.imshow('sample_Lego', segmented_img)
            # cv.waitKey(5)
            segmented_img = np.array(segmented_img).reshape(-1, 256, 256, 1)
            segmented_img = segmented_img / 255.0

            # ADD CODE HERE:
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

        for j in range(len(results)):
            segmented_img_b_box = results[j]['bounding_box']
            segmented_img_b_box_c_x = results[j]['c_x']
            segmented_img_b_box_c_y = results[j]['c_y']
            segmented_img_label = results[j]['label']
            segmented_img_probability = '{0:.3}'.format(results[j]['probability'])  # ADDED IN!!! JUL21-3AM

            colour = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))
            # Draw contour of bounding box on unsegmented image in a random colour
            cv.drawContours(unsegmented_img_copy, [segmented_img_b_box], 0, colour, 10)

            draw_text_with_outline(img=unsegmented_img_copy,
                                   text=segmented_img_label,
                                   center=(segmented_img_b_box_c_x, segmented_img_b_box_c_y))

            draw_text_with_outline(img=unsegmented_img_copy,
                                   text=segmented_img_probability,
                                   center=(segmented_img_b_box_c_x, segmented_img_b_box_c_y + 40))

        img_height = int(unsegmented_img_copy.shape[0])
        img_width = int(unsegmented_img_copy.shape[1])
        down_points = (img_width // 5, img_height // 5)  # Resize image so it fits on monitor
        unsegmented_img_downsized = cv.resize(src=unsegmented_img_copy,
                                              dsize=down_points,
                                              interpolation=cv.INTER_LINEAR)
        sc.true_contours.clear()
        # Display image with bounding box information
        winname = "unsegmented img with bounding boxes"
        cv.imshow(winname, unsegmented_img_downsized)
        cv.waitKey()
        cv.destroyAllWindows()

    dst_dir = rf'D:\lego-classification-using-ml\segmented-testing-images'
    src_dir = rf'D:\lego-classification-using-ml\testing-images'

    #subprocess.run(["/home/lewocp/take_pic.sh"])
    run_test(src_dir, dst_dir)


if __name__ == '__main__':
    main()