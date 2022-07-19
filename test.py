import os

import cv2 as cv
import smart_crop as sc
import numpy as np
import random


# The objective of this program is to take an image of a pile of scattered pieces
# and segment it into several images
# The coordinates of the bounding box and the centroid of each Lego piece
# will be saved in a JSON file

BLACK = [0, 0, 0]
WHITE = [255, 255, 255]

def main():

    # Create a dictionary that contains information on the bounding box coordinates
    # and the model prediction & probabilities of each Lego piece so that this information
    # can be drawn on the unsegmented image taken by the camera
    def run_test(src_dir, dst_dir):
        results = {}  # A dict containing the prediction, probability of prediction, coordinates
        # of the four corners of the bounding box of each piece,
        # and the center coordinates of each piece
        unsegmented_img_src_dir = rf'{src_dir}\0.png'
        unsegmented_img = cv.imread(unsegmented_img_src_dir)  # The image of the scattered Lego pieces
        # taken by the camera
        unsegmented_img_copy = unsegmented_img.copy()
        sc.rotate_and_square_crop(unsegmented_img, dst_dir)  # Generate segmented images
        true_contours = sc.get_true_contours()
        # and return a list of the contours enclosing a piece (the 'true contours') in the unsegmented_img
        segmented_img_names = os.listdir(dst_dir)
        segmented_img_names_enum = enumerate(segmented_img_names)
        for i, segmented_img_name in segmented_img_names_enum:
            segmented_img_src_dir = rf'{dst_dir}\{segmented_img_name}'  # Path of the segmented image
            segmented_img = cv.imread(segmented_img_src_dir)  # NumPy array of the segmented image

            # ADD CODE HERE:
            # segmented_img: input to the model
            prediction = 'Plate 4 x 4 Corner'  # prediction: output from model
            probability = 0.900  # 3 decimals  # probability: output from model
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
            colour = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))
            # Draw contour of bounding box on unsegmented image in a random colour
            cv.drawContours(unsegmented_img_copy, [segmented_img_b_box], 0, colour, 10)
            img_with_text = cv.putText(img=unsegmented_img_copy, text=prediction,
                                       org=(segmented_img_b_box_c_x, segmented_img_b_box_c_y),
                                       fontFace=cv.FONT_HERSHEY_SIMPLEX,
                                       fontScale=1, color=WHITE, thickness=3,
                                       lineType=cv.LINE_AA)
            img_height = int(unsegmented_img_copy.shape[0])
            img_width = int(unsegmented_img_copy.shape[1])
            down_points = (img_width//4, img_height//4)  # Resize image so it fits on monitor
            unsegmented_img_downsized = cv.resize(src=unsegmented_img_copy,
                                      dsize=down_points,
                                      interpolation=cv.INTER_LINEAR)

        # Display image with bounding box information
        cv.imshow("unsegmented img with bounding boxes", unsegmented_img_downsized)
        cv.waitKey()
        cv.destroyAllWindows()

    dst_dir = rf'D:\lego-classification-using-ml\segmented-testing-images'
    src_dir = rf'D:\lego-classification-using-ml\testing-images'

    run_test(src_dir, dst_dir)


if __name__ == '__main__':
    main()