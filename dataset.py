import argparse
import cv2 as cv
import numpy as np
import os
import random
from tqdm import tqdm

__author__ = "Shahed E."
__date__ = "6/11/2022"
__status__ = "Development"

class TrainingDataset:
    # Class variables (belong to the class)

    def __init__(self, dir_training_images):
        # Instance variables (belong to each object)
        self.dir_training_images = dir_training_images
        self.enum_class_names = self.get_class_names(self.dir_training_images)

    def get_class_names(self, dir_training_images):
        try:
            list_element_id = os.listdir(self.dir_training_images)
            enum_element_id  = enumerate(list_element_id)
        except FileNotFoundError as error:
            print(error)
        return enum_element_id

    def get_np_array_img(self, dir):
        img = cv.imread(dir, 0)  # 0: image is converted to greyscale upon import
        np_array_img = np.array(img, dtype='float32')
        return np_array_img

    def smart_crop(self, img):
        min_x, min_y, max_x, max_y, c_x, c_y = self.get_bounding_box(img)
        bounding_img = img[min_y:max_y, min_x:max_x]
        right = 70 - (max_x - c_x)
        top = 70 - (c_y - min_y)
        left = 70 - (c_x - min_x)
        bottom = 70 - (max_y - c_y)
        square_img = cv.copyMakeBorder(bounding_img, top, bottom, left, right, cv.BORDER_REPLICATE)
        return square_img

    def get_bounding_box(self, img):
        img_g = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        threshold, array_img = cv.threshold(img_g, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
        array_contours = cv.findContours(array_img, 1, 2)[0]
        contour = array_contours[0]
        x_min, y_min, width, height = cv.boundingRect(contour)  # Image only contains one contour
        offset = 2
        x_min -= offset
        y_min -= offset
        x_max = x_min + width + 2 * offset
        y_max = y_min + height + 2 * offset
        m = cv.moments(contour)
        c_x = int(m['m10'] / m['m00'])
        c_y = int(m['m01'] / m['m00'])
        return x_min, y_min, x_max, y_max, c_x, c_y

    def traverse_dir_training_images(self, op, dir_dst):
        for label, element_id in self.enum_class_names:
            dir_element_id = rf'{self.dir_training_images}\{element_id}'
            list_img = os.listdir(dir_element_id)
            for img_name in list_img:
                dir_img = rf'{dir_element_id}\{img_name}'
                img = cv.imread(dir_img)

                if op == 0:
                    # Given a directory of images, crop image to have an aspect ratio of 1:1
                    # and remove as much bg as possible
                    square_img = self.smart_crop(img)
                    cv.imwrite(dir_dst, square_img)

                if op == 1:
                    # Create training dataset
                    np_array_img = self.get_np_array_image(dir_img)
                    self.features.append(np_array_img)
                    #self.labels.append(label)


def main():

    images_dir = r'E:\lego-classification-using-ml\training-images'  # dir_src
    squares_dir = r'E:\lego-classification-using-ml\square-training-images'  #dir_dst
    training_dataset = TrainingDataset(images_dir)
    training_dataset.traverse_dir_training_images(0, squares_dir)



if __name__ == '__main__':
    main()