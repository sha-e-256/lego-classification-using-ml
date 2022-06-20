import argparse
import cv2 as cv
import numpy as np
import os
import smart_crop as sc
import random
from tqdm import tqdm

__author__ = "Shahed E."
__date__ = "6/11/2022"
__status__ = "Development"

class Dataset:
    # Class variables (belong to the class)

    def __init__(self, dir_training_images):
        # Instance variables (belong to each object)
        self.dir_training_images = dir_training_images
        self.enum_class_names = self.get_class_names(self.dir_training_images)

    def get_class_names(self, dir_training_images):
        try:
            list_element_id = os.listdir(self.dir_training_images)
            enum_element_id = enumerate(list_element_id)
        except FileNotFoundError as error:
            print(error)
        return enum_element_id

    def get_np_array_img(self, img):
        np_array_img = np.array(img, dtype='float32')
        return np_array_img

    def traverse_dir_training_images(self, dst_dir, op):
        flag = True
        for label, element_id in self.enum_class_names:
            dir_element_id = rf'{self.dir_training_images}\{element_id}'
            list_img = os.listdir(dir_element_id)
            for img_name in tqdm(list_img):
                dir_img = rf'{dir_element_id}\{img_name}'
                img = cv.imread(dir_img)

                if op == 0:
                    # Given a directory of images, crop image to have an aspect ratio of 1:1
                    # and remove as much bg as possible
                   if flag:
                        try:
                            os.mkdir(rf'{dst_dir}\{element_id}')
                        except OSError as error:
                            pass
                        square_img = sc.smart_crop(img)
                        dst_dir_img = rf'{dst_dir}\{element_id}\{img_name}'
                        # cv.imshow('square', square_img)
                        # cv.waitKey(0)
                        # cv.destroyAllWindows()
                        cv.imwrite(dst_dir_img, square_img)

                if op == 1:
                    # Create training dataset
                    np_array_img = self.get_np_array_img(img)
                    self.features.append(np_array_img)
                    #self.labels.append(label)


def main():

    images_dir = r'D:\lego-classification-using-ml\training-images'  # src_dir
    squares_dir = r'D:\lego-classification-using-ml\square-training-images'  # dst_dir
    training_dataset = Dataset(images_dir)
    training_dataset.traverse_dir_training_images(squares_dir, 0)



if __name__ == '__main__':
    main()