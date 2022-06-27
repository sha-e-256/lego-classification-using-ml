import cv2 as cv
from IPython.display import display, Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import smart_crop as sc
import random
from tqdm import tqdm

__author__ = "Shahed E."

class Dataset:
    # Class variables (belong to the class)

    def __init__(self, dir_training_images):
        # Instance variables (belong to each object)
        self.dir_training_images = dir_training_images
        self.enum_class_names = self.get_class_names(self.dir_training_images)
        self.training_dataset = []
        self.features = []
        self.labels = []

    def get_class_names(self, dir_training_images):
        try:
            list_element_id = os.listdir(self.dir_training_images)
            enum_element_id = enumerate(list_element_id)
        except FileNotFoundError as error:
            print(error)
        return enum_element_id

    def get_max_border(self):
        max_border = 0
        for label, element_id in self.enum_class_names:
            dir_element_id = rf'{self.dir_training_images}\{element_id}'
            list_img = os.listdir(dir_element_id)
            for img_name in tqdm(list_img):
                dir_img = rf'{dir_element_id}\{img_name}'
                img = cv.imread(dir_img)
                max_border = sc.find_max_border(max_border, min_x, min_y, max_x, max_y, c_x, c_y)
        return max_border

    def generate_dataset(self, dst_dir):
        flag = True
        for label, element_id in self.enum_class_names:
            dir_element_id = rf'{self.dir_training_images}\{element_id}'
            list_img = os.listdir(dir_element_id)
            for img_name in tqdm(list_img):
                dir_img = rf'{dir_element_id}\{img_name}'
                img = cv.imread(dir_img)
                # Given a directory of images, crop image to have an aspect ratio of 1:1
                # and remove as much bg as possible
                if flag:  # If sub-folder has already been generated, do not re-generate
                    try:
                        os.mkdir(rf'{dst_dir}\{element_id}')
                    except OSError as error:
                        pass
                    square_img = sc.smart_crop(75, img)  # 0 for only one object in image
                    dst_dir_img = rf'{dst_dir}\{element_id}\{img_name}'
                    cv.imwrite(dst_dir_img, square_img)
                    self.training_dataset.append([img, label])  # More efficient to have list of NumPy arrays than a 2D NumPy array

        random.shuffle(self.training_dataset) # Shuffling allows the model to be trained more effectively
        self.features = self.training_dataset[0]
        self.labels = self.training_dataset[1]
        np.save('features.npy', self.features)
        np.save('labels.npy', self.labels)


    def display_training_dataset(self):
        pass
        # !!To do: Display features & labels arrays in a dataframe
        # dataframe = pd.DataFrame(self.labels)
        # pd.options.display.max_columns = 20
        # display(dataframe)

def main():

    images_dir = r'D:\lego-classification-using-ml\training-images'  # src_dir
    squares_dir = r'D:\lego-classification-using-ml\square-training-images'  # dst_dir
    training_dataset = Dataset(images_dir)
    training_dataset.generate_dataset(squares_dir)
    #training_dataset.display_training_dataset()

if __name__ == '__main__':
    main()