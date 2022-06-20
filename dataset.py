import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import random
from tqdm import tqdm

__author__ = "Shahed E."
__date__ = "5/27/2022"
__status__ = "Development"

# Note: Give variables short lives; define variables close to where you plan on using them
# Note 2: Keep all class variables private


class Dataset:
    # Private class variables
    images_dir = None
    masks_dir = None
    list_class_names = None
    shuffle = False
    list_training_dataset = []

    def __init__(self, images_dir, masks_dir, shuffle):
        # Class Variables
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.shuffle = shuffle
        self.get_class_names()
        self.generate_training_dataset()

    def __getitem__(self, image_dir, mask_dir, index):
        image = cv2.imread(image_dir)
        image_greyscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        np_array_image = np.array(image_greyscale, dtype='float32')  # Convert image into numpy array
        mask = cv2.imread(mask_dir)
        np_array_mask = np.array(mask)  # Convert mask into numpy array
        np_array_image, __np_array_mask = self.normalize(np_array_image, np_array_mask, index)
        return np_array_image, __np_array_mask

    def get_np_array_images(self):
        return self.list_training_dataset[0]

    def get_np_array_masks(self):
        return self.list_training_dataset[1]

    def normalize(self, np_array_image, np_array_mask, index):
        np_array_image = np_array_image
        np_array_mask[np_array_mask == 255.000] = index + 1  # Consider 0 as the background class
        # --> Creates an input segmentation map such that every pixel
        # corresponding to a white pixel is given a class number
        # Therefore, background class remains == 0
        return np_array_image, np_array_mask

    def get_class_names(self):
        try:
            self.list_class_names = os.listdir(self.images_dir)
        except FileNotFoundError as error:
            print(error)

    def generate_training_dataset(self):
        try:
            for class_name in self.list_class_names:
                class_name_path = os.path.join(self.images_dir, class_name)  # Go through folder of every piece
                index = self.list_class_names.index(class_name)  # 0, 1, 2, ..., 9
                for image_name in tqdm(os.listdir(class_name_path)):
                    try:
                        image_dir = os.path.join(class_name_path, image_name)
                        mask_dir = os.path.join(self.masks_dir, f"{class_name}\mask-{image_name}")
                        self.list_training_dataset.append(self.__getitem__(image_dir, mask_dir, index))
                    except Exception as error:  # Ignore corrupted images
                        print(error)
                        pass
        except TypeError as error:
            print(error)

        if self.shuffle:
            random.shuffle(self.list_training_dataset)
        return

def main():
    images_dir = r"D:\lego-classification-using-ml\training-images"  # Convert to raw string to ignore /t escape key
    masks_dir = r'D:\lego-classification-using-ml\training-masks'
    training_dataset = Dataset(images_dir, masks_dir, shuffle=True)
    print(training_dataset.get_np_array_images())
if __name__ == '__main__':
    main()
