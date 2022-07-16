import cv2 as cv
import numpy as np
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

    # def get_max_border(self):
    #     max_border = 0
    #     for label, element_id in self.enum_class_names:
    #         dir_element_id = rf'{self.dir_training_images}\{element_id}'
    #         list_img = os.listdir(dir_element_id)
    #         for img_name in tqdm(list_img):
    #             dir_img = rf'{dir_element_id}\{img_name}'
    #             img = cv.imread(dir_img)
    #             max_border = sc.find_max_border(max_border, min_x, min_y, max_x, max_y, c_x, c_y)
    #     return max_border

    def generate_dataset(self, dst_dir):
        max_border = 0
        flag = True
        for label, element_id in self.enum_class_names:
            dir_element_id = rf'{self.dir_training_images}\{element_id}'
            list_img = os.listdir(dir_element_id)
            for img_name in tqdm(list_img):
                dir_img = rf'{dir_element_id}\{img_name}'
                img = cv.imread(dir_img)

                contours_array = sc.get_contours(img)  # Only need first contour
                # Given a directory of images, crop image to have an aspect ratio of 1:1
                # and remove as much bg as possible
                if flag:  # If sub-folder has already been generated, do not re-generate
                    try:
                        os.mkdir(rf'{dst_dir}\{element_id}')
                    except OSError as error:
                        pass
                    dst_dir_square_img = rf'{dst_dir}\{element_id}\{img_name}'

                    # cv.imshow("img", img)
                    # cv.waitKey()
                    # cv.destroyAllWindows()

                    #print(dst_dir_square_img)

                    for contour in contours_array:
                        min_x, min_y, max_x, max_y, c_x, c_y = sc.get_bounding_box(contour)

                        if (((max_x - min_x) > 20) and ((max_y - min_y) > 20)):
                            img_g = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # Convert to greyscale

                            threshold = cv.threshold(img_g, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[0]
                            mask = cv.inRange(img_g, threshold, 255)  # Create a mask of all pixels where the
                            img_copy = img.copy()
                            img_copy[mask > threshold] = (255, 255, 255)
                            white = [255, 255, 255]
                            img_copy = cv.copyMakeBorder(img_copy, 100, 100, 100, 100, cv.BORDER_CONSTANT, None,
                                                    white)  # Add a white border
                            max_border = sc.find_max_border(max_border, min_x, min_y, max_x, max_y, c_x, c_y)
                            sc.smart_crop(img_copy, 214, dst_dir_square_img)  # 0 for only one object in image

                        self.training_dataset.append([img, label])  # More efficient to have list of NumPy arrays than a 2D NumPy array
        print(max_border)

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

    images_dir = r'E:\lego-classification-using-ml\real-training-images'  # src_dir
    squares_dir = r'E:\lego-classification-using-ml\square-real-training-images'  # dst_dir
    training_dataset = Dataset(images_dir)
    training_dataset.generate_dataset(squares_dir)
    #training_dataset.display_training_dataset()

if __name__ == '__main__':
    main()