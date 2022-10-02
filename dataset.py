import os
import cv2 as cv  # openCV
import numpy as np
import smart_crop as sc  # Self-made library used to crop images

# The objective of this program is to pre-process images that will be used to
# generate a training dataset. To begin, the background of each image is
# removed, then the piece is centered in the image, and then the image is
# cropped to have a square (1:1) aspect ratio.

def main():

    def get_class_names(src_dir):
        try:
            class_names = os.listdir(src_dir)  # Names of subdirectories
            # in the source directory
        except FileNotFoundError as error:
            print(error)
        return class_names


    # *Check for not a directory error
    # Traverse through training image directories and pre-process images
    def process_images():
        flag = True
        src_dir = rf'D:\lego-classification-using-ml\real-training-images'
        dst_dir = rf'D:\lego-classification-using-ml\square-real-training-images'
        # directories, respectively
        class_names = get_class_names(src_dir)

        for class_name in class_names:
            class_name_dir = rf'{src_dir}\{class_name}'  # Path of each
            # subdirectory
            # r' ignores escape characters in string
            # For ex., \t is an escape character which creates an indent
            img_names = sorted(os.listdir(class_name_dir))  # Names of images in
            # subdirectory
            for img_name in img_names:
                img_src_dir = rf'{class_name_dir}\{img_name}'  # Path of image
                img = cv.imread(img_src_dir)  # np array of image
                # destination directory does not exist...
                if flag:
                    try:
                        os.mkdir(rf'{dst_dir}\{class_name}')  # ...Create the
                        # subdirectory
                    except OSError as error:
                        pass    # Do not generate a subdirectory if it already
                        # exists
                    img_dst_dir = rf'{dst_dir}\{class_name}\{img_name.split(".")[0]}'
                    max_border = 399 + 10  # Value determined experimentally
                    sc.smart_crop(img, max_border, img_dst_dir, is_test_flag=False)

    process_images()

if __name__ == '__main__':
    main()
