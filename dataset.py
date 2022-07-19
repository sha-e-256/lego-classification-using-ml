import os
import argparse
import random
import cv2 as cv  # openCV
import numpy as np
from tqdm import tqdm
import smart_crop as sc  # Self-made library used to crop images

# The objective of this program is to pre-process images that will be used to
# generate a training dataset. To begin, the background of each image is
# removed, then the piece is centered in the image, and then the image is
# cropped to have a square (1:1) aspect ratio.

def main():

    # Allow user to supply the paths of directories using the command line
    # For ex., enter the following into the terminal:
    # python dataset.py 'D:\lego-classification-using-ml\training-images'
    # 'D:\lego-classification-using-ml\square-training-images'
    # Use 'CTRL-C' to exit a script running in terminal
    def get_src_dst_dir():
        parser = argparse.ArgumentParser(
            description='''Pre-processes images that will be used to generate
            a training dataset''')

        parser.add_argument(
            'src_dir', metavar='src_dir', type=str,
            help='''Enter top-most source directory containing images which
            will be pre-processed''')
        parser.add_argument(
            'dst_dir', metavar='dst_dir', type=str,
            help='''Enter top-most destination directory where pre-processed
            images will be saved''')

        args = parser.parse_args()
        src_dir = args.src_dir
        dst_dir = args.dst_dir

        return src_dir, dst_dir

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
        src_dir, dst_dir = get_src_dst_dir()  # Paths of source and destination
        # directories, respectively
        class_names = get_class_names(src_dir)
        # print(f'\nsrc_dir: {src_dir} \ndst_dir: {dst_dir}')  # Debug statement
        # print(f'\nclass names:{class_names}')  # Debug statement
        for class_name in class_names:
            class_name_dir = rf'{src_dir}\{class_name}'  # Path of each
            # subdirectory
            # r' ignores escape characters in string
            # For ex., \t is an escape character which creates an indent
            img_names = os.listdir(class_name_dir)  # Names of images in
            # subdirectory
            for img_name in tqdm(img_names):
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
                    sc.rotate_and_square_crop(img, img_dst_dir)

    process_images()

if __name__ == '__main__':
    main()
