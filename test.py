import cv2 as cv
import smart_crop as sc
import json
import numpy as np
import os

__author__ = "Shahed E."

# The objective of this program is to take an image of a pile of scattered pieces
# and segment it into several images
# The coordinates of the bounding box and the centroid of each Lego piece
# will be saved in a JSON file

def main():

    def draw_bounding_box(img, min_x, min_y, max_x, max_y):
        cv.rectangle(img, (min_x, min_y), (max_x, max_y ), (0, 255, 0), 1)
        cv.imshow("Image w/ Bounding Box", img)
        cv.waitKey()
        cv.destroyAllWindows()

    def get_upright_bounding_box():
        pass

    def write_to_JSON_file(data):

        with open('segmented-images.json', 'w') as file:
            json.dump(data, file, indent=4)
        file.close()

    def create_JSON_annotation_file(img, dst_dir):
        max_border = 0
        contours_array = sc.get_contours(img)

        img_g = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # Convert to greyscale
        threshold = cv.threshold(img_g, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[0]
        mask = cv.inRange(img_g, threshold, 255)  # Create a mask of all pixels where the
                                                  # pixel value is from threshold to 255 (white)
        img_copy = img.copy()
        img_copy[mask > threshold] = (255, 255, 255)
        data = {}  # Create a dictionary
        for i in range(len(contours_array)):
            contour = contours_array[i]

            # If angle changes, center coordinates and width/height do not change
            b_rect = cv.minAreaRect(contour)  # The center coordinates, width, height, and angle of rotation of b box
            rotation_matrix = cv.getRotationMatrix2D(center=b_rect[0], angle=b_rect[2], scale=1)
            rotated_img = cv.warpAffine(src=img_copy, M=rotation_matrix, dsize=(1280, 720))

            b_box = np.int0(cv.boxPoints(b_rect))  # Un-rotated corner coordinates
            min_x = b_box[0, 0] - 2
            max_x = b_box[2, 0] + 4
            min_y = b_box[1, 1] - 2
            max_y = b_box[3, 1] + 4

            # Tuples are immutable
            (c_x, c_y) = b_rect[0] # Used for rotated and un-rotated images

            (width, height) = b_rect[1]
            rotated_b_rect = (b_rect[0], b_rect[1], 0)  # Bounding box of piece after rotating image
            rotated_b_box = np.int0(cv.boxPoints(rotated_b_rect))  # The coordinates of the four corners of the bounding box
            # print(b_box)
            rotated_min_x = rotated_b_box[0, 0] - 2
            rotated_max_x = rotated_b_box[2, 0] + 4
            rotated_min_y = rotated_b_box[1, 1] - 2
            rotated_max_y = rotated_b_box[3, 1] + 4

            data[i] = {}
            data[i]['label'] = 'TBA'
            data[i]['min_x'] = int(min_x)
            data[i]['min_y'] = int(min_y)
            data[i]['max_x'] = int(max_x)
            data[i]['max_y'] = int(max_y)
            data[i]['c_x'] = int(c_x)
            data[i]['c_y'] = int(c_y)

            max_border = sc.find_max_border(max_border, min_x, min_y, max_x, max_y, c_x, c_y)
            cropped_img = rotated_img[rotated_min_y:rotated_max_y, rotated_min_x:rotated_max_x]
            #cv.drawContours(rotated_img, [b_box], 0, (0, 0, 255), 1)
            # cv.imshow("img with no bg", cropped_img)
            # cv.waitKey()
            # cv.destroyAllWindows()
            # draw_bounding_box(img, min_x, min_y, max_x, max_y)

        # cv.imshow("img with no bg", img_copy)
        # cv.waitKey()
        # cv.destroyAllWindows()
        write_to_JSON_file(data)
        sc.smart_crop(img_copy, max_border, dst_dir)

    dst_dir = r'E:\lego-classification-using-ml\segmented-testing-images'
    # On laptop E drive; on PC D drive
    src_dir = r'E:\lego-classification-using-ml\testing-images\00.png'
    img = cv.imread(src_dir)

    create_JSON_annotation_file(img, dst_dir)

if __name__ == '__main__':
    main()
