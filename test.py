import cv2 as cv
import smart_crop as sc
import json
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

    def write_to_JSON_file(data):

        with open('segmented-images.json', 'w') as file:
            json.dump(data, file, indent=4)
        file.close()

    def create_JSON_annotation_file(img, dst_dir):
        max_border = 0
        contours_array = sc.get_contours(img)
        data = {}  # Create a dictionary
        for i in range(len(contours_array)):
            contour = contours_array[i]
            min_x, min_y, max_x, max_y, c_x, c_y = sc.get_bounding_box(contour)
            data[i] = {}
            data[i]['label'] = 'TBA'
            data[i]['min_x'] = min_x
            data[i]['min_y'] = min_y
            data[i]['max_x'] = max_x
            data[i]['max_y'] = max_y
            data[i]['c_x'] = c_x
            data[i]['c_y'] = c_y
            # max_border = sc.find_max_border(max_border, min_x, min_y, max_x, max_y, c_x, c_y)
            # print(max_border) -- 186
            # draw_bounding_box(img, min_x, min_y, max_x, max_y)
        write_to_JSON_file(data)
        sc.smart_crop(img, 186, dst_dir)

    dst_dir = r'D:\lego-classification-using-ml\segmented-testing-images'
    src_dir = r'D:\lego-classification-using-ml\testing-images\00.png'
    img = cv.imread(src_dir)

    create_JSON_annotation_file(img, dst_dir)

if __name__ == '__main__':
    main()
