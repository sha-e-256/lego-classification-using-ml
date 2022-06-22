import cv2
import smart_crop as sc
import json
import os

def main():

    def draw_bounding_box(img, x_min, y_min, x_max, y_max):
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max ), (0, 255, 0), 1)
        cv2.imshow("Image w/ Bounding Box", img)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def write_to_JSON_file(dst_dir, file_name, data, is_tst_img):
        try:
            print(f'hello: {dst_dir}')
            os.mkdir(dst_dir)
        except OSError as error:
            print(f'Directory with path {dst_dir} already exists.')

        with open(f'{dst_dir}\{file_name}.json', 'w') as file:
            json.dump(data, file, indent=4)
        file.close()

    def create_JSON_annotation_file(img, src_dir, dst_dir, is_tst_img):
        """Creates a JSON file that lists annotation details of an image such as the image label, the
        dimensions of the bounding box (x_min, y_min, x_max, y_max), and the coordinates of the centroid (c_x, y_x).


                Parameters
                ----------
                src_dir : str
                    The directory of the image that will be used to generate annotations.
                dst_dir: str
                    The directory where the annotations will be saved.
                is_tst_img: bool
                    If the image being used to generate the annotation is a testing image,
                    than the JSON file will also contain the accuracy of each prediction.

                Raises
                ------
                FileNotFoundError
                    If the image directory cannot be found.
                """
        contours_array = sc.get_contours(img)
        data = {}  # Create a dictionary

        for i in range(len(contours_array)):
            x_min, y_min, x_max, y_max, c_x, c_y = sc.get_bounding_box(contours_array, i)
            data[i] = {}
            data[i]['label'] = os.path.basename(src_dir).split('-')[0] if (is_tst_img == False) else 'TBA'
            data[i]['x_min'] = x_min
            data[i]['y_min'] = y_min
            data[i]['x_max'] = x_max
            data[i]['y_max'] = y_max
            data[i]['c_x'] = c_x
            data[i]['c_y'] = c_y
            draw_bounding_box(img, x_min, y_min, x_max, y_max)

        if not is_tst_img:
            write_to_JSON_file(f"{dst_dir}\{str(data[0]['label'])}", os.path.basename(src_dir).split('.')[0], data, is_tst_img)
        else:
            write_to_JSON_file(f"{dst_dir}", os.path.basename(src_dir).split('.')[0], data, is_tst_img)

    #dst_dir = r'D:\lego-classification-using-ml\training-images-annotations'
    dst_dir = r'D:\lego-classification-using-ml\testing-images-annotations'

    try:
        os.mkdir(dst_dir)
    except OSError as error:
        print(f'Directory with path {dst_dir} already exists.')
    src_dir = r'D:\lego-classification-using-ml\testing-images\00.jpg'  # is_tst_img = True
    #src_dir = r'D:\lego-classification-using-ml\training-images\3308\3308-000.png'  # is_tst)img = False
    img = cv2.imread(src_dir)  # r: treat directory as a raw string
    create_JSON_annotation_file(img, src_dir, dst_dir, False)


if __name__ == '__main__':
    main()
