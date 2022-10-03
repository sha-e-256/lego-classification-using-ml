# lego-classification-using-ml

A Lego classification system that uses computer vision and machine learning 
to perform object detection and object classification to identify each Lego 
piece in a pile of scattered Lego pieces.

## training images: cad-training-images
## & real-training-images

The images in the 'cad-training-images' and 'real-training-images' folder are
used to generate the training dataset which is used to train the machine 
learning model. These images have not undergone any pre-processing.

#### cad-training-images
<kbd>
<img src="https://user-images.githubusercontent.com/105937174/193477813-be073db6-12e7-40e5-8c87-c84c2daeffcf.png" width="250" height="250" />
</kbd>

&nbsp;

The 'cad-training-images' folder contains images of 3D Lego pieces that were 
generated in LDCad using the script 'rotation_script.lua'.

#### real-training-images
<kbd>
<img src="https://user-images.githubusercontent.com/105937174/193478246-757e0264-56c4-4767-b12b-d21c1f2454ce.png" width="250" height="250" border="1" />
</kbd>

&nbsp;

The real-training-images folder contains images of real Lego pieces that were 
taken using the Raspberry Pi camera. These images could not be uploaded to 
this repo because their file size is too large.

Each one of these images contains two Lego pieces; this was done to speed up 
the process of taking images. The script 'dataset.py' is used to segment these
images into images of individual Lego pieces. 

## scripts: dataset.py & rotation.lua

### dataset.py

The main purpose of the Python script 'dataset.py' is to navigate file 
directories. This script is used to pre-process all the training images found
in the cad-training-images and real-training-images folders. 

Pre-processing is performed by calling the get_segmented_imgs function 
available in the smart crop library on each training image. 

### rotation_script.lua

The main purpose of the Lua script 'rotation_script.lua' is to automate the 
process of creating 3D training images of Lego pieces in LDCad. This script is
used to create an animation of a Lego piece being rotated 90 degrees on two 
axes. Each frame of this animation is then saved to the 'cad-training-images'
folder. 

## libraries

### smart_crop.py

The main purpose of the library 'smart_crop.py' is to provide functions that
can be used to pre-process images. Pre-processing an image involves segmenting
the image such that each image contains only one Lego piece, removing the 
background of the image, centering the Lego piece in the image, rotating the 
Lego piece in the image such that it is perpendicular to the borders of the 
images, and scaling the Lego piece in the image with respect to the largest 
available Lego piece.

## square training images: square-cad-training-images & square-real-training-images

The 'square-cad-training-images' and 'square-real-training-images' folders contain 
the training images after undergoing pre-processing.

Pre-processing is performed using dataset.py and the functions available in the
smart crop library.

### square-cad-training-images

The 'square-cad-training-images' folder contains the cad-training-images after
undergoing pre-processing.

<kbd>
<img src="https://user-images.githubusercontent.com/105937174/193480931-ecb74b3b-9c4d-4859-81ec-01c9895b6fca.png" width="250" height="250" />
</kbd>


### square-real-training-images

The 'square-real-training-images' folder contains the real-training-images 
after undergoing pre-processing.

As mentioned earlier, each real training image contains two Lego pieces;
therefore, after pre-processing, that image is segmented such that each image 
contains only one Lego piece. As a result, two images are generated.

<kbd>
<img src="https://user-images.githubusercontent.com/105937174/193481027-16bb8b67-b16b-4311-be06-5ac05d4301e9.png" width="250" height="250" />
</kbd>

<kbd>
<img src="https://user-images.githubusercontent.com/105937174/193481095-0bd7338f-1858-471f-a7e9-da10c1221921.png" width="250" height="250" />
</kbd>

## testing images: testing-images & segmented-testing-images

### testing-images

The 'testing-images' folder contains the image that is taken by the Raspberry Pi camera during
the demonstration.

### segmented-testing-images

The 'segmented-testing-images' folder contains the segmented images of the 
testing image; in other words, if the testing image in 'testing-images' 
contains eight Lego pieces, then 'segmented-testing-images' will contain eight
images with each image containing only one Lego piece. 

## demonstration

### test.py

The main purpose of the Python module 'test.py' is to perform the
demonstration. This module displays a livefeed of the Raspberry Pi camera, and
then takes an image after receiving input from the user. This image is then 
segmented such that each segmented image contains only one Lego piece. The 
machine learning model then takes each segmented image as an input and 
generates a prediction of what piece is in the segmented image. This prediction
is then displayed as a percentage on the original image that was taken.

<kbd>
<img src="https://user-images.githubusercontent.com/105937174/193481490-6adf8b75-83f6-42f2-bf3f-d802b533c928.png" width="250" height="250" />
</kbd>


<kbd>
<img src="https://user-images.githubusercontent.com/105937174/193481536-7cddeffc-955d-425c-a6b5-621dd565a75e.png" width="250" height="250" />
</kbd>


### element_ids_for_demonstration.txt

The 'element_ids_for_demonstration.txt' file contains the names of all the 
Lego pieces that are used in the demonstration.

### neural_net

This 'neural_net' folder contains the weights of the machine learning model
that is used to create predictions. Predictions are generated in the 'test.py'
module.

## file tree
```
lego-classification-using-ml
├── cad-training-images
│   ├── 2639
│   │   ├── 2639-01.png
│   │   ├── 2639-03.png
│   │   ├── 2639-05.png
│   │   ├── ...
│   │   └── 2639-63.png
│   ├── 3009
│   │   ├── 3009-00.png
│   │   ├── 3009-01.png
│   │   ├── 3009-02.png
│   │   ├── ...
│   │   └── 3009-63.png
│   ├── 3021
│   │   ├── 3021-01.png
│   │   ├── 3021-03.png
│   │   ├── 3021-05.png
│   │   ├── ...
│   │   └── 3021-63.png
│   ├── 3034
│   │   ├── 3034-01.png
│   │   ├── 3034-03.png
│   │   ├── 3034-05.png
│   │   ├── ...
│   │   └── 3034-63.png
│   ├── 3308
│   │   ├── 3308-00.png
│   │   ├── 3308-01.png
│   │   ├── 3308-02.png
│   │   ├── ...
│   │   └── 3308-63.png
│   ├── 3666
│   │   ├── 3666-01.png
│   │   ├── 3666-03.png
│   │   ├── 3666-05.png
│   │   ├── ...
│   │   └── 3666-63.png
│   ├── 3710
│   │   ├── 3710-01.png
│   │   ├── 3710-03.png
│   │   ├── 3710-05.png
│   │   ├── ...
│   │   └── 3710-63.png
│   └── 41769
│       ├── 41769-01.png
│       ├── 41769-03.png
│       ├── 41769-05.png
│       ├── ...
│       └── 41769-63.png
├── dataset.py
├── element_ids_for_demonstration.txt
├── neural_net
│   ├── keras_metadata.pb
│   ├── saved_model.pb
│   └── variables
│       ├── variables.data-00000-of-00001
│       └── variables.index
├── README.md
├── testing-images
│   └── 0.png
├── rotation_script.lua
├── segmented-testing-images
│   ├── 0.png
│   ├── 1.png
│   ├── 2.png
│   ├── 3.png
│   ├── 4.png
│   ├── 5.png
│   ├── 6.png
│   └── 7.png
├── smart_crop.py
├── square-cad-training-images
│   ├── 2639
│   │   ├── 2639-01.png
│   │   ├── 2639-03.png
│   │   ├── 2639-05.png
│   │   ├── ...
│   │   └── 2639-63.png
│   ├── 3009
│   │   ├── 3009-00.png
│   │   ├── 3009-01.png
│   │   ├── 3009-02.png
│   │   ├── ...
│   │   └── 3009-63.png
│   ├── 3021
│   │   ├── 3021-01.png
│   │   ├── 3021-03.png
│   │   ├── 3021-05.png
│   │   ├── ...
│   │   └── 3021-63.png
│   ├── 3034
│   │   ├── 3034-01.png
│   │   ├── 3034-03.png
│   │   ├── 3034-05.png
│   │   ├── ...
│   │   └── 3034-63.png
│   ├── 3308
│   │   ├── 3308-00.png
│   │   ├── 3308-01.png
│   │   ├── 3308-02.png
│   │   ├── ...
│   │   └── 3308-63.png
│   ├── 3666
│   │   ├── 3666-01.png
│   │   ├── 3666-03.png
│   │   ├── 3666-05.png
│   │   ├── ...
│   │   └── 3666-63.png
│   ├── 3710
│   │   ├── 3710-01.png
│   │   ├── 3710-03.png
│   │   ├── 3710-05.png
│   │   ├── ...
│   │   └── 3710-63.png
│   └── 41769
│       ├── 41769-01.png
│       ├── 41769-03.png
│       ├── 41769-05.png
│       ├── ...
│       └── 41769-63.png
├── square-real-training-images
│   ├── 2639
│   │   ├── photo1-0.png
│   │   ├── photo2-0.png
│   │   ├── photo3-0.png
│   │   ├── ...
│   │   └── photo150-0.png
│   ├── 3009
│   │   ├── photo1-0.png
│   │   ├── photo2-0.png
│   │   ├── photo3-0.png
│   │   ├── ...
│   │   └── photo150-1.png
│   ├── 3021
│   │   ├── photo1-1.png
│   │   ├── photo2-1.png
│   │   ├── photo3-1.png
│   │   ├── ...
│   │   └── photo150-0.png
│   ├── 3034
│   │   ├── photo1-1.png
│   │   ├── photo2-1.png
│   │   ├── photo3-1.png
│   │   ├── ...
│   │   └── photo150-1.png
│   ├── 3308
│   │   ├── photo1-1.png
│   │   ├── photo2-0.png
│   │   ├── photo3-0.png
│   │   ├── ...
│   │   └─── photo150-1.png
│   ├── 3666
│   │   ├── photo1-1.png
│   │   ├── photo2-1.png
│   │   ├── photo3-1.png
│   │   ├── ...
│   │   └── photo150-1.png
│   ├── 3710
│   │   ├── photo1-0.png
│   │   ├── photo2-0.png
│   │   ├── photo3-0.png
│   │   ├── ...
│   │   └── photo150-0.png
│   └── 41769
│       ├── photo1-0.png
│       ├── photo2-1.png
│       ├── photo3-1.png
│       ├── ....
│       └── photo150-0.png
└── test.py
```

## important notes

The code used to generate the machine learning model is available in a
Google Colab notebook. 
