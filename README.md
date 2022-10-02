# lego-classification-using-ml
A Lego classification system that uses computer vision (CV) and machine
learning (ML) to perform object detection and object classification to
identify the element ID of each Lego piece in a pile of scattered Lego pieces.

## File Tree
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

## cad-training-images & real-training-images

These images were used to generate the training dataset. These images have not
undergone any pre-processing.

#### cad-training-images
<kbd>
<img src="https://user-images.githubusercontent.com/105937174/193477813-be073db6-12e7-40e5-8c87-c84c2daeffcf.png" width="250" height="250" />
</kbd>


The cad-training-images folder contains images of Lego pieces that were 
generated in LDCad using the script rotation_script.lua.

#### real-training-images
<kbd>
<img src="https://user-images.githubusercontent.com/105937174/193478246-757e0264-56c4-4767-b12b-d21c1f2454ce.png" width="250" height="250" border="1" />
</kbd>


The real-training-images folder contains images of Lego pieces that were taken
using the Raspberry Pi camera. These images could not be uploaded to this repo
because their file size is too large.
Each one of these images contains two Lego pieces; this was done to speed up 
the process of taking images. The script dataset.py was used to segment these
images into individual Lego pieces. 

## dataset.py

The main purpose of this Python script is to navigate file directories. This script is
used to pre-process all the training images found in the cad-training-images
and real-training-images folders. Pre-processing is performed by calling the
get_segmented_imgs function available in the smart crop library on each
training image. 

## element_ids_for_demonstration.txt

This text file contains the names of all the Lego pieces that are used in the
demonstration.

## neural_net

This folder contains the weights of the machine learning model that was
developped using TensorFlow. The model weights are imported into test.py to
create predictions.

## testing-images

This folder contains the image that is taken by the Raspberry Pi camera during
the demonstration.

## segmented-testing-images

This folder contains segmented images of the testing image; in other words, if
the testing image contains eight Lego pieces, then segmented-testing-images
will contain eight images such that each image only contains one Lego piece. 

## rotation_script.lua

The main purpose of this Lua script is to automate the process of creating 3D 
training images of Lego pieces in LDCad. This script is used to create an 
animation of a Lego piece being rotated 90 degrees on two axes. Each frame of 
this animation is then saved to cad-training-images. 
