# lego-classification-using-ml
A Lego classification system that uses computer vision (CV) and machine
learning (ML) to perform object detection and object classification to
identify the element ID of each Lego piece in a pile of scattered Lego pieces.

## File Tree

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
├── real-testing-images
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
├── testing-images
│   └── 0.png
└── test.py
