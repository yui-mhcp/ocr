# :yum: Optical Character Recognition

**NEW** : this repository is new and experimental, do not hesitate to open issues if you have any question or bug, or even suggestions to improve the project ! :yum:

Check the [CHANGELOG](https://github.com/yui-mhcp/yui-mhcp/blob/main/CHANGELOG.md) file to have a global overview of the latest modifications ! :yum:

## Project structure

```bash
├── custom_architectures
│   ├── crnn_arch.py        : defines the CRNN main architecture for OCR (with CTC decoding)
│   ├── unet_arch.py        : defines variants of the UNet architectures used in the EAST detector
│   └── yolo_arch.py        : defines the YOLOv2 architecture
├── custom_layers
├── custom_train_objects
├── datasets
├── hparams
├── loggers
├── models
│   ├── detection           : used to detect texts in images (with the EAST detector)
│   ├── ocr
│   │   ├── base_ocr.py     : abstract class for OCR models
│   │   └── crnn.py         : main CRNN class (OCR)
├── pretrained_models
│   └── yolo_backend        : directory where to save the yolo_backend weights
├── unitest
├── utils
├── example_crnn.ipynb
└── pcr.ipynb
```

Check [the main project](https://github.com/yui-mhcp/base_dl_project) for more information about the unextended modules / structure / main classes. 

Check [the detection project](https://github.com/yui-mhcp/detection) for more information about the `detection` module and the [EAST](https://arxiv.org/abs/1704.03155) Scene-Text Detection model. 

## Available features

- **Detection** (module `models.detection`) :

| Feature   | Fuction / class   | Description |
| :-------- | :---------------- | :---------- |
| OCR       | `ocr`  | Performs OCR on the given image(s)   |

You can check the `ocr` notebook for a concrete demonstration

## Available models

### Model architectures

Available architectures : 
- `detection` :
    - [EAST](https://arxiv.org/abs/1704.03155)
- `OCR` :
    - [CRNN](https://arxiv.org/abs/1507.05717)

### Model weights

| Classes   | Dataset   | Architecture  | Trainer   | Weights   |
| :-------: | :-------: | :-----------: | :-------: | :-------: |

Models must be unzipped in the `pretrained_models/` directory !

The pretrained `CRNN` models come from the [EasyOCR](https://github.com/JaidedAI/EasyOCR) library. Weights are automatically downloaded given the language or the model's name, and converted in `tensorflow` ! The `easyocr` is therefore not required, by `pytorch` is required for weights loading (for convertion).

The pretrained version of EAST can be downloaded [from this project](https://github.com/SakuraRiven/EAST). It should be set in `pretrained_models/pretrained_weights/east_vgg16.pth` (`torch` is required to transfer the weights : `pip install torch`).

## Installation and usage

1. Clone this repository : `git clone https://github.com/yui-mhcp/ocr.git`
2. Go to the root of this repository : `cd ocr`
3. Install requirements : `pip install -r requirements.txt`
4. Open `detection` notebook and follow the instructions !

**Important Note** : some *heavy* requirements are removed in order to avoid unnecessary installation of such packages (e.g. `torch` and `transformers`), as they are used only in very specific functions.  It is therefore possible that some `ImportError` occurs when using specific functions, such as `TextEncoder.from_transformers_pretrained(...)`. 

## TO-DO list :

- [x] Make the TO-DO list
- [x] Convert the `CRNN` architecture / weights from the `easyocr` library to `tensorflow`
- [ ] Convert the `CRNN + attention` architecture from [this repo](https://github.com/clovaai/deep-text-recognition-benchmark) to `tensorflow`
- [x] Add examples to initialize pretrained models (both EAST and CRNN)
- [x] Add an example to perform OCR on image (with text detection)
- [ ] Add an example to perform OCR on camera
- [x] Allow to combine texts in lines / paragraphs (as EAST detects individual words)
- [ ] Take into account the text rotation in the combination procedure

## Contacts and licence

You can contact [me](https://github.com/yui-mhcp) at yui-mhcp@tutanota.com or on [discord](https://discord.com) at `yui#0732`

The objective of these projects is to facilitate the development and deployment of useful application using Deep Learning for solving real-world problems and helping people. 
For this purpose, all the code is under the [Affero GPL (AGPL) v3 licence](LICENCE)

All my projects are "free software", meaning that you can use, modify, deploy and distribute them on a free basis, in compliance with the Licence. They are not in the public domain and are copyrighted, there exist some conditions on the distribution but their objective is to make sure that everyone is able to use and share any modified version of these projects. 

Furthermore, if you want to use any project in a closed-source project, or in a commercial project, you will need to obtain another Licence. Please contact me for more information. 

For my protection, it is important to note that all projects are available on an "As Is" basis, without any warranties or conditions of any kind, either explicit or implied. However, do not hesitate to report issues on the repository's project or make a Pull Request to solve it :smile: 

If you use this project in your work, please add this citation to give it more visibility ! :yum:

```
@misc{yui-mhcp
    author  = {yui},
    title   = {A Deep Learning projects centralization},
    year    = {2021},
    publisher   = {GitHub},
    howpublished    = {\url{https://github.com/yui-mhcp}}
}
```

## Notes and references

The code for the CRNN architecture is highly inspired from the `easyocr` repo :
- [1] [EasyOCR library](https://github.com/JaidedAI/EasyOCR) : official repo of the `easyocr` library
The code for the EAST part of this project is highly inspired from this repo :
- [2] [SakuraRiven pytorch implementation](https://github.com/SakuraRiven/EAST) : pytorch implementation of the EAST paper.


Papers and tutorials :
- [1] [An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition](https://arxiv.org/abs/1507.05717) : the original CRNN paper
- [2] [What Is Wrong With Scene Text Recognition Model Comparisons? Dataset and Model Analysis](https://arxiv.org/abs/1904.01906) : a great benchmark of OCR model + an open-source repository with pretrained models and datasets
- [3] [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) : U-net original paper
- [4] [EAST: An Efficient and Accurate Scene Text Detector](https://arxiv.org/abs/1704.03155) : text detection (with possibly rotated bounding-boxes) with a segmentation model (U-Net). 


Datasets :
- [COCO Text](https://vision.cornell.edu/se3/coco-text-2/) dataset : an extension of COCO for text detection
