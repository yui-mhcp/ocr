# :yum: Optical Character Recognition (OCR)

Check the [CHANGELOG](https://github.com/yui-mhcp/yui-mhcp/blob/main/CHANGELOG.md) file to have a global overview of the latest modifications! :yum:

## Project structure

```bash
├── architectures            : utilities for model architectures
│   ├── layers               : custom layer implementations
│   ├── transformers         : transformer architecture implementations
│   ├── common_blocks.py     : defines common blocks (e.g., Conv + BN + ReLU)
│   ├── crnn_arch.py         : CRNN architecture
│   ├── east_arch.py         : EAST architecture
│   ├── generation_utils.py  : utilities for text and sequence generation
│   ├── hparams.py           : hyperparameter management
│   ├── simple_models.py     : defines classical models such as CNN / RNN / MLP and siamese
│   └── yolo_arch.py         : YOLOv2 architecture
├── custom_train_objects     : custom objects used in training / testing
├── loggers                  : logging utilities for tracking experiment progress
├── models                   : main directory for model classes
│   ├── detection            : detector implementations
│   │   ├── base_detector.py : abstract base class for all detectors
│   │   ├── east.py          : EAST implementation for text detection
│   │   └── yolo.py          : YOLOv2 implementation for general object detection
│   ├── interfaces           : directories for interface classes
│   ├── ocr                  : OCR implementations
│   │   ├── base_ocr.py      : abstract base class for all OCR models
│   │   └── crnn.py          : CRNN implementation for OCR
│   └── weights_converter.py : utilities to convert weights between different models
├── tests                    : unit and integration tests for model validation
├── utils                    : utility functions for data processing and visualization
├── LICENCE                  : project license file
├── ocr.ipynb                : notebook demonstrating model creation + OCR features
├── README.md                : this file
└── requirements.txt         : required packages
```

Check [the main project](https://github.com/yui-mhcp/base_dl_project) for more information about the unextended modules / structure / main classes. 

Check [the detection project](https://github.com/yui-mhcp/detection) for more information about the `detection` module and the [EAST](https://arxiv.org/abs/1704.03155) Scene-Text Detection model. 

## Available features

- **OCR** (module `models.ocr`) :

| Feature   | Function / class   | Description |
| :-------- | :---------------- | :---------- |
| OCR       | `ocr`  | Performs OCR on the given image(s)   |

You can check the `ocr` notebook for a concrete demonstration.

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

Models should be unzipped in the `pretrained_models/` directory!

The pretrained `CRNN` models come from the [EasyOCR](https://github.com/JaidedAI/EasyOCR) library. Weights are automatically downloaded given the language or the model name, and converted to `keras`! The `easyocr` library is therefore not required, while `pytorch` is required for weights loading (for conversion).

The pretrained version of EAST can be downloaded [from this project](https://github.com/SakuraRiven/EAST). It should be placed in `pretrained_models/pretrained_weights/east_vgg16.pth` (`torch` is required to convert the weights: `pip install torch`).

## Installation and usage

See [the installation guide](https://github.com/yui-mhcp/blob/master/INSTALLATION.md) for a step-by-step installation :smile:

Here is a summary of the installation procedure, if you have a working python environment :
1. Clone this repository: `git clone https://github.com/yui-mhcp/ocr.git`
2. Go to the root of this repository: `cd ocr`
3. Install requirements: `pip install -r requirements.txt`
4. Open the `ocr` notebook and follow the instructions!

## TO-DO list:

- [x] Make the TO-DO list
- [x] Convert the `CRNN` architecture / weights from the `easyocr` library to `tensorflow`
- [ ] Convert the `CRNN + attention` architecture from [this repo](https://github.com/clovaai/deep-text-recognition-benchmark) to `tensorflow`
- [x] Add examples to initialize pretrained models (both EAST and CRNN)
- [x] Add an example to perform OCR on image (with text detection)
- [ ] Add an example to perform OCR on camera
- [x] Allow to combine texts in lines / paragraphs (as EAST detects individual words)
- [ ] Take into account the text rotation in the combination procedure

## Notes and references 

### GitHub projects

The code for the CRNN architecture is highly inspired from the `easyocr` repo:
- [EasyOCR library](https://github.com/JaidedAI/EasyOCR): official repo of the `easyocr` library

The code for the EAST part of this project is highly inspired from this repo:
- [SakuraRiven pytorch implementation](https://github.com/SakuraRiven/EAST): pytorch implementation of the EAST paper.

- [Awesome-OCR](https://github.com/kba/awesome-ocr) : A curated list of OCR resources
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) : The official Tesseract repository
- [Deep Text Recognition Benchmark](https://github.com/clovaai/deep-text-recognition-benchmark) : A comprehensive benchmark of Scene Text Recognition models
- [CRAFT-pytorch](https://github.com/clovaai/CRAFT-pytorch) : Character Region Awareness for Text Detection
- [mmocr](https://github.com/open-mmlab/mmocr) : OpenMMLab Text Detection, Recognition and Understanding Toolbox

### Papers

- [An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition](https://arxiv.org/abs/1507.05717) : the original CRNN paper
- [What Is Wrong With Scene Text Recognition Model Comparisons? Dataset and Model Analysis](https://arxiv.org/abs/1904.01906) : a great benchmark of OCR models + an open-source repository with pretrained models and datasets
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) : U-net original paper
- [EAST: An Efficient and Accurate Scene Text Detector](https://arxiv.org/abs/1704.03155) : text detection (with possibly rotated bounding-boxes) with a segmentation model (U-Net).

### Datasets

- [COCO Text](https://vision.cornell.edu/se3/coco-text-2/): an extension of COCO for text detection
- [ICDAR 2015](https://rrc.cvc.uab.es/?ch=4): a standard dataset for text detection and recognition
- [Synthetic Word Dataset](https://www.robots.ox.ac.uk/~vgg/data/text/): synthetic word dataset for OCR training

## Tutorials

- [A Comprehensive Guide to OCR with Tesseract, OpenCV and Python](https://nanonets.com/blog/ocr-with-tesseract/) : A great introduction to classical OCR approaches
- [Scene Text Detection with OpenCV](https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/) : Tutorial on implementing EAST text detector
- [Attention Mechanisms in OCR](https://towardsdatascience.com/attention-in-neural-networks-e66920838742) : How attention mechanisms improve OCR accuracy


## Contacts and licence

Contacts:
- **Mail**: `yui-mhcp@tutanota.com`
- **[Discord](https://discord.com)**: yui0732

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0). See the [LICENSE](LICENSE) file for details.

This license allows you to use, modify, and distribute the code, as long as you include the original copyright and license notice in any copy of the software/source. Additionally, if you modify the code and distribute it, or run it on a server as a service, you must make your modified version available under the same license.

For more information about the AGPL-3.0 license, please visit [the official website](https://www.gnu.org/licenses/agpl-3.0.html)

## Citation

If you find this project useful in your work, please add this citation to give it more visibility! :yum:

```
@misc{yui-mhcp
    author  = {yui},
    title   = {A Deep Learning projects centralization},
    year    = {2021},
    publisher   = {GitHub},
    howpublished    = {\url{https://github.com/yui-mhcp}}
}
```