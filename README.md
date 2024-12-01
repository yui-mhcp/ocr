# :yum: Optical Character Recognition (OCR)

Check the [CHANGELOG](https://github.com/yui-mhcp/yui-mhcp/blob/main/CHANGELOG.md) file to have a global overview of the latest modifications ! :yum:

## Project structure

```bash
├── custom_architectures
│   ├── crnn_arch.py        : defines the CRNN main architecture for OCR (with CTC decoding)
│   ├── generation_utils.py : inference methods for CRNN with attention model *
│   ├── east_arch.py        : defines EAST text detector architecture
│   └── yolo_arch.py        : defines the YOLOv2 architecture
├── custom_layers
├── custom_train_objects
├── loggers
├── models
│   ├── detection           : used to detect texts in images (with the EAST detector)
│   ├── ocr
│   │   ├── base_ocr.py     : abstract class for OCR models
│   │   └── crnn.py         : main CRNN class (OCR)
├── pretrained_models
│   └── yolo_backend        : directory where to save the yolo_backend weights
├── unitests
├── utils
├── example_crnn.ipynb
└── pcr.ipynb
```

\* This architecture is still experimental. Pretrained models / examples will be provided in the next update

Check [the main project](https://github.com/yui-mhcp/base_dl_project) for more information about the unextended modules / structure / main classes. 

Check [the detection project](https://github.com/yui-mhcp/detection) for more information about the `detection` module and the [EAST](https://arxiv.org/abs/1704.03155) Scene-Text Detection model. 

## Available features

- **OCR** (module `models.ocr`) :

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

The pretrained `CRNN` models come from the [EasyOCR](https://github.com/JaidedAI/EasyOCR) library. Weights are automatically downloaded given the language or the model name, and converted in `keras` ! The `easyocr` is therefore not required, while `pytorch` is required for weights loading (for convertion).

The pretrained version of EAST can be downloaded [from this project](https://github.com/SakuraRiven/EAST). It should be placed in `pretrained_models/pretrained_weights/east_vgg16.pth` (`torch` is required to convert the weights : `pip install torch`).

## Installation and usage

Check [this installagion guide](https://github.com/yui-mhcp/yui-mhcp/blob/main/INSTALLATION.md) for the step-by-step instructions !

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

Contacts :
- **Mail** : `yui-mhcp@tutanota.com`
- **[Discord](https://discord.com)** : yui0732

### Terms of use

The goal of these projects is to support and advance education and research in Deep Learning technology. To facilitate this, all associated code is made available under the [GNU Affero General Public License (AGPL) v3](AGPLv3.licence), supplemented by a clause that prohibits commercial use (cf the [LICENCE](LICENCE) file).

These projects are released as "free software", allowing you to freely use, modify, deploy, and share the software, provided you adhere to the terms of the license. While the software is freely available, it is not public domain and retains copyright protection. The license conditions are designed to ensure that every user can utilize and modify any version of the code for their own educational and research projects.

If you wish to use this project in a proprietary commercial endeavor, you must obtain a separate license. For further details on this process, please contact me directly.

For my protection, it is important to note that all projects are available on an "As Is" basis, without any warranties or conditions of any kind, either explicit or implied. However, do not hesitate to report issues on the repository's project, or make a Pull Request to solve it :smile: 

### Citation

If you find this project useful in your work, please add this citation to give it more visibility ! :yum:

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
- [EasyOCR library](https://github.com/JaidedAI/EasyOCR) : official repo of the `easyocr` library
The code for the EAST part of this project is highly inspired from this repo :
- [SakuraRiven pytorch implementation](https://github.com/SakuraRiven/EAST) : pytorch implementation of the EAST paper.


Papers and tutorials :
- [An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition](https://arxiv.org/abs/1507.05717) : the original CRNN paper
- [What Is Wrong With Scene Text Recognition Model Comparisons? Dataset and Model Analysis](https://arxiv.org/abs/1904.01906) : a great benchmark of OCR model + an open-source repository with pretrained models and datasets
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) : U-net original paper
- [EAST: An Efficient and Accurate Scene Text Detector](https://arxiv.org/abs/1704.03155) : text detection (with possibly rotated bounding-boxes) with a segmentation model (U-Net). 


Datasets :
- [COCO Text](https://vision.cornell.edu/se3/coco-text-2/) dataset : an extension of COCO for text detection
