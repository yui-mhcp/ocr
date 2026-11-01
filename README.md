# :yum: Optical Character Recognition (OCR)

![Python](https://img.shields.io/badge/python-3.12-blue.svg)
![License](https://img.shields.io/badge/licence-AGPL--3.0-green.svg)
![Version](https://img.shields.io/badge/version-1.1.0-orange.svg)

Check the [CHANGELOG](CHANGELOG.md) file for the latest updates of this project, and the [global CHANGELOG](https://github.com/yui-mhcp/yui-mhcp/blob/main/CHANGELOG.md) for a cross-project overview ! :yum:

## Project structure

```bash
├── architectures            : utilities for model architectures
│   ├── layers                   : custom layer implementations
│   ├── transformers             : transformer architecture implementations
│   ├── crnn_arch.py             : CRNN architecture (text recognition)
│   ├── current_blocks.py        : defines common blocks (e.g., Conv + BN + ReLU)
│   ├── east_arch.py             : EAST architecture (scene-text detection)
│   ├── generation_utils.py      : utilities for text and sequence generation
│   ├── hparams.py               : hyperparameter management
│   ├── simple_models.py         : defines classical models such as CNN / RNN / MLP and siamese
│   └── yolo_arch.py             : YOLOv2 architecture (generic object detection)
├── custom_train_objects     : custom objects used in training / testing
│   ├── callbacks                : training callbacks (checkpoint / history)
│   ├── generators               : custom data generators
│   ├── losses                   : loss functions (incl. `text_loss` for CTC-based OCR)
│   ├── metrics                  : metrics (incl. `text_accuracy`)
│   ├── optimizers               : optimizer loading + custom LR schedulers
│   ├── checkpoint_manager.py    : handle model checkpoint management
│   └── history.py               : main History class to track training statistics / config
├── loggers                  : custom utilities for the `logging` module*
├── models                   : main directory for model classes
│   ├── core                     : the `BaseModel` foundation
│   │   ├── mixins                   : composable features mixed into `BaseModel`
│   │   │   ├── audio_mixin.py           : audio processing functions
│   │   │   ├── checkpoint_mixin.py      : checkpoint / weights management
│   │   │   ├── classification_mixin.py  : classification utilities
│   │   │   ├── image_mixin.py           : image processing functions
│   │   │   ├── processing_mixin.py      : generic input / output processing
│   │   │   ├── text_mixin.py            : text encoding / decoding / processing
│   │   │   └── training_mixin.py        : training loop / `fit` utilities
│   │   ├── utils                    : internal helpers for the model core
│   │   │   ├── describe.py              : model description / summary helpers
│   │   │   ├── migration.py             : legacy config / weights migration
│   │   │   └── saving.py                : model saving / loading helpers
│   │   └── base_model.py            : main `BaseModel` class assembling the mixins
│   ├── detection                : detector implementations
│   │   ├── base_detector.py         : abstract base class for all detectors
│   │   ├── east.py                  : EAST implementation for text detection
│   │   └── yolo.py                  : YOLOv2 implementation for general object detection
│   ├── ocr                      : OCR implementations
│   │   ├── base_ocr.py              : abstract base class for all OCR models
│   │   ├── box_filters.py           : filtering / post-processing of detected text boxes
│   │   ├── crnn.py                  : CRNN implementation for text recognition
│   │   ├── region_tracker.py        : spatial memory of the text regions for streaming de-duplication (experimental)
│   │   └── streaming.py             : OCR over a stream of images
│   └── weights_converter.py     : utilities to convert weights between different models
├── pretrained_models        : pretrained checkpoints (`crnn_en`, `east`, `yolo_coco`)
├── tests                    : `pytest` suite mirroring the `utils` / `loggers` / model tree*
├── utils                    : shared data-processing / keras utilities*
├── ocr.ipynb                : notebook demonstrating model creation + OCR features
├── text.jpg                 : example image used by the notebook
├── CHANGELOG.md             : project-specific changelog
├── CITATION.cff             : citation metadata
├── CONTRIBUTING.md          : contribution guidelines
├── INSTALLATION.md          : shared GPU environment installation guide
├── LICENCE                  : project license file
├── pyproject.toml           : project metadata, dependencies and tooling config
├── README.md                : this file
└── __version__.py           : single-source project version
```

\* The `loggers`, `utils` and most of the `tests` tree are the shared foundation maintained in [the data_processing repository](https://github.com/yui-mhcp/data_processing) — check it for more information on these modules.

Check [the base project](https://github.com/yui-mhcp/base_dl_project) for more information about the `BaseModel` interface, the `architectures` / `custom_train_objects` modules and the general project structure.

Check [the detection project](https://github.com/yui-mhcp/detection) for more information about the `detection` module and the [EAST](https://arxiv.org/abs/1704.03155) Scene-Text Detection model.

## Available features

- **OCR** (module `models.ocr`) :

| Feature   | Function / class   | Description |
| :-------- | :---------------- | :---------- |
| OCR       | `ocr`    | Performs OCR on the given image(s) (detection + recognition) |
| Streaming | `stream` | Runs OCR on a stream of images   |

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

See [the installation guide](INSTALLATION.md) for a step-by-step setup of the shared GPU environment (NVIDIA driver, CUDA, `mamba` and the deep-learning backends) :smile:

Here is a summary of the installation procedure, once your python environment is ready :
1. Clone this repository : `git clone https://github.com/yui-mhcp/ocr.git`
2. Go to the root of this repository : `cd ocr`
3. Install the package with a backend : `pip install -e .[tf]` (or `pip install -e .[torch]`)
4. Open the `ocr` notebook and follow the instructions !

Like `base_dl_project`, this project runs real models, so it needs a **keras backend**. The **image** stack (detection / OCR I/O) and the **text** stack (cleaning / tokenization for the recognition head) are core and already bundled in the base install — you only pick the backend matching your setup :

```bash
pip install -e .[tf]         # keras + tensorflow
pip install -e .[torch]      # keras + torch
pip install -e .[keras]      # alias of [tf]

# optional helpers :
pip install -e .[datasets]   # utils/datasets (tensorflow-datasets, pandas)
pip install -e .[dev]        # test tooling (pytest & plugins)
```

The backend is selected at runtime through the `KERAS_BACKEND` environment variable (e.g., `os.environ['KERAS_BACKEND'] = 'tensorflow'`), as shown at the top of the example notebook.

**Note on pretrained weights** : the pretrained `CRNN` (EasyOCR) and `EAST` (SakuraRiven) checkpoints are stored as `torch` weights and converted to `keras` on the fly, so `torch` is required for the conversion **even if you use the tensorflow backend** (`pip install torch`).

### Quickstart

```python
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

from models.ocr import ocr

# Detect the text boxes (EAST) then recognize them (CRNN), on a single image.
# The pretrained `crnn_en` / `east` weights are downloaded + converted automatically.
infos = ocr(
    'text.jpg', model = 'crnn_en', detector = 'east', display = True
)[0]
print(infos)
```

You can also build the models explicitly from the `BaseModel` interface :

```python
from models.detection import EAST
from models.ocr import CRNN

detector = EAST(nom = 'east', labels = 'text', input_size = (None, None, 3))
model    = CRNN(nom = 'crnn_en', lang = 'english', pretrained_lang = 'english')
print(model)
```

See the `ocr.ipynb` notebook for the full walkthrough (model creation, detection, recognition, line/paragraph combination and streaming). :smile:

### Testing

The tests use [`pytest`](https://docs.pytest.org/) and live in the `tests/` directory, mirroring the `utils` / `loggers` and model tree.

```bash
pip install -e .[dev]        # pytest, pytest-cov, pytest-xdist, pytest-timeout

pytest                       # run the whole suite
pytest -n auto               # run in parallel (pytest-xdist)
pytest -m "not slow"         # skip the slow / heavy tests
pytest --cov                 # run with coverage report
pytest tests/models          # run a single subpackage
```

Tests are annotated with markers (declared in `pyproject.toml`) so the suite adapts to your environment : the `tensorflow`, `torch`, `keras`, `cv2` and `gpu` markers are **auto-skipped** when the corresponding dependency (or hardware) is missing, meaning you can run the tests for the backend you installed without pulling every dependency.

## TO-DO list:

- [x] Make the TO-DO list
- [x] Convert the `CRNN` architecture / weights from the `easyocr` library to `tensorflow`
- [ ] Convert the `CRNN + attention` architecture from [this repo](https://github.com/clovaai/deep-text-recognition-benchmark) to `tensorflow`
- [x] Add examples to initialize pretrained models (both EAST and CRNN)
- [x] Add an example to perform OCR on image (with text detection)
- [ ] Add an example to perform OCR on camera
- [x] Allow to combine texts in lines / paragraphs (as EAST detects individual words)
- [ ] Take into account the text rotation in the combination procedure
- [x] Track the text regions across frames to avoid re-emitting the same texts in streaming (`stream(track_regions = True)`, experimental)

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

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0). See the [LICENCE](LICENCE) file for details.

This license allows you to use, modify, and distribute the code, as long as you include the original copyright and license notice in any copy of the software/source. Additionally, if you modify the code and distribute it, or run it on a server as a service, you must make your modified version available under the same license.

For more information about the AGPL-3.0 license, please visit [the official website](https://www.gnu.org/licenses/agpl-3.0.html)

## Citation

If you find this project useful in your work, please add this citation to give it more visibility! :yum:

```
@misc{yui-mhcp,
    author  = {yui},
    title   = {A Deep Learning projects centralization},
    year    = {2021},
    publisher   = {GitHub},
    howpublished    = {\url{https://github.com/yui-mhcp}}
}
```
