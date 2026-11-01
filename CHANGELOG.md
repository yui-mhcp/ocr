# :yum: CHANGELOG :yum:

This file tracks the major updates of the [ocr](https://github.com/yui-mhcp/ocr) project. For the global, cross-project overview of all the `yui-mhcp` repositories, see the [central CHANGELOG](https://github.com/yui-mhcp/yui-mhcp/blob/main/CHANGELOG.md) :smile:

## [v1.1.0] - 01/11/2026 - November update ! :yum:

Minor release, mainly synced with the shared foundation, with a new (experimental) region tracking for the streaming OCR !

### Major updates

- The shared `utils` / `loggers` / `models/core` foundation has been synced (see the [data_processing CHANGELOG](https://github.com/yui-mhcp/data_processing/blob/main/CHANGELOG.md) and the [base_dl_project CHANGELOG](https://github.com/yui-mhcp/base_dl_project/blob/main/CHANGELOG.md)) : new `image_augmentation` module, `Tokenizer` support of `transformers 5`, new `TensorRTLLMAPIRuntime` (TensorRT-LLM `LLM API`), new append-only `JSONLDatabase`, `keras 3.16` compatibility and an extended test suite

### Minor updates

- **[EXPERIMENTAL]** New `RegionTracker` (`models/ocr/region_tracker.py`) : a persistent spatial memory associating the text *slots* (positions) to their last OCR-ised *content*, enabled by `stream(track_regions = True)` (configured by `tracker_config`). It replaces the `RepetitionFilter`, the "already emitted" IoA filter and the emitted-texts sets, and only re-OCR-ises the regions whose content changed
- The legacy streaming pipeline now applies the "already emitted" IoA filter first, keeps the edit-distance history across (possibly false) scene switches, and its default history size `k` has been increased (5 -> 15)
- The line combination relies on the new `combine_line_continuations` / `sanitize_line_rows` methods
- `keras 3.16` compatibility of the shared `architectures` : the RNN masking is explicitly re-enabled (`enable_rnn_masking`), `backend_rnn` resolves the moved `keras.src.backend.rnn` module, and the masks are handled through `get_keras_mask` / `set_keras_mask` instead of the private `_keras_mask` attribute

### Bugs fixed

- `pip install -e .` works again with the recent `setuptools` versions : the `License ::` classifier, superseded by the PEP 639 `license` expression (`AGPL-3.0-or-later`), has been removed from the `pyproject.toml`

## [v1.0.0] - 21/07/2026 - Major refactoring release ! :yum:

First tagged release (`v1.0.0`) following a large refactoring of the OCR models on top of the restructured `BaseModel` foundation (see the [base_dl_project CHANGELOG](https://github.com/yui-mhcp/base_dl_project/blob/main/CHANGELOG.md)) and the shared `utils` / `loggers` socle (see the [data_processing CHANGELOG](https://github.com/yui-mhcp/data_processing/blob/main/CHANGELOG.md)) !

### Major updates

- The installation now relies on `pyproject.toml` instead of `requirements.txt` : install with `pip install -e .[tf]` (or `[torch]`) — the **image** and **text** stacks are core dependencies (OCR = EAST detection on images + CRNN recognition) — plus the optional helpers you need (`datasets`, `dev`)
- The backend is no longer hard-coded : it is selected at runtime through the `KERAS_BACKEND` environment variable, and the base install ships no backend
- The test suite has been migrated to `pytest`, mirroring the `utils` / `loggers` and model tree, with auto-skipped markers (`tensorflow`, `torch`, `keras`, `cv2`, `gpu`, `slow`, ...)
- The version is single-sourced in `__version__.py`
- **[BREAKING CHANGE]** the model interfaces have been fully restructured : the old `models/interfaces/base_*_model.py` classes are gone. `BaseModel` now lives in `models/core/base_model.py` and is composed from small, reusable **mixins** in `models/core/mixins/` (`audio`, `image`, `text`, `classification`, `processing`, `checkpoint`, `training`)
- **[BREAKING CHANGE]** model-core internals (saving / loading, legacy migration, description) have moved to `models/core/utils/{saving, migration, describe}.py`

### Minor updates

- New `models/ocr/box_filters.py` : filtering / post-processing helpers for detected text boxes
- New `models/ocr/streaming.py` and a `stream` entry-point (`from models.ocr import stream`) to run OCR on a stream of images
- The `models/detection` module keeps the `EAST` (scene-text detection) and `YOLOv2` (generic detection) implementations behind the shared `BaseDetector` interface
- The `architectures` module keeps the OCR / detection architectures (`crnn_arch`, `east_arch`, `yolo_arch`), the generic layers, transformer architectures and the `simple_models` helpers
- Pretrained `CRNN` (EasyOCR) and `EAST` (SakuraRiven) weights are still auto-downloaded and converted to `keras` on the fly (`torch` required for the conversion)
