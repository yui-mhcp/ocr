# :yum: Installation guide

This guide sets up a **GPU-enabled environment** (NVIDIA driver, CUDA, `mamba`, and the
deep-learning backends) shared by every `yui-mhcp` project. It is **project-agnostic**:
once the environment is ready, install a given project with the short *Installation*
section of its own `README.md` (typically `pip install -e .[<extras>]`).

> **Versions.** The exact versions below (CUDA 13.1 toolkit, `tensorrt_llm 1.3.0rc29` with
> `torch 2.14` cu132, the Blackwell TensorFlow wheel) are a **known-working combination tested
> on 2026-10**. Adjust them to your GPU / CUDA using the linked archives.

> **Distribution.** Commands target **Debian 13**; adapt the package-manager steps on other
> distributions.

> **Backends.** You only need the backend(s) your target project uses — see its `README.md`
> and the matching subsection of [§4](#4-python-environment) : TensorRT-LLM for
> `language_models`, TensorFlow for almost all the others, or both to combine an LLM with
> another project.

## 1. System packages

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y wget rsync git ffmpeg linux-headers-$(uname -r)

# Python build dependencies
sudo apt-get install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev \
    libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev libbz2-dev

# Required by some Python libraries (plotting, PDF, audio I/O)
sudo apt install -y graphviz poppler-utils lzma portaudio19-dev

# Required for TensorRT-LLM only
sudo apt-get install -y openmpi-bin libopenmpi-dev git-lfs
```

## 2. NVIDIA driver

### Blackwell generation (RTX 50xx)

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/debian13/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

sudo apt install -y linux-headers-amd64 dkms
sudo apt install -y nvidia-open

nvidia-smi
```

Verify the kernel module is loaded:

```bash
lsmod | grep nvidia
# if the list is empty:
sudo modprobe nvidia
```

### Alternative — manual installer

Download the driver from the [NVIDIA website](https://www.nvidia.com/Download/index.aspx)
(GeForce → *Proprietary*), then:

```bash
sudo sh <driver_file>.run
```

## 3. CUDA toolkit

Pick your version from the [CUDA archive](https://developer.nvidia.com/cuda-toolkit-archive).
Example for **CUDA 13.1** on **Debian 13**:

```bash
wget https://developer.download.nvidia.com/compute/cuda/13.1.0/local_installers/cuda_13.1.0_590.44.01_linux.run
sudo sh cuda_13.1.0_590.44.01_linux.run
```

## 4. Python environment

### Mamba (conda-forge)

```bash
mkdir -p installations
wget -O installations/Miniforge3.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh
bash installations/Miniforge3.sh -b -u -p ~/.conda

source ~/.conda/bin/activate
conda init --all
conda config --set auto_activate false

mamba update -n base -c conda-forge conda mamba -y
mamba shell init --shell bash --root-prefix ~/.conda
source ~/.bashrc
mamba deactivate
```

### Create the environment

```bash
mamba create -n yui python=3.12
mamba activate yui
```

Wire CUDA into the environment so it is on the path **only** while the env is active. The
snippets below write the resolved CUDA path into the activate/deactivate hooks:

```bash
CUDA_PATH=/usr/local/cuda-13.1

mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/initializer.sh" <<EOF
#! /bin/bash
export PATH="\$PATH:$CUDA_PATH/bin"
export LD_LIBRARY_PATH="\$LD_LIBRARY_PATH:\$CONDA_PREFIX/lib:$CUDA_PATH/lib64"
EOF

mkdir -p "$CONDA_PREFIX/etc/conda/deactivate.d"
cat > "$CONDA_PREFIX/etc/conda/deactivate.d/cleaner.sh" <<EOF
#! /bin/bash
export PATH=\$(echo \$PATH | sed -e "s|:$CUDA_PATH/bin||")
export LD_LIBRARY_PATH=\$(echo \$LD_LIBRARY_PATH | sed -e "s|:\$CONDA_PREFIX/lib:$CUDA_PATH/lib64||")
EOF
```

Re-activate to apply: `mamba deactivate && mamba activate yui`.

### Base libraries

```bash
pip install jupyter jupyterlab notebook
```

Then install the backend(s) your project needs, from the subsections below.

### TensorRT-LLM (`language_models`)

```bash
# TensorRT-LLM (NVIDIA index) pulls `torch` and `torchvision` built for CUDA 13.2 (PyTorch index)
pip install tensorrt_llm==1.3.0rc29 \
    --extra-index-url https://download.pytorch.org/whl/cu132 \
    --extra-index-url https://pypi.nvidia.com
```

### TensorFlow (almost all the other projects)

```bash
pip install tensorflow[and-cuda]
```

#### Blackwell GPUs (RTX 50xx)

TensorFlow 2.20 is not yet compiled with Blackwell support — install a custom wheel instead,
with its `[and-cuda]` extra (the CUDA 12 libraries):

```bash
# python 3.12 wheel:
pip install "tensorflow-2.20.0dev0+selfbuild-cp312-cp312-linux_x86_64.whl[and-cuda]"
```

Download it from [this release](https://github.com/mypapit/tensorflowRTX50/releases)
(3.11 / 3.12 / 3.13 builds also available
[here](https://github.com/nhsmit/tensorflow-rtx-50-series/releases/tag/2.20.0dev)).

### TensorRT-LLM + TensorFlow (an LLM combined with another project)

Install [TensorRT-LLM](#tensorrt-llm-language_models) **first**, then
[TensorFlow](#tensorflow-almost-all-the-other-projects) (Blackwell wheel included), to keep
matching cuda / cudnn versions.

**Blackwell wheel only — load TensorFlow with `RTLD_LOCAL`.** The custom wheel is a static
build : it loads its native library with `RTLD_GLOBAL`, which exports its own LLVM (21) to the
whole process. When `triton` (LLVM 23, imported by `torch` / `transformers` / TensorRT-LLM) is
loaded **after** TensorFlow, it binds to these symbols and the process crashes with a
*Segmentation fault* in `triton/knobs.py`. The hook below makes TensorFlow fall back to the
default `RTLD_LOCAL` flags, without modifying TensorFlow (so it survives its re-installation) :

```bash
SITE_PACKAGES=$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")

cat > "$SITE_PACKAGES/_tf_rtld_local.py" <<'EOF'
""" Load TensorFlow with `RTLD_LOCAL` : its `RTLD_GLOBAL` LLVM symbols crash `triton` """
import os
import sys
import types

if os.environ.get('TF_RTLD_GLOBAL', '0') != '1':
    _name = 'tensorflow.python.pywrap_dlopen_global_flags'
    _module = types.ModuleType(_name, __doc__)
    _module.set_dlopen_flags   = lambda: None
    _module.reset_dlopen_flags = lambda: None
    sys.modules.setdefault(_name, _module)
EOF

echo "import _tf_rtld_local" > "$SITE_PACKAGES/_tf_rtld_local.pth"
```

`TF_RTLD_GLOBAL=1` restores the original behavior : it is only required by custom ops loaded
with `tf.load_op_library`.

## 5. Verify the GPU

```bash
mamba activate yui
```

```python
# TensorFlow
import tensorflow as tf
print('GPU available (tensorflow):', tf.config.list_physical_devices('GPU') != [])

# TensorRT-LLM : imported AFTER tensorflow, this also checks the `RTLD_LOCAL` hook
import torch, torch._dynamo, tensorrt_llm
print('GPU available (torch):', torch.cuda.is_available())
```

Start Jupyter:

```bash
jupyter lab --ip 0.0.0.0 --allow-root
```

---

Your environment is ready. To install a specific project, follow the *Installation* section
of its `README.md` (e.g. `pip install -e .[image,tf]`).
