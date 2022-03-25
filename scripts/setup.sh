#!/bin/bash
if [ ! -f $HOME/miniconda/bin/python ]; then
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
        bash ~/miniconda.sh -b -p $HOME/miniconda
fi

$HOME/miniconda/bin/python -m venv ./env-nvdiffrec
source ./env-nvdiffrec/bin/activate
pip install --upgrade pip
pip install --upgrade setuptools
pip install --upgrade wheel

pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip install --global-option="--no-networks" git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install -r requirements.txt

imageio_download_bin freeimage

cd data
python download_datasets.py
