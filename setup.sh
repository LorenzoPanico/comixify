#!/bin/bash

export CAFFE_ROOT='caffe_git/'

# Clone caffe repo to download pretrained agents
echo "Downloading caffe repo"
git clone -b 1.0 --depth 1 https://github.com/BVLC/caffe.git $CAFFE_ROOT

pushd $CAFFE_ROOT
echo "Downloading pretrained caffe models"
python scripts/download_model_binary.py 'models/bvlc_googlenet'
popd

echo "Unzip pretrained models"
# Unzip pretrained models
unzip popularity/pretrained_model/svr_test_11.10.sk.zip -d popularity/pretrained_model/

echo "Installing requirements"
pip install -r requirements.txt
