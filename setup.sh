#!/bin/bash

export CAFFE_ROOT='caffe_git/'

# Clone caffe repo to download pretrained agents
git clone -b 1.0 --depth 1 https://github.com/BVLC/caffe.git $CAFFE_ROOT
pushd $CAFFE_ROOT
python scripts/download_model_binary.py 'models/bvlc_googlenet'
popd

# Unzip pretrained models
unzip popularity/pretrained_model/svr_test_11.10.sk.zip -d popularity/pretrained_model/

pip install -r requirements.txt
