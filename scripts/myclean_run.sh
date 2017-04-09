#!/bin/bash

TRAIN_DIR=/data/minglin/output/train/cifar10/ConvSLMnet_fc34/
rm -rf ${TRAIN_DIR}
chmod u+x scripts/ -R
mkdir -p ${TRAIN_DIR}
cp ./scripts/ming_train_conv_slm_net_on_cifar10.sh ${TRAIN_DIR}
cp ./nets/conv_slm.py ${TRAIN_DIR}
./scripts/ming_train_conv_slm_net_on_cifar10.sh
