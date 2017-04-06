#!/bin/bash

TRAIN_DIR=/data/minglin/output/train/cifar10/bilinearnet/
rm -rf ${TRAIN_DIR}
chmod u+x scripts/ -R
mkdir -p ${TRAIN_DIR}
cp ./scripts/ming_train_bilinearnet_on_cifar10.sh ${TRAIN_DIR}
./scripts/ming_train_bilinearnet_on_cifar10.sh