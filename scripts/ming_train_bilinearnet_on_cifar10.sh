#!/bin/bash
# Usage:
# ./scripts/train_cifar_net_on_mnist.sh


DATASET_NAME=cifar10
MODEL_NAME=bilinearnet
PREPROCESSING=cifarnet

DATASET_DIR=~/dataset_repo/${DATASET_NAME}
TRAIN_DIR=/data/minglin/output/train/${DATASET_NAME}/${MODEL_NAME}
mkdir -p ${TRAIN_DIR}

# Run training.
python train_image_classifier.py \
  --model_name=${MODEL_NAME} \
  --train_dir=${TRAIN_DIR} \
  --preprocessing_name=${PREPROCESSING} \
  --dataset_name=${DATASET_NAME} \
  --dataset_dir=${DATASET_DIR} \
  --dataset_split_name=train \
  --max_number_of_steps=100000 \
  --batch_size=128 \
  --save_interval_secs=120 \
  --save_summaries_secs=120 \
  --log_every_n_steps=100 \
  --optimizer=rmsprop \
  --learning_rate_decay_type=polynomial \
  --learning_rate=0.1 \
  --end_learning_rate=0.000001 \
  --num_epochs_per_decay=200 \
  --weight_decay=0.001 \
  --num_clones=4 \
  --clone_on_CPU=False \
  --moving_average_decay=0.9999 \
  2>&1 | tee -a ${TRAIN_DIR}/train_stdout.log


# Run evaluation.
python eval_image_classifier.py \
  --model_name=${MODEL_NAME} \
  --preprocessing_name=${PREPROCESSING} \
  --checkpoint_path=${TRAIN_DIR}  \
  --eval_dir=${TRAIN_DIR}  \
  --dataset_name=${DATASET_NAME} \
  --dataset_split_name=test \
  --dataset_dir=${DATASET_DIR} \
  2>&1 | tee -a ${TRAIN_DIR}/test_stdout.log
  