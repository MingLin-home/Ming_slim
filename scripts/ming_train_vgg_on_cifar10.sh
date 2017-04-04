
#!/bin/bash
#
# This script performs the following operations:
# 1. Downloads the Cifar10 dataset
# 2. Trains a Vggnet model on the Cifar10 training set.
# 3. Evaluates the model on the Cifar10 testing set.
#
# Usage:
# cd slim
# ./scripts/train_cifar_net_on_mnist.sh

# Run training.
python train_image_classifier.py \
  --train_dir=/data/minglin/output/cifar10_train/cifarnet \
  --dataset_name=cifar10 \
  --dataset_split_name=train \
  --dataset_dir=/data/yelab/dataset_repo/cifar10 \
  --model_name=cifarnet \
  --preprocessing_name=cifarnet \
  --max_number_of_steps=100000 \
  --batch_size=128 \
  --save_interval_secs=120 \
  --save_summaries_secs=120 \
  --log_every_n_steps=100 \
  --optimizer=sgd \
  --learning_rate=0.1 \
  --learning_rate_decay_factor=0.1 \
  --num_epochs_per_decay=200 \
  --weight_decay=0.004 \
  --num_clones=4
  

# Run evaluation.
python eval_image_classifier.py \
  --checkpoint_path=/data/minglin/output/cifar10_train//cifarnet  \
  --eval_dir=/data/minglin/output/cifar10_train//cifarnet  \
  --dataset_name=cifar10 \
  --dataset_split_name=test \
  --dataset_dir=/data/yelab/dataset_repo/cifar10 \
  --model_name=cifarnet
