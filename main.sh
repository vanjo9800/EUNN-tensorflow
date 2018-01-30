#!/bin/bash

echo "----------------PREPROCESSING----------------"
python src/preprocess.py --data_dir=$1
echo "-------------------TRAINING------------------"
python src/train.py --model=$2 --train_data=$3
echo "-------------------TESTING-------------------"
python src/test.py --model=$2 --test_data=$3
echo "-------------------FINISHED------------------"
