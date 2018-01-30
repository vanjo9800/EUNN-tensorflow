#!/bin/bash

echo "-------------------TRAINING------------------"
python3.6 train.py --model=EUNN --train_data="../datasets/fb-babi/processed/en-10k" 
