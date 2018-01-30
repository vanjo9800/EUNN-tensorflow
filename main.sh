#!/bin/bash

echo "-------------------TRAINING------------------"
python3.6 main.py --model=EUNN --train_data="../datasets/fb-babi/processed/en-10k" 
