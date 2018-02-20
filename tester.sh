#!/bin/bash

for i in `seq 4 20`;
do
	python3.5 babi_rnn.py LSTM $i | tee test6QA${i}LSTM_50.txt;
	python3.5 babi_rnn.py EUNN $i | tee test6QA${i}EUNNTun2_200.txt;
done
