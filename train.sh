#!/bin/bash

for i in 10 11 12 13 14 
do
	python train.py --seed $i --dataset cora  --attack meta --ptb_rate 0 --epoch 400 --alpha 1.0  --gamma 1.0 --lambda_ 0.001 --lr  1e-3 &
done
