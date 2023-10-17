#!/bin/bash

python convert_to_dataset.py --preprocess mc
wait
python train_mc.py 
