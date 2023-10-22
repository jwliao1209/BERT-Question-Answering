#!/bin/bash

python convert_to_dataset.py --preprocess qa
wait
python train_qa.py
