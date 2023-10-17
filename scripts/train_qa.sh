#!/bin/bash

python convert_to_dataset.py --preprocess qa
python train_qa.py
