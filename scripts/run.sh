#!/bin/bash

python infer_mc.sh --checkpoint_folder

wait

python infer_qa.sh --checkpoint_folder

wait

python convert_to_submission.py --prediction_path
