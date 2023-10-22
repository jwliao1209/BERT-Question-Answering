#!/bin/bash

if [ ! -f data ]; then
    unzip data.zip
fi

if [ ! -f best_checkpoints ]; then
    unzip best_checkpoints.zip
fi

python convert_to_dataset.py \
       --inference \
       --preprocess mc \
       --context_data "${1}" \
       --test_data "${2}"

wait

python infer_mc.py \
       --checkpoint_folder best_checkpoints/mc_epoch=7_acc=0.9704 \

wait

python infer_qa.py \
       --checkpoint_folder best_checkpoints/qa_epoch=10_acc=83.9481

wait

python convert_to_submission.py \
       --output_path "${3}"
