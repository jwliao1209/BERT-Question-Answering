#!/bin/bash

python multiple_choice/run_swag_no_trainer.py \
    --train_file data/train_mc.json \
    --validation_file data/valid_mc.json \
    --max_seq_length 512 \
    --per_device_train_batch_size 32 \
    --per_device_train_batch_size 2 \
    --lr_scheduler_type cosine \
    --model_name_or_path hfl/chinese-bert-wwm-ext \
    --tokenizer_name bert-base-chinese \
    --with_tracking \
    --output_dir outputs
