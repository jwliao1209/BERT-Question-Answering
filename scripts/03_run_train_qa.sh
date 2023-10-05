#!/bin/bash

# python question_answering/run_qa_no_trainer.py \
#     --train_file data/train_qa.json \
#     --validation_file data/valid_qa.json \
#     --max_seq_length 512 \
#     --model_name_or_path hfl/chinese-roberta-wwm-ext-large \
#     --per_device_eval_batch_size 2 \
#     --gradient_accumulation_steps 32 \
#     --learning_rate 3e-5 \
#     --num_train_epochs 5 \
#     --num_warmup_steps 0 \
#     --with_tracking \
#     --output_dir question_answering/output1

    # --test_file data/test_qa.json \


python question_answering/run_qa.py \
  --do_train \
  --do_eval \
  --model_name_or_path hfl/chinese-roberta-wwm-ext-large \
  --output_dir  question_answering/output1 \
  --train_file data/train_qa.json \
  --validation_file data/valid_qa.json \
  --cache_dir ./cache/qa \
  --per_gpu_train_batch_size 10 \
  --gradient_accumulation_steps 8 \
  --per_gpu_eval_batch_size 10 \
  --eval_accumulation_steps  8 \
  --learning_rate 3e-5 \
  --num_train_epochs 10 \
  --max_seq_length 512 \
  --doc_stride 128 \
  --warmup_ratio 0.1 \
  --overwrite_output_dir
