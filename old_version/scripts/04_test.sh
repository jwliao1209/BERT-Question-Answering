#!/bin/bash

python multiple_choice/test_mc.py
	--test_file data/test_mc.json
	--ckpt_dir outputs1
	--test_batch_size 32
	--out_file mc_pred_test.json

python question_answering/run_qa.py \
	--do_predict \
	--model_name_or_path question_answering/output1 \
	--output_dir question_answering/pred1 \
	--test_file mc_pred_test.json \
	--pad_to_max_length \
	--max_seq_length 512 \
	--doc_stride 128 \
	--per_gpu_eval_batch_size 10 \
