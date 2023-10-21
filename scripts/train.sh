
python train_mc.py --lr_scheduler cosine --accum_grad_step 4 --epoch 5 --model_name_or_path bert-base-chinese
wait
python train_qa.py --lr_scheduler cosine --accum_grad_step 4 --epoch 5 --model_name_or_path bert-base-chinese
wait
