#!/bin/bash
lang=$1 #programming language
lr=3e-5
batch_size=32
beam_size=5
source_length=256
target_length=128
data_dir=/data/data/$1/java/
output_dir=/data/model_ccsd_cedge/$lang
train_file=$data_dir/train
dev_file=$data_dir/valid
eval_steps=1000 #400 for ruby, 600 for javascript, 1000 for others
train_steps=100000 #20000 for ruby, 30000 for javascript, 50000 for others
pretrained_model=microsoft/codebert-base #Roberta: roberta-base
config_name=microsoft/codebert-base
tokenizer_name=microsoft/codebert-base
load_model_path=$output_dir/checkpoint-best-bleu/pytorch_model.bin

singularity exec --nv ~/tf-1.11-gpu-py3.simg python3 run_ccsd_cedge.py --do_train --do_eval --model_type roberta --config_name $config_name --tokenizer_name $tokenizer_name --model_name_or_path $pretrained_model --train_filename $train_file --dev_filename $dev_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --train_batch_size $batch_size --eval_batch_size $batch_size --learning_rate $lr --train_steps $train_steps --eval_steps $eval_steps 
