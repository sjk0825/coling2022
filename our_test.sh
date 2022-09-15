#!/bin/bash
lang=tl_codesum #programming language
lr=3e-5
batch_size=64
beam_size=5
source_length=256
target_length=128
data_dir=data/$1/java
output_dir=model/$lang
dev_file=$data_dir/test
test_file=$data_dir/test
test_model=$output_dir/checkpoint-best-bleu/pytorch_model.bin
pretrained_model=microsoft/codebert-base #Roberta: roberta-base
config_name=microsoft/codebert-base
tokenizer_name=microsoft/codebert-base

CUDA_VISIBLE_DEVICES=0,1 python3 run.py --do_test --model_type roberta --config_name $config_name --tokenizer_name $tokenizer_name --model_name_or_path $pretrained_model --load_model_path $test_model --dev_filename $dev_file --test_filename $test_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --eval_batch_size $batch_size
