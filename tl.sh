#!/bin/bash
lang=tl_codesum #programming language
lr=3e-5
batch_size=32
beam_size=5
source_length=256
target_length=128
data_dir=../data/$lang/java/
output_dir=./model_tl_30/$lang
train_file=$data_dir/train
dev_file=$data_dir/valid
eval_steps=1000 #400 for ruby, 600 for javascript, 1000 for others
train_steps=66000 #20000 for ruby, 30000 for javascript, 50000 for others
pretrained_model=microsoft/codebert-base #Roberta: roberta-base
config_name=microsoft/codebert-base
tokenizer_name=microsoft/codebert-base
load_model_path=$output_dir/checkpoint-best-bleu/pytorch_model.bin
npy_dir=$data_dir/npy
train_node_idx=$npy_dir/tl_train_tokens_mask.npy
train_node_len=$npy_dir/tl_train_node_len.npy 
train_node_edge=$npy_dir/tl_train_edge.npy
valid_node_idx=$npy_dir/tl_valid_tokens_mask.npy
valid_node_len=$npy_dir/tl_valid_node_len.npy
valid_node_edge=$npy_dir/tl_valid_edge.npy
test_node_idx=$npy_dir/tl_test_tokens_mask.npy
test_node_len=$npy_dir/tl_test_node_len.npy
test_node_edge=$npy_dir/tl_test_edge.npy

singularity exec --nv ~/tf-1.11-gpu-py3.simg python3 run.py --do_train --do_eval --model_type roberta --config_name $config_name --tokenizer_name $tokenizer_name --model_name_or_path $pretrained_model --train_filename $train_file --dev_filename $dev_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --train_batch_size $batch_size --eval_batch_size $batch_size --learning_rate $lr --train_steps $train_steps --eval_steps $eval_steps --train_node_idx $train_node_idx --train_node_len $train_node_len --train_node_edge $train_node_edge --valid_node_idx $valid_node_idx --valid_node_len $valid_node_len --valid_node_edge $valid_node_edge --test_node_idx $test_node_idx --test_node_len $test_node_len --test_node_edge $test_node_edge 
