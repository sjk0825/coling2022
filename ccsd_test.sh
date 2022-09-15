#!/bin/bash
lang=c_functions #programming language
lr=3e-5
batch_size=256
beam_size=5
source_length=256
target_length=128
data_dir=../data/$lang/java
output_dir=./model_ccsd_cedge_30/$lang
dev_file=$data_dir/out_test
test_file=$data_dir/out_test
pretrained_model=microsoft/codebert-base #Roberta: roberta-base
config_name=microsoft/codebert-base
tokenizer_name=microsoft/codebert-base
test_model=$output_dir/checkpoint-best-bleu/pytorch_model.bin
npy_dir=$data_dir/npy
train_node_idx=$npy_dir/train_mask.npy
train_node_len=$npy_dir/train_node_len.npy 
train_node_edge=$npy_dir/train_edge.npy
valid_node_idx=$npy_dir/valid_mask.npy
valid_node_len=$npy_dir/valid_node_len.npy
valid_node_edge=$npy_dir/valid_edge.npy
test_node_idx=$npy_dir/out_test_mask.npy
test_node_len=$npy_dir/out_test_node_len.npy
test_node_edge=$npy_dir/out_test_edge.npy

singularity exec --nv ~/tf-1.11-gpu-py3.simg python3 run.py --do_test --model_type roberta --config_name $config_name --tokenizer_name $tokenizer_name --model_name_or_path $pretrained_model --load_model_path $test_model --dev_filename $dev_file --test_filename $dev_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --eval_batch_size $batch_size  --train_node_idx $train_node_idx --train_node_len $train_node_len --train_node_edge $train_node_edge --valid_node_idx $valid_node_idx --valid_node_len $valid_node_len --valid_node_edge $valid_node_edge --test_node_idx $test_node_idx --test_node_len $test_node_len --test_node_edge $test_node_edge 
