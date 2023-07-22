lr=2e-5
lora_rank=20
lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
#lora_trainable="q_proj,v_proj,k_proj,o_proj"
modules_to_save="embed_tokens,lm_head"
lora_dropout=0.1
pretrained_model="/your_path/models--meta-llama--Llama-2-7b-hf/snapshots/4e4d531bcab430a66c4d562b7e89e21c0fa235ea"
tokenizer_name_or_path="/your_path/models--meta-llama--Llama-2-7b-hf/snapshots/4e4d531bcab430a66c4d562b7e89e21c0fa235ea"
dataset_name="/your_path/datasets/pretrain_data/sft_corpus/"
dataset_cache_dir="/your_path/datasets/sft_data/cache1"
per_device_batch_size=1
per_device_batch_size=1
gradient_accumulation_steps=160
training_steps=30000
output_dir="./experiments/output/chinese-llama2-sft-7b-pt-v0"
# deepspeed_config_file="internal/deepspeed_config_zero2.json"
deepspeed_config_file="internal/deepspeed_config_zero2_bf16.json"

torchrun \
  --nnodes 1 \
  --nproc_per_node 4 \
  --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:12589 \
   internal/run_clm_lora_instruct.py \
    --deepspeed ${deepspeed_config_file} \
    --model_name_or_path ${pretrained_model} \
    --tokenizer_name_or_path ${tokenizer_name_or_path} \
    --dataset_name ${dataset_name} \
    --dataset_cache_dir ${dataset_cache_dir} \
    --validation_split_percentage 0.001 \
    --per_device_train_batch_size ${per_device_batch_size} \
    --per_device_eval_batch_size ${per_device_batch_size} \
    --do_train \
    --seed 100 \
    --bf16 \
    --max_steps ${training_steps} \
    --lr_scheduler_type cosine \
    --leraning_rate ${lr} \
    --warmup_ratio 0.05 \
    --weight_decay 0.0001 \
    --logging_strategy steps \
    --logging_steps 1 \
    --save_strategy steps \
    --save_total_limit 500 \
    --save_steps 200 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --preprocessing_num_workers 8 \
    --block_size 1024 \
    --output_dir ${output_dir} \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --lora_rank ${lora_rank} \
    --trainable ${lora_trainable} \
    --modules_to_save ${modules_to_save} \
    --lora_dropout ${lora_dropout} \
    --torch_dtype auto

#   --deepspeed ${deepspeed_config_file} \

# CUDA_VISIBLE_DEVICES="0,1,2,3" ./src/sft/run_train_instruct.sh