lr=5e-5
lora_rank=6
lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
#lora_trainable="q_proj,v_proj,k_proj,o_proj"
modules_to_save="embed_tokens,lm_head"
lora_dropout=0.1
pretrained_model="/public/home/xlwang2/codes/Chinese-LlaMA2/resources/models--meta-llama--Llama-2-13b-chat-hf/snapshots/7a4fd30fda5a3fef3bebda9b6d9e938e7a4d58ab"
tokenizer_name_or_path="/public/home/xlwang2/codes/Chinese-LlaMA2/experiments/output/chinese-llama2-chat-7b-pt-v3/checkpoint-1500"
dataset_name="/public/home/xlwang2/codes/Chinese-LlaMA2/datasets/pretrain_data/tiger_corpus/"
dataset_cache_dir="/public/home/xlwang2/codes/Chinese-LlaMA2/datasets/pretrain_data/cache"
per_device_batch_size=1
per_device_batch_size=1
gradient_accumulation_steps=256
training_steps=50000
output_dir="./experiments/output/chinese-llama2-chat-13b-fp-v0"
# deepspeed_config_file="internal/deepspeed_config_zero2.json"
#deepspeed_config_file="internal/deepspeed_config_zero2_bf16.json"
deepspeed_config_file="internal/deepspeed_config_zero2_bf16_offload.json"
#peft_path=None


torchrun \
  --nnodes 1 \
  --nproc_per_node 4 \
  --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:12589 \
   internal/run_clm_lora.py \
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
    --warmup_ratio 0.03 \
    --weight_decay 0.0001 \
    --logging_strategy steps \
    --logging_steps 1 \
    --save_strategy steps \
    --save_total_limit 500 \
    --save_steps 100 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --preprocessing_num_workers 8 \
    --block_size 512 \
    --output_dir ${output_dir} \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --lora_rank ${lora_rank} \
    --trainable ${lora_trainable} \
    --modules_to_save ${modules_to_save} \
    --lora_dropout ${lora_dropout} \
    --torch_dtype auto
#    --peft_path ${peft_path}

#   --deepspeed ${deepspeed_config_file} \

# CUDA_VISIBLE_DEVICES="0,1,2,3" ./internal/run_train_fp_13b.sh