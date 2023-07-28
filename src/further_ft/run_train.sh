lr=1e-4
lora_rank=24
lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
#lora_trainable="q_proj,v_proj,k_proj,o_proj"
modules_to_save="embed_tokens,lm_head"
lora_dropout=0.1
pretrained_model="/you_path_to_llama_hf_ckpt"
tokenizer_name_or_path="/public/home/xlwang2/codes/Med_Prompts/models--ziqingyang--chinese-llama-plus-lora-7b/snapshots/32115d9a87767a8e00464dc560030a12bf38cb24"
dataset_name="/public/home/xlwang2/codes/Chinese-LlaMA2/datasets/pretrain_data/"
dataset_cache_dir="/public/home/xlwang2/codes/Chinese-LlaMA2/datasets/pretrain_data/cache"
per_device_batch_size=2
per_device_batch_size=2
gradient_accumulation_steps=128
training_steps=10000
output_dir="./experiments/output/chinese-llama2-7b-pt-v0"
deepspeed_config_file="src/further_ft/deepspeed_config_zero2_bf16.json"

torchrun \
  --nnodes 1 \
  --nproc_per_node 2 \
  --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:12356 \
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
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_total_limit 25 \
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

#   --deepspeed ${deepspeed_config_file} \

# CUDA_VISIBLE_DEVICES="2,3" ./internal/run_train.sh