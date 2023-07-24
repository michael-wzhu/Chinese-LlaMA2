

## vllm部署操作步骤


预测时，我们采用[vllm项目](https://github.com/vllm-project/vllm)对模型进行serving.同时这部分代码参照了[KnowLM项目](https://github.com/zjunlp/KnowLM/tree/main/inference)

在使用vllm时，我们首先需要把训练得到的lora参数与LlaMA主干进行合并 (假设我们采用训练第800步的lora权重)：

```bash

CUDA_VISIBLE_DEVICES="3" python src/vllm_serving/merge_llama_with_lora.py \
    --base_model /your_path/models--meta-llama--Llama-2-7b-chat-hf \
    --lora_model /your_path/peft_path \
    --output_type huggingface \
    --output_dir /your_path/merged_model_path

```

然后采用下面的命令启动模型服务。注意，我们修改了`src/vllm_serving/llm_engine.py`第148行的`gpu_memory_utilization`参数取值，大家可以根据显卡情况修改。

```bash
CUDA_VISIBLE_DEVICES="3" python src/ft_llama_lora/vllm_serving/launch_vllm.py \
    --port 8000 \
    --model /your_path/merged_model_path \
    --use-np-weights \
    --max-num-batched-tokens 4096 \
    --dtype half \
    --tensor-parallel-size 1

```

我们在生成的时候，不会传入有效的`parameters`字段，所以采样参数会使用`src/vllm_serving/launch_vllm.py`的63行处`SamplingParams`的默认值。大家可以根据需求修改。vllm服务起好之后，我们可以通过下面的例子进行服务调用，从而进行测试集预测：

```bash
src/vllm_serving/web_service_test.py

```

通过vllm部署模型，我们测试下来预计加速2.5倍左右。
