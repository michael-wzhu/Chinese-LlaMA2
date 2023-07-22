

## 不扩充词表的指令微调


### 数据格式

此处我们采用指令数据微调LlaMA-2模型，这些数据都是人机交互数据，一问一答形式，一般有两个角色, `human` 和 `assistant`. 所以我们采用如下的形式将数据拼接：
```text
<s>
human:
<query>
</s><s>
assistant:
<response>
</s>
```

如果是多轮对话，也是按照这个格式进行依次将对话进行拼接


### 训练

在`src/sft/run_train_sft.sh`根据本地环境配置模型路径，模型超参数等.我们训练采用deepspeed框架，模型训练过程中，我们加入并微调LoRA参数，同时也微调`embedding`层和`lm_head`层。微调`Llama-2-7b-hf`模型，运行：
```bash
CUDA_VISIBLE_DEVICES="0,1,2,3" ./src/sft/run_train_sft.sh

```

因为LlaMA-2开源了chat版本的模型，所以我们其实采用`Llama-2-7b-chat-hf`应该是更适合的。微调`Llama-2-7b-chat-hf`模型，运行：
```bash
CUDA_VISIBLE_DEVICES="0,1,2,3" ./src/sft/run_train_sft.sh

```


### 加载和起服务

### PEFT模型操作

加载训练保存的lora模块前要修改文件名。这个主要是hf的trainer中关于deepspeed的代码没完全适配PEFT。一般，运行上面的训练代码，模型文件保存在类似于`/your_path/checkpoint-100`名称的路径中，此路径下会有`pytorch_model.bin`文件，这个文件存储的是我们微调的参数。我们需要做如下两个步骤:
- 将`pytorch_model.bin`文件重命名为`adapter_model.bin`
- 将`src/sft/adapter_config.json`文件拷贝到`/your_path/checkpoint-100`路径


### 起服务

采用flask，对hf模型进行简单部署，请参照:

```bash
CUDA_VISIBLE_DEVICES=0 python src/sft/web_service_with_lora.py
```

测试服务时，参照`src/sft/web_service_test.py`中的请求格式发送请求。


### 高效推理部署

hf可以帮助我们快速部署模型，但其效率毕竟较低。为了实现LLM高效部署，请参照[vllm部署](./src/vllm_serving).


