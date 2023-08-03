# Chinese-LlaMA2

<p align="center">
    <br>
    <img src="./assets/chinese-llama2-banner.png" width="600"/>
    <br>
</p>
<p align="center">
    <img alt="GitHub" src="https://img.shields.io/github/license/ymcui/Chinese-LLaMA-Alpaca.svg?color=blue&style=flat-square">
    <img alt="GitHub top language" src="https://img.shields.io/github/languages/top/ymcui/Chinese-LLaMA-Alpaca">
</p>

就在不久前，Meta最新开源了Llama 2模型，完全可商用，看来Meta势必要与OpenAI (ClosedAI) 硬刚到底。虽然Llama 2对原版的LlaMA模型做了升级，但是其仍然对中文没有太好的支持，需要在中文上做定制化。所以我们决定在次开展Llama 2的中文汉化工作：
- 🚀 [Chinese-LlaMA2-chat-sft](https://huggingface.co/michaelwzhu/Chinese-LlaMA2-chat-7B-sft-v0.3)：对Llama-2直接进行有监督微调，
  - 采用开源指令微调数据，如UltraChat, 各种版本的中文alpaca语料(如Chinese-alpaca, BELLE)等；
  - 注意LlaMA词表本身是支持中文的，所以我们会训练不扩充词表版本和扩充词表版本
- ⏳ [Chinese-LlaMA2](https://huggingface.co/michaelwzhu/Chinese-LlaMA2-7B): 对Llama 2进行大规模中文预训练；
  - 第一步：先在42G中文语料上进行训练；后续将会加大训练规模 
- ⏳ [Chinese-LlaMA2-chat](https://huggingface.co/michaelwzhu/Chinese-LlaMA2-7B-chat): 对[Chinese-LlaMA2](https://huggingface.co/michaelwzhu/Chinese-LlaMA2-7B)进行指令微调和多轮对话微调，以适应各种应用场景和多轮对话交互。

注意，遵循相应的许可，我们将发布完整的, 合并LoRA权重的完整，且同时发布LoRA权重，方便开源社区使用。

同时，我们将会围绕Chinese-LlaMA2打造各种垂直领域模型：
- ⏳[Chinese-LlaMA2-chatmed](https://huggingface.co/michaelwzhu/Chinese-LlaMA2-7B-chatmed): Chinese-LlaMA2医学领域大模型，支持多轮在线问诊；
- ⏳[Chinese-LlaMA2-tcm](https://huggingface.co/michaelwzhu/Chinese-LlaMA2-7B-tcm): Chinese-LlaMA2中医药大模型，专注于中医药细分领域，赋能中医药传承

----

[中文医疗大模型ChatMed](https://github.com/michael-wzhu/ChatMed) |  [业内首个中医药大模型ShenNong-TCM-LLM](https://github.com/michael-wzhu/ShenNong-TCM-LLM) | [PromptCBLUE-中文医疗大模型评测基准](https://github.com/michael-wzhu/PromptCBLUE) | [PrompTS-探索采用大模型处理各种时间序列任务](https://github.com/michael-wzhu/PrompTS)


## 更新

2023/07/28 更新了扩展词表，且微调了300w中文指令数据的模型[Chinese-LlaMA2-chat-sft-v0.3](https://huggingface.co/michaelwzhu/Chinese-LlaMA2-chat-7B-sft-v0.3) (v0.2), 包含LoRA参数和合并后的完整模型参数。这一模型还在继续训练中，训练规模将会达到1500w条中文指令/对话数据; 同时更新了量化模型进行部署的代码和gradio demo的代码

2023/07/25 更新了一个不扩充词表，微调了300w中文指令数据的模型[Chinese-LlaMA2-chat-sft](https://huggingface.co/michaelwzhu/Chinese-LlaMA2-chat-7B-sft) (v0.2), 包含LoRA参数和合并后的完整模型参数。扩充词表，进行指令微调的模型将于两日后发布

2023/07/24 更新了一个不扩充词表，微调了100w中文指令数据的模型[Chinese-LlaMA2-chat-sft](https://huggingface.co/michaelwzhu/Chinese-LlaMA2-chat-7B-sft) (v0.1)
  - 几个测试例子见[test examples](./assets/20230724/test_examples.json)；
  - 此模型是PEFT模型加上微调后的embedding和LM head参数，使用部署请参照[SFT-README](./src/sft/SFT-README.md)或者[vllm-REAME](./src/vllm_serving/REAME.md);
  - 此模型具有初步的中文沟通和任务能力，但是中文知识仍然有限；我们会持续更新更加强大的版本

2023/07/20 采用开源中文指令数据对LlaMA-2-7B进行微调(不扩充词表/扩充词表); 采用vllm对模型进行serving

2023/07/19 启动LlaMA-2中文大模型项目；


## 为什么LlaMA-2需要汉化？

我们发现，Meta开源的LlaMA-2模型虽然是支持中文的，但是其在被要求做中文生成时，容易产生中英混杂或者全是英文的现象（参看[example1](./assets/llama-2_original_example1.png)）。所以为了更好的支持中文应用和落地，对齐做中文适配是必经之路。

但是这里有两个需要思考的地方：
- 是否需要扩充词表？ 是否扩充词表后效果会更好？另外扩充词表必然是需要更大规模预训练的，毕竟会引入初始化参数；
- 是否需要中文上的预训练？多大的中文预训练合适？在中文上预训练是否会退化其英文AIGC能力？


## 快速上手

### 获得llama-2权重

现在LlaMA-2权重需要在Meta指定的官方网站申请，具体说明见[LlaMA-的hf页面](https://huggingface.co/meta-llama/Llama-2-70b-hf)。当你没有通过申请时，在这个网页上看到的是一个申请表，你需要根据他的说明进行申请，申请通过后就可以看到权重文件了。

下载模型权重，运行：
```bash
src/further_ft/download_checkpoints.py
```

### 指令微调

对LlaMA-2进行指令微调(不扩充词表/扩充词表)，也就是现在常见的SFT，见[SFT-README.md](./src/sft/SFT-README.md);

### model serving

- vllm 部署：模型部署采用huggingface原生代码效率比较慢，为了获得2.7倍左右推理速度提升，我们采用vllm框架进行部署，操作步骤参照[vllm-serving-README](./src/serving/vllm_serving/REAME.md).
- 模型量化：参考ChatGLM的量化代码，对Chinese-llama2模型进行量化。详见[量化部署代码](./src/serving/web_service_with_quantized_model.py)
- gradio demo代码：见[gradio demo code](./src/serving/gradio_demo.py)


### 扩充词表和扩展embedding层

我们现在采用的方案是：使用[Chinese-LLaMA](https://github.com/ymcui/Chinese-LLaMA-Alpaca)的词表，该词表是对llama原始词表的扩充，将词汇量从32000扩展到49953大小。同时LlaMA-2模型会进行embedding层的resize，即采用随机初始化的参数扩展embedding层和lm_head层。

在一些我们关注的垂直领域，我们后续也会自己训一个sentencepiece模型来更新llama-2的词表。


### 继续预训练

由于扩展词表后，LlaMA-2的embedding层和lm_head层会有随机初始化的参数，所以我们需要采用大规模的预训练学习中文语料的知识。继续预训练运行以下命令(数据，模型的路径，卡数等需要自行配置)：

```bash
CUDA_VISIBLE_DEVICES="2,3" ./src/further_ft/run_train.sh

```



## 评测交流与技术交流

PromptCBLUE与大模型技术交流微信交流群二维码（截止至7月23日有效）：
<p align="left">
    <br>
    <img src="./assets/wechat_qrcode.jpg" width="300"/>
    <br>
</p>


## 团队介绍

本项目由华东师范大学计算机科学与技术学院智能知识管理与服务团队完成，团队指导老师为王晓玲教授。

团队成员：
- [michael-wzhu](https://github.com/michael-wzhu)