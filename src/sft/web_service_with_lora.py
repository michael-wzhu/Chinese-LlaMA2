import json
import time

import torch
from transformers import AutoConfig

import sys


sys.path.append("./")

from peft import PeftModel

from internal.modeling_llama import LlamaForCausalLM
from internal.tokenization_llama import LlamaTokenizer
from internal.configuration_llama import LlamaConfig

model_path = "/public/home/xlwang2/codes/Chinese-LlaMA2/resources/models--meta-llama--Llama-2-7b-chat-hf/snapshots/902d6349aebd6a0f82b1f6280ea57f65e0d03638"

config = LlamaConfig.from_pretrained(
    model_path,
)
print(config)

with torch.no_grad():
    torch_dtype = torch.bfloat16
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch_dtype,
        # low_cpu_mem_usage=True
    )
    print(model)
    # model = model.cuda()

    tokenizer = LlamaTokenizer.from_pretrained(
        model_path,
    )
    print(tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.resize_token_embeddings(len(tokenizer))

    # 加载lora
    # peft_model_path = "resources/ChatMed-Consult_llama_lora_pt_v0"
    peft_model_path = "/your_peft_path"
    model = PeftModel.from_pretrained(model, peft_model_path)
    model.eval()
    print(model)

generation_config = dict(
    do_sample=False,
    num_beams=1,
    repetition_penalty=1.3,
    max_new_tokens=512
)

from flask import Flask, request

app = Flask(__name__)


@app.route("/llama_generate", methods=["POST"])
def cough_predict():
    input_data = json.loads(
        request.get_data().decode("utf-8")
    )

    query = input_data.get("query")
    max_new_tokens = input_data.get("max_new_tokens", 256)

    t0 = time.time()
    with torch.no_grad():
        # device = torch.device("cuda")
        device = torch.device("cpu")
        inputs = tokenizer(query, return_tensors="pt", add_special_tokens=False)  # add_special_tokens=False ?
        generation_output = model.generate(
            # input_ids=inputs["input_ids"].to(device),
            # attention_mask=inputs['attention_mask'].to(device),

            input_ids=inputs["input_ids"],
            attention_mask=inputs['attention_mask'],
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            **generation_config
        )
        s = generation_output[0]
        print(s)
        output = tokenizer.decode(s, skip_special_tokens=True)

        response = output

    print(output)

    t1 = time.time()
    print("time cost: ", t1 - t0)

    return {
        "query": query,
        "response": response
    }


app.run(host="0.0.0.0", port=9008, debug=False)

'''

CUDA_VISIBLE_DEVICES=0 python internal/web_service_with_lora.py


'''
