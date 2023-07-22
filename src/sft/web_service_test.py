


# coding=utf-8
# Created by Michael Zhu
# DataSelect AI, 2023

import json
import time

import urllib.request

import sys
sys.path.append("/")


def test_service():
    header = {'Content-Type': 'application/json'}

    # prompt = "<s>海明威的小说有哪些？"
    # prompt = "<s>中国有哪些旅游经典值得去："
    # prompt = "<s>\nhuman:\n你能写一首诗吗？\n</s><s>assistant:\n"
    # prompt = "<s>\nhuman:\n请介绍一下什么是钢琴？\n</s><s>assistant:\n"
    prompt = "<s>\nhuman:\n帮我写一个请假邮件，因为要去参加运动会，所以不去上课了\n</s><s>assistant:\n"
    # prompt = """<s>
    # human:\n你能写一首诗吗？
    # </s><s>
    # assistant:\n当然可以，请问您需要什么主题或情绪表达的内容? 我会用自己最好的方式来创作。
    # </s><s>
    # human:\n我想用这首诗歌表达对国人的祝福
    # </s><s>
    # assistant:\n"""
    # prompt = "<s>Can you write a poem for me? "

    data = {
          # "query": "<s>" + "[Round 0]\n问：男，目前28岁，最近几年，察觉，房事不太给力，另外，每次才开始就已经射了，请问：男生早泄是由于哪些原因诱发的。\n答：",
          "query": prompt,
          "max_new_tokens": 512,
          # "query": "<s>" + "[Round 0]\n问：2/3的患儿在使用免疫球蛋白后的24小时内即热退，90%的在48小时内热退，若48小时后体温仍较高，可考虑加用一次静脉免疫球蛋白1g/kg。\n这个句子里面实体有哪些？实体选项: 疾病，药物，身体部位，医疗程序，医学检验项目，医院科室\n答：上述句子中的实体包含：\n身体部位实体: \n药物实体: 免疫球蛋白\n疾病实体: \n医院科室实体: \n医学检验项目实体: 体温\n医疗程序实体: 静脉\n请根据上述例子进行回答\n问：实体抽取：\n主要表现为精神萎靡、嗜睡、呼吸深长呈叹息状，口唇樱红意识不清。\n选项:药物，身体部位，临床表现，医疗设备，微生物类，医疗程序，疾病，医学检验项目\n答："

    }
    request = urllib.request.Request(
        url='http://127.0.0.1:9008/llama_generate',
        headers=header,
        data=json.dumps(data).encode('utf-8')
    )
    response = urllib.request.urlopen(request)
    res = response.read().decode('utf-8')
    result = json.loads(res)
    print(json.dumps(data, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))

    return result


if __name__ == "__main__":

    test_service()

