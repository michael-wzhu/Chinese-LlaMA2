# coding=utf-8
# Created by Michael Zhu
# DataSelect AI, 2023

import json
import time

import urllib.request

import sys
sys.path.append("./")


def test_service(input_text):
    header = {'Content-Type': 'application/json'}

    prompt = "<s>问：\n{}\n答：\n".format(input_text.strip())

    data = {
          "input": input_text.strip().replace("答：", ""),
          "parameters": {},
    }
    request = urllib.request.Request(
        url='http://127.0.0.1:9005/chatmed_generate',
        headers=header,
        data=json.dumps(data).encode('utf-8')
    )

    result = None
    try:
        response = urllib.request.urlopen(request, timeout=30)
        res = response.read().decode('utf-8')
        result = json.loads(res)
        print(json.dumps(data, ensure_ascii=False, indent=2))
        print(json.dumps(result, ensure_ascii=False, indent=2))

    except Exception as e:
        print(e)

    return result


if __name__ == "__main__":

    f_out = open("src/web_services/test_examples/test_preds_0.json", "a", encoding="utf-8", buffering=1)
    with open("data/promptcblue/test_a_open_0/dev.json", "r", encoding="utf-8") as f:

        for line in f:
            line = line.strip()
            if not line:
                continue

            line = json.loads(line)

            t0 = time.time()
            result = test_service(line["input"])
            t1 = time.time()
            print("time cost: ", t1 - t0)

            f_out.write(
                json.dumps(result, ensure_ascii=False) + "\n"
            )

