import gradio as gr
import mdtex2html
import urllib
import json
def test_service(input_text, history=None):
    header = {'Content-Type': 'application/json'}

    system_prompt = """<<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>"""

    prompt = ""
    for i, round in enumerate(history):

        if i == 0:
            prompt += f"<s>[INST]{system_prompt}\n{round[0].strip()}\n[/INST]\n{round[1].strip()}</s>"
        else:
            prompt += f"<s>[INST]\n{round[0].strip()}\n[/INST]\n{round[1].strip()}</s>"

    prompt += f"<s>[INST]\n{input_text.strip()}\n[/INST]\n"

    data = {
          "query": prompt,
          "max_new_tokens": 1024,
    }
    request = urllib.request.Request(
        url='http://127.0.0.1:9005/llm_generate',
        headers=header,
        data=json.dumps(data).encode('utf-8')
    )
    response = urllib.request.urlopen(request)
    res = response.read().decode('utf-8')
    result = json.loads(res)
    # print(json.dumps(data, ensure_ascii=False, indent=2))
    # print(json.dumps(result, ensure_ascii=False, indent=2))

    return result

"""Override Chatbot.postprocess"""


def postprocess(self, y):
    if y is None:
        return []

    # print("y: ", y)
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y


gr.Chatbot.postprocess = postprocess


def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    text = "".join(lines)
    return text


def predict(input, chatbot, max_length, top_p, temperature, history):
    chatbot.append((parse_text(input), ""))

    print("input: ", input)
    print("history: ", history)
    # response = api.decode(
    #     input,
    #     history=history,
    #     max_length=max_length,
    #     top_p=top_p,
    #     temperature=temperature
    # )
    response = test_service(
        input,
        history=history,
        # max_length=max_length,
        # top_p=top_p,
        # temperature=temperature
    )
    response = response.get("response", "")
    print(response)
    print(parse_text(response))
    chatbot[-1] = (parse_text(input), parse_text(response))
    history.append((input, response))
    print(history)
    yield chatbot, history


def reset_user_input():
    return gr.update(value='')


def reset_state():
    return [], []


with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">Chinese-LlaMA2 - demo </h1>""")
    chatbot = gr.Chatbot()
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Column(scale=12):
                user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10).style(
                    container=False)
            with gr.Column(min_width=32, scale=1):
                submitBtn = gr.Button("Submit", variant="primary")

        with gr.Column(scale=1):
            emptyBtn = gr.Button("Clear History")
            max_length = gr.Slider(0, 1024, value=512, step=1.0, label="Maximum length", interactive=True)
            top_p = gr.Slider(0, 1, value=0.7, step=0.01, label="Top P", interactive=True)
            temperature = gr.Slider(0, 1, value=0.95, step=0.01, label="Temperature", interactive=True)

    history = gr.State([])

    submitBtn.click(predict,
                    [user_input, chatbot, max_length, top_p, temperature, history],
                    [chatbot, history],
                    show_progress=True
                    )
    submitBtn.click(reset_user_input, [], [user_input])

    emptyBtn.click(reset_state, outputs=[chatbot, history], show_progress=True)

demo.queue().launch(share=True, inbrowser=True)
