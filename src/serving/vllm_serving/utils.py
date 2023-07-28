import json
import os.path as osp
from typing import Union


class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_path: str = "", verbose: bool = False):
        self._verbose = verbose
        # if not osp.exists(template_path):
        #     raise ValueError(f"Can't read {template_path}")
        #
        # with open(template_path) as fp:
        #     self.template = json.load(fp)

        self.template = {
            "prompt_input": "<s>问：\n<input>\n答：\n",
            "response_split": "\n答：\n",
        }

    def generate_prompt(
            self,
            instruction: str,
            input: Union[None, str] = None,
            label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        res = self.template["prompt_input"].replace("<input>", input)

        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()