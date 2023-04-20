from collections import OrderedDict
import logging
import re
import time
from transformers import AutoTokenizer, AutoConfig, LlamaForCausalLM
import torch
import subprocess
import os

from subprocess import DEVNULL, STDOUT
from tensorizer import TensorDeserializer
from tensorizer.utils import no_init_or_tensor

from subclass import YieldingLlama

DEFAULT_MODEL_NAME = "weights/vicuna-13b/tensorizer_weights/vicuna-13b-16fp.tensors"  # path from which we pull weights when there's no COG_WEIGHTS environment variable
DEFAULT_CONFIG_PATH = "weights/vicuna-13b/tensorizer_weights/config.json"
TOKENIZER_PATH = "tokenizer/LLaMA-13b"

CONFIG_LOCATION = "llama_weights/llama-13b"

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"


def load_tokenizer():
    """Same tokenizer, agnostic from tensorized weights/etc"""
    print(TOKENIZER_PATH)
    print(os.listdir(TOKENIZER_PATH))
    tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    tok.add_special_tokens(
        {
            "eos_token": DEFAULT_EOS_TOKEN,
            "bos_token": DEFAULT_BOS_TOKEN,
            "unk_token": DEFAULT_UNK_TOKEN,
            "pad_token": DEFAULT_PAD_TOKEN,
        }
    )
    return tok


def load_tensorizer(
    weights, plaid_mode: bool = True, cls: LlamaForCausalLM = YieldingLlama, config_path = None,
):
    st = time.time()
    weights = str(weights)



    if os.path.exists(weights):
        config = AutoConfig.from_pretrained(config_path)
        logging.disable(logging.WARN)
        model = no_init_or_tensor(
            lambda: cls.from_pretrained(
                None, config=config, state_dict=OrderedDict(), torch_dtype=torch.float16
            )
        )
        logging.disable(logging.NOTSET)

        des = TensorDeserializer(weights, plaid_mode=plaid_mode)
        des.load_into_module(model)
        print(f"weights loaded in {time.time() - st}")

    else:
        


        pattern = r'https://pbxt\.replicate\.delivery/([^/]+/[^/]+)'
        match = re.search(pattern, weights)
        if match:
            weights = f"gs://replicate-files/{match.group(1)}"


        print(f"deserializing weights")
        local_weights = "/src/llama_tensors"
        command = (
            f"/gc/google-cloud-sdk/bin/gcloud storage cp {weights} {local_weights}".split()
        )
        res = subprocess.run(command)
        if res.returncode != 0:
            raise Exception(
                f"gcloud storage cp command failed with return code {res.returncode}: {res.stderr.decode('utf-8')}"
            )
        config = AutoConfig.from_pretrained(CONFIG_LOCATION)

        logging.disable(logging.WARN)
        model = no_init_or_tensor(
            lambda: cls.from_pretrained(
                None, config=config, state_dict=OrderedDict(), torch_dtype=torch.float16
            )
        )
        logging.disable(logging.NOTSET)

        des = TensorDeserializer(local_weights, plaid_mode=plaid_mode)
        des.load_into_module(model)
        print(f"weights loaded in {time.time() - st}")

    return model
