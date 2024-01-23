# Author: Wei Li
# Email: wei.li@nus.edu.sg
import random

# import torch
# from torch.utils.data import Dataset
# from torch.nn.utils.rnn import pad_sequence
# import pickle
# import pandas as pd
# import numpy as np
# from torch.utils.data.sampler import SubsetRandomSampler
# from torch.utils.data import DataLoader

import openai
from openai import OpenAI
import tiktoken
from utils import TokenPricer
import os

def num_tokens_from_string(string, encoding_name):
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# OPENAI_API_KEY = "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" # luyao
# openai.api_key = "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" # wei

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)
max_tok = 1700


import inflect

def number_to_ordinal_words(number):
    if number not in ['N', 0]:
        p = inflect.engine()
        words = p.ordinal(number+1)
        return words
    else:
        return ''

def load_data(filename, path="./data/"):
    data = []
    with open(os.path.join(path, filename), "r") as f:
        lines = f.readlines()
        for line in lines:
            data.append(line.strip())

    return data


if __name__ == '__main__':

    scenarios = load_data('sce_topic_final.txt', path='./data')
    # get response from ChatGPT
    for scenario in scenarios:
        # chat = [{"role": 'system', "content": scenario}]

        Profile = "You are expert Alex. You are supposed to examine or evaluate several script scenes that happened in one person or between two persons." \
                  "Here are some examples of good script scenes: 'Max is goign to perform at a concert.', 'John's design is rejected by the magzaine.', 'Oliver is admitted to his dream university.'," \
                  "'Tom got promoted at the company.', 'Cynthia did not pass the math exam at the end of last term.' " \
                  "You have to carefully determine whether a script scene is good or bad. If good,  If not good, please modify the scene according to the above examples. The output should be in the following format:" \
                  + "[The script scene] {}".format(scenario) \
                  + "[Judgment]" \
                  + "[Modification]"

        query = "Please fill the Judgement. Please do not fill the modification unless the judgment is bad."

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": Profile + scenario,
                }
            ],
            model="gpt-3.5-turbo",
        )
        res = chat_completion.choices[0].message.content

        tokens = TokenPricer()
        print(tokens.gpt_get_estimated_cost(res, max_tokens=0))
        print(res)

    print("done!")