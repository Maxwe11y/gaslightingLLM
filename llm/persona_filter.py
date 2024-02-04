import argparse
import os
import json
import pandas as pd
import torch
import numpy as np
from openai import AsyncOpenAI
import asyncio
from tqdm import tqdm
import random

client = AsyncOpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)


def load_persona_json(filename, path):
    # load scene-persona pairs in python list
    sce_per = []
    with open(os.path.join(path, filename), 'r') as f:
        data = json.load(f)
        for scene in data:
            # personas.append(data[scene])
            sce_per.append([scene, data[scene]])
    f.close()

    return sce_per


async def chat(personas):
    chat_completion = await client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": personas,
            }
        ],
        model="gpt-3.5-turbo",
    )
    return chat_completion.choices[0].message.content


def check_velnerable(persona, num_per):

    prompt = "You are an expert psychologist specialized in analyzing persona. " \
                 + "\nGiven a set of personas below, kindly indicate 'yes' if it is a persona that would be attacked by malicious intentions, and 'no' if it is not: " \
                 + "\nPersonas:\n{}".format('\n'.join(persona)) \
                 + "\nThe general observations for the above {} personas are: \n".format(num_per)


    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(chat(prompt))

    while result.lower() not in ['entailment', 'contradiction', 'neutral']:
        result = loop.run_until_complete(chat(prompt))

    return result.lower()


if __name__ == "__main__":
    data = load_persona_json("match_sce_per_v2.json", './embedding')
    for (scene, personas) in data:
        check_velnerable(personas, len(personas))

    print('done!')