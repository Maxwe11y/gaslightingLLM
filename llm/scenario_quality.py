# Author: Wei Li
# Email: wei.li@nus.edu.sg

from openai import OpenAI, AsyncOpenAI
import tiktoken
from utils import TokenPricer
import os
import inflect
import random
import string
import re
import asyncio
import numpy as np
def num_tokens_from_string(string, encoding_name):
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# OPENAI_API_KEY = "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" # luyao
# openai.api_key = "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" # wei

# client = OpenAI(
#     # This is the default and can be omitted
#     api_key=os.environ.get("OPENAI_API_KEY"),
# )

client = AsyncOpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

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


def check_scenario(scenario):

    pattern = re.compile(r'^[0-9]+')

    result = pattern.sub('', scenario)

    return result

def get_hints(tokens, scenarios, num_hints, hint_limit):
    shuffled_scenario = random.sample(scenarios, len(scenarios))
    # shuffled_scenario = np.random.shuffle(scenarios)
    hints = []
    for scenario in shuffled_scenario:
        if tokens.gpt_get_estimated_cost(scenario, max_tokens=0) <= hint_limit:
            hints.append(scenario)
        if len(hints) == num_hints:
            break

    # length = [tokens.gpt_get_estimated_cost(hint, max_tokens=0) for hint in hints]
    print('the actual hints are {}'.format(len(hints)))
    return hints


async def chat(Profile_new):
    chat_completion = await client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": Profile_new,
            }
        ],
        model="gpt-3.5-turbo",
    )
    return chat_completion.choices[0].message.content


if __name__ == '__main__':

    max_tok = 2000

    # scenarios = load_data('sce_topic_final.txt', path='./data')
    # # get response from ChatGPT
    # for scenario in scenarios:
    #     # chat = [{"role": 'system', "content": scenario}]
    #
    #     Profile = "You are expert Alex. You are supposed to examine or evaluate several script scenes that happened in one person or between two persons." \
    #               "Here are some examples of good script scenes for your reference: 'Max is going to perform at a concert.', 'John's design is rejected by the magzaine.', 'Oliver is admitted to his dream university.'," \
    #               "'Tom got promoted at the company.', 'Cynthia did not pass the math exam at the end of last term.' " \
    #               "You have to carefully determine whether a script scene is good or bad based on the good script scenes above. If good, please fill [Modification] with NA. If bad, please modify the scene using the aforementioned good script scenes as references. " \
    #                "The [Modification] should be precise and short. Given a script scenario below, please fill the Judgment. Please do not fill the modification unless the Judgment is bad. \n" \
    #               + "[The script scene] {} \n".format(scenario) \
    #               + "[Judgment] \n" \
    #               + "[Modification]"
    #
    #
    #     query = "Please fill the Judgement. Please do not fill the modification unless the Judgment is bad."
    #
    #     chat_completion = client.chat.completions.create(
    #         messages=[
    #             {
    #                 "role": "system",
    #                 "content": Profile,
    #             }
    #         ],
    #         model="gpt-3.5-turbo",
    #     )
    #     res = chat_completion.choices[0].message.content
    #
    #     tokens = TokenPricer()
    #     print(tokens.gpt_get_estimated_cost(res, max_tokens=0))
    #     print(res)


    generated_scenarios = []

    example_hints = ['Max is going to perform at a concert.', "John's design is rejected by the magzaine.", "Anderson is rejected by his dream school",
                     'Oliver is admitted to his dream university', 'Tom got promoted at the company.', 'Cynthia did not pass the math exam at the end of last term.']

    num_gen = 40
    target = 5000
    num_hints = 5
    hint_limit = 15
    tokens = TokenPricer()

    while len(generated_scenarios) <= target:

        Profile_new = "You are expert Alex. You are supposed to provide several short, diverse, concise and precise script scenes described in one short sentence. " \
                      "Please make sure the script scence starts from the name of the character. Here are some examples of good script scenes for your reference: {} ".format(example_hints) \
                      + (" Please generate {} good script scenes using the aforementioned good script scenes as references."
                         "The outputs should be separated by newline character '\\n'.\n").format(
            num_gen)

        # chat_completion = client.chat.completions.create(
        #     messages=[
        #         {
        #             "role": "system",
        #             "content": Profile_new,
        #         }
        #     ],
        #     model="gpt-3.5-turbo",
        # )

        # results = chat_completion.choices[0].message.content

        results = asyncio.run(chat(Profile_new=Profile_new))

        scenarios = results.split('\n')
        scenarios = [check_scenario(sce) for sce in scenarios if sce != ""]

        if isinstance(scenarios, list):

            # hints = random.sample(scenarios, num_hints)
            # hints = np.random.choice(scenarios, size = num_hints, replace=False)
            if len(scenarios) == num_gen:
                hints = get_hints(tokens=tokens, scenarios=scenarios, num_hints=num_hints, hint_limit=hint_limit)
            else:
                print('generation number exception!')
                print('scenarios', scenarios)
                print('num_scenarios', len(scenarios))
                print('results', results)
                continue

        else:
            print('instance error, not a list!')
            continue

        example_hints = ' '.join(hints)
        generated_scenarios.extend(scenarios)

        # length = [tokens.gpt_get_estimated_cost(i, max_tokens=0) for i in generated_scenarios]
        # avg_length = sum(length)/80


    with open('./data/gpt_generated_scenarios.txt', 'w') as f:
        f.writelines('\n'.join(generated_scenarios))


    print("done!")