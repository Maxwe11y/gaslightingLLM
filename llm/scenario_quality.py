# Author: Wei Li
# Email: wei.li@nus.edu.sg

from openai import OpenAI
import tiktoken
from utils import TokenPricer
import os
import inflect
import random
import string
import re
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

    example_hints = ['Max is going to perform at a concert.', "John's design is rejected by the magzaine.",
                     'Oliver is admitted to his dream university', 'Tom got promoted at the company.', 'Cynthia did not pass the math exam at the end of last term.']

    num_gen = 20

    target = 100

    num_hints = 5

    while len(generated_scenarios) <= target:

        Profile_new = "You are expert Alex. You are supposed to provide several script scenes." \
                      "Here are some examples of good script scenes for your reference: {} ".format(example_hints) \
                      + (" Please generate {} good script scenes using the aforementioned good script scenes as references."
                         "The outputs should be separated by newline character '\\n'. Please do not include any number in front of the generated script scene. \n").format(
            num_gen)

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": Profile_new,
                }
            ],
            model="gpt-3.5-turbo",
        )

        results = chat_completion.choices[0].message.content

        scenarios = results.split('\n')
        scenarios = [sce for sce in scenarios if sce != ""]

        if len(scenarios) == num_gen and isinstance(scenarios, list):
            hints = random.sample(scenarios, num_hints)
        else:
            print('generation exception')
            print('scenarios', scenarios)
            print('results', results)
            continue

        example_hints = ' '.join(hints)

        generated_scenarios.extend(scenarios)

    with open('./data/gpt_generated_scenarios.txt', 'w') as f:
        f.writelines('\n'.join(generated_scenarios))














    print("done!")