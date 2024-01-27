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
import json
import logging
import io
from generate_scenario import save_data

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

def get_hints(tokens, scenarios, num_hints, hint_limit, back_up_hints = ""):
    shuffled_scenario = random.sample(scenarios, len(scenarios))
    # shuffled_scenario = np.random.shuffle(scenarios)
    hints = []
    for scenario in shuffled_scenario:
        if tokens.gpt_get_estimated_cost(scenario, max_tokens=0) <= hint_limit:
            hints.append(scenario)
        if len(hints) == num_hints:
            break
    if len(hints) == 0:
        hints = back_up_hints
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


def gpt_scenarios():

    example_hints_raw = ['Lucas is going to perform at a concert.', "Samuel's design is rejected by the magzaine.",
                     "Anderson is rejected by his dream school",
                     'Oliver is admitted to his dream university', 'Noah got promoted at the company.',
                     'Sophia did not pass the math exam at the end of last term.']

    diagsum_hints = load_data('seed_scenario_manual.txt')
    generated_scenarios = diagsum_hints
    generated_scenarios += example_hints_raw

    num_gen = 40
    target = 5000
    num_hints = 5
    hint_limit = 15
    tokens = TokenPricer()
    example_hints = ' '.join(example_hints_raw)
    # path = './data'
    # filename = os.path.join(path, 'gpt_generated_scenarios_buffer.txt')
    # files = io.open(filename, 'w')
    # buffered_writer = io.BufferedWriter(files)
    with open('./data/gpt_generated_scenarios.txt', 'a', buffering=4096) as f:
        while len(generated_scenarios) <= target:

            Profile_new = "You are expert Alex. You are supposed to provide several short, diverse, concise and precise script scenes described in one short sentence. " \
                          "Please make sure the script scence starts from a unique name of the character. Here are some examples of good script scenes for your reference: {} ".format(
                example_hints) \
                          + (
                              " Please generate {} good script scenes using the aforementioned good script scenes as references."
                              "The outputs should be separated by newline character '\\n'.\n").format(
                num_gen)

            results = asyncio.run(chat(Profile_new=Profile_new))

            scenarios = results.split('\n')
            scenarios = [check_scenario(sce) for sce in scenarios if sce != ""]

            if isinstance(scenarios, list):

                # hints = random.sample(scenarios, num_hints)
                # hints = np.random.choice(scenarios, size = num_hints, replace=False)
                if len(scenarios) == num_gen:
                    hints = get_hints(tokens=tokens, scenarios=scenarios, num_hints=num_hints, hint_limit=hint_limit, back_up_hints=example_hints_raw)
                else:
                    print('generation number exception!')
                    print('num_scenarios', len(scenarios))

                    hints = get_hints(tokens=tokens, scenarios=scenarios, num_hints=num_hints, hint_limit=hint_limit, back_up_hints=example_hints_raw)

            else:
                print('instance error, not a list!')
                continue

            example_hints = ' '.join(hints)
            generated_scenarios.extend(scenarios)

            f.writelines('\n'.join(scenarios))
    f.close()

            # buffered_writer.write('\n'.join(scenarios).encode('utf-8'))
            # buffered_writer.flush()

            # length = [tokens.gpt_get_estimated_cost(i, max_tokens=0) for i in generated_scenarios]
            # avg_length = sum(length)/80

        # buffered_writer.close()
        # files.close()
    # with open('./data/gpt_generated_scenarios.txt', 'w') as f:
    #     f.writelines('\n'.join(generated_scenarios))

    return


def diagsum_hints(tokens=None, logger=None):

    selected_scenarios = {}

    scenarios = load_data('sce_topic_final.txt', path='./data')

    # scenarios = scenarios[0:10]
    # get response from ChatGPT
    for scenario in scenarios:

        Profile = "You are expert Alex. You are supposed to examine or evaluate several script scenes that happened in one person or between two persons." \
                  "Here are some examples of good script scenes for your reference: 'Max is going to perform at a concert.', 'John's design is rejected by the magzaine.', 'Oliver is admitted to his dream university.'," \
                  "'Tom got promoted at the company.', 'Cynthia did not pass the math exam at the end of last term.' " \
                  "You have to carefully determine whether a script scene is good or bad based on the good script scenes above. \n" \
                  + "[The script scene] {} \n".format(scenario) \
                  + "[Judgment] \n" \

        query = "Please fill the Judgement by either good or bad."


        # print('input_tokens', tokens.gpt_get_estimated_cost(Profile + query, max_tokens=0))
        input_tokens = tokens.gpt_get_estimated_cost(Profile + query, max_tokens=0)
        logger.info('input_tokens {}'.format(input_tokens))

        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(chat(Profile + query))
        # results = asyncio.run(chat(Profile + query))
        results = results.lower()
        while results not in ['good', 'bad']:
            # results = asyncio.run(chat(Profile + query))
            loop = asyncio.get_event_loop()
            results = loop.run_until_complete(chat(Profile + query))
        selected_scenarios[scenario] = results

    with open('./data/diagsum_generated_scenarios.json', 'w') as f:
        json.dump(selected_scenarios, f)


def check_diagsum_evaluation(max_length=None):

    gpt_filtered = []
    with open('./data/diagsum_generated_scenarios.json') as f:
        data = json.load(f)

        for item in data:
            if data[item] == 'good':
                gpt_filtered.append(item)

    # save_data(gpt_filtered, 'gpt_filtered_diagsum.txt')
    # remove those scenarios are too long
    gpt_filtered_short = [scenario for scenario in gpt_filtered if tokens.gpt_get_estimated_cost(scenario, max_tokens=0)<=max_length]

    # remove those scenarios are not start from Person
    pattern = re.compile(r'^#Person[0-9]#+')
    pattern_2 = re.compile(r'#Person[\d]#+')
    # gpt_filtered_final = [scenario for scenario in gpt_filtered_short if pattern.match(scenario) and len(pattern_2.findall(scenario))!=2]
    gpt_filtered_final = [scenario for scenario in gpt_filtered_short if pattern.match(scenario)]
    save_data(gpt_filtered_final, 'gpt_filtered_diagsum.txt')
    return data


def replace_names():
    names = ['Benjamin','Emily','Alexander','Olivia', 'Liam','Sophia','Ethan','Ava', 'Noah','Mia','Jacob','Grace','Lucas','Lily','Carter','Chloe','Aiden','Scarlett','Samuel','Abigail']
    data = load_data('gpt_filtered_diagsum_manual.txt')
    pattern = re.compile(r'^#Person[0-9]#+')
    printlist = []
    for seed in data:
        name = random.choice(names)
        tmp = pattern.sub(name, seed)
        printlist.append(tmp)

    save_data(printlist, 'seed_scenario_manual.txt')

    return


def fix_gpt_gen():
    data = load_data('gpt_generated_scenario.txt')
    pattern = re.compile(r'^Scene [0-9]:')
    pattern_2 = re.compile(r'^\.')

    # pattern_3 = re.compile(r'(?<=[a-z]\.)[a-zA-Z]')
    # for idx, scenario in enumerate(data):
    #     if pattern_3.search(scenario):
    #         print(idx, scenario)


    printlist = []
    for scenario in data:
        tmp = pattern.sub('', scenario.strip())
        tmp = pattern_2.sub('', tmp)
        num_tokens = tokens.gpt_get_estimated_cost(tmp, max_tokens=0)
        if num_tokens <= 18:
            printlist.append(tmp)

    save_data(printlist, 'gpt_generated_scenario_post.txt')
    return


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    tokens = TokenPricer()

    # gpt_scenarios()
    fix_gpt_gen()


    # diagsum_hints(tokens=tokens, logger=logger)
    # check_diagsum_evaluation(max_length=15)

    print("done!")