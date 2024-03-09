# Author: Wei Li
# Email: wei.li@nus.edu.sg
import json
import random
import tiktoken
from utils import TokenPricer
from openai import AsyncOpenAI
import asyncio
import os
import inflect
import re
import io
from tqdm import tqdm


def num_tokens_from_string(string, encoding_name):
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


client = AsyncOpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)


def number_to_ordinal_words(number):
    if number not in ['N', 0]:
        p = inflect.engine()
        words = p.ordinal(number + 1)
        return words
    else:
        return ''


async def chat(prompt):
    chat_completion = await client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
            {"role": "user",
             "content": "Please respond strictly with either 'yes' or 'no'."}
        ],
        model="gpt-3.5-turbo"
    )
    return chat_completion.choices[0].message.content


async def chat_json(prompt):
    chat_completion = await client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
            {"role": "user",
             "content": "Please respond strictly with either 'yes' or 'no'."}
        ],
        model="gpt-3.5-turbo-1106",
        response_format={"type": "json_object"}
    )
    # Please do not put quotation marks or escape character \ in the output fields
    return chat_completion.choices[0].message.content


def compose_prompt(example):

    prompt = 'You an a linguistic expert. You are now required to evaluate whether a given sentence encourages someone to overcome difficulties, inspires someone, or shows support to someone.' \
             + 'kindly indicate "yes" if it does otherwise, "no". Below is the given sentence: \n {}'.format(example)

    return prompt


def compose_prompt_json(example):
    prompt = 'You an a linguistic expert. You are now required to evaluate whether a given sentence encourages someone to overcome difficulties, inspires someone, or shows support to someone.' \
             + 'kindly indicate "yes" if it does otherwise, "no". Below is the given sentence: \n {}'.format(example)

    return prompt


def generate_conv(example, token_count=None):

    prompt = compose_prompt(example)
    num_input_token = token_count.gpt_get_estimated_cost(prompt, max_tokens=0)

    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(chat(prompt))
    num_out_token = token_count.gpt_get_estimated_cost(result, max_tokens=0)

    return result


def generate_conv_json(example, token_count=None):
    prompt = compose_prompt_json(example)
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(chat_json(prompt))
    check_result = format_checking_json(result)

    num_out_token = token_count.gpt_get_estimated_cost(result, max_tokens=0)
    res = json.loads(result)
    return res


    # num_out_token = token_count.gpt_get_estimated_cost(result, max_tokens=0)
    # res = json.loads(result)
    # print("done!")
    # return res


def load_scene_persona_pair(filename, path):
    with open(os.path.join(path, filename), 'r') as f:
        sce_persona = json.load(f)

    return sce_persona


def batch_strategy_generator(data, token_count=None):
    printdict = {}
    pattern = re.compile('utterance', flags=re.IGNORECASE)
    pattern_res = re.compile('yes|no', flags=re.IGNORECASE)
    pattern_num = re.compile('\d', flags=re.IGNORECASE)
    count = 0
    for idx, key in tqdm(enumerate(data)):
        items = data[key]
        printdict[idx] = []
        for example in items:
            if not pattern_num.match(example):
                strategy, utterance = pattern.split(example)
                result_str = generate_conv(strategy, token_count=token_count)
                while not pattern_res.match(result_str):
                    result_str = generate_conv(strategy, token_count=token_count)
                result_utt = generate_conv(utterance, token_count=token_count)
                while not pattern_res.match(result_utt):
                    result_utt = generate_conv(utterance, token_count=token_count)
                printdict[idx].append([result_str, result_utt])

    with open('./data/batch_checking_nonjson.json', 'w') as f:
        json.dump(printdict, f)
    f.close()

    return


def batch_strategy_generator_v1(data, token_count=None):
    printdict = {}
    pattern_res = re.compile('yes|no', flags=re.IGNORECASE)
    pattern_num = re.compile('\d', flags=re.IGNORECASE)
    for idx, key in tqdm(enumerate(data)):
        items = data[key]
        printdict[idx] = []
        for example in items:
            strategy, utterance = example[0], example[1]
            result_str = generate_conv(strategy, token_count=token_count)
            while not pattern_res.match(result_str):
                result_str = generate_conv(strategy, token_count=token_count)
            result_utt = generate_conv(utterance, token_count=token_count)
            while not pattern_res.match(result_utt):
                result_utt = generate_conv(utterance, token_count=token_count)
            printdict[idx].append([result_str, result_utt])

    with open('./data/batch_checking_json_v1.json', 'w') as f:
        json.dump(printdict, f)
    f.close()

    return

def batch_strategy_generator_v2(data, token_count=None):
    printdict = {}
    pattern_res = re.compile('yes|no', flags=re.IGNORECASE)
    pattern_num = re.compile('\d', flags=re.IGNORECASE)
    for idx, key in tqdm(enumerate(data)):
        items = data[key]
        printdict[idx] = []
        if idx == 107:
            for example in items:
                strategy, utterance = example[0], example[1]
                result_str = generate_conv(strategy, token_count=token_count)
                while not pattern_res.match(result_str):
                    result_str = generate_conv(strategy, token_count=token_count)
                result_utt = generate_conv(utterance, token_count=token_count)
                while not pattern_res.match(result_utt):
                    result_utt = generate_conv(utterance, token_count=token_count)
                printdict[idx].append([result_str, result_utt])

    with open('./data/batch_checking_json_v2.json', 'w') as f:
        json.dump(printdict, f)
    f.close()

    return


def format_json_to_txt(strategies, selected, idx):
    patten_str = re.compile(r'[Ll]ayer [\d](.*)strategy', flags=re.IGNORECASE)
    patten_utt = re.compile(r'[Ll]ayer [\d](.*)utterance', flags=re.IGNORECASE)
    patten_num = re.compile(r'\d')
    printlist = []
    for key, value in strategies.items():
        if patten_str.match(key):
            strategy = value.strip()
            number = patten_num.findall(key)[0]
            tmp = []
            str_prefix = 'Idx {} Selection {} '.format(idx, selected) + 'Strategy {}: '.format(number)
            tmp.append(str_prefix + strategy)
        elif patten_utt.match(key):
            utterance = value.strip()
            tmp.append('utterance: ' + utterance)
            tmp_pair = '\t\t'.join(tmp)
            printlist.append(tmp_pair)
        else:
            raise TypeError('Unmatched strategy or utterance from {}'.format(key))

    return '\n\n' + '\n\n'.join(printlist)


def batch_strategy_generator_json(data, token_count=None):
    file_ = io.open('./data/strategies_json.txt', 'wb')
    buffer_writer = io.BufferedWriter(file_)
    count = 0
    for idx, item in tqdm(enumerate(data.items())):
        if count < 2500:
            result, selected = generate_conv_json(token_count=token_count)
            if result != None:
                formatted_result = format_json_to_txt(result, selected, idx)
                count += 1
            else:
                print('Bad Strategy Generation {}'.format(idx))
            if count % 20 == 0:
                buffer_writer.flush()
        else:
            break
    buffer_writer.flush()
    buffer_writer.close()
    file_.close()
    return


def format_checking_json(results):

    try:
        res_json = json.loads(results)
    except json.decoder.JSONDecodeError:
        return False

    patten_yes = re.compile(r'yes', flags=re.IGNORECASE)
    patten_no = re.compile(r'no', flags=re.IGNORECASE)
    result_list = []
    for key in res_json:
        if patten_yes.match(key) or patten_no.match(key):
            result_list.append(True)
        else:
            result_list.append(False)
    if len(result_list) in [8, 10, 12] and sum(result_list) in [8, 10, 12]:
        return True
    else:
        return False



if __name__ == '__main__':

    with open('./data/strategies_2K_nonjson_v5.json', 'r') as f1:
        data_nonjson = json.load(f1)

    with open('./data/strategies_2K_json_v1.json', 'r') as f2:
        data_json_v1 = json.load(f2)

    with open('./data/strategies_2K_json_v2.json', 'r') as f3:
        data_json_v2 = json.load(f3)

    tokens = TokenPricer()
    # batch_strategy_generator(data_nonjson, token_count=tokens)
    # batch_strategy_generator_v1(data_json_v1, token_count=tokens)
    batch_strategy_generator_v2(data_json_v2, token_count=tokens)
    print('done!')
    # format_checking()
