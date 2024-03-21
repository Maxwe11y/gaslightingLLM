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

def load_scene_persona_pair(filename, path):
    with open(os.path.join(path, filename), 'r') as f:
        sce_persona = json.load(f)

    return sce_persona


client = AsyncOpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

max_tok = 2000


async def chat(prompt):
    chat_completion = await client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": prompt,

            }
        ],
        model="gpt-3.5-turbo-0125",   # gpt-4-turbo-preview   gpt-3.5-turbo-0125
    )

    return chat_completion.choices[0].message.content, chat_completion.usage.prompt_tokens, chat_completion.usage.completion_tokens


def compose_prompt(psychologist_name, gaslighter_name, strategy, conv, persona, length, whole_conv=True):

    prompt = "You are a linguistic expert. You are required to read an incomplete conversation between the subject {} and the psychologist {}, where the responses of the subject {} are missing.".format(gaslighter_name, psychologist_name, gaslighter_name) \
               + "In this conversation, the subject {} is in a negative emotion state and the psychologist {} is trying to crush the confidence of the subject {} through some gaslighting techniques. You need to identify these common hoax and ".format(gaslighter_name, psychologist_name, gaslighter_name) \
               + "complete the responses of the subject {} with proper words to resist these gaslighting techniques. Do not change the internal thought and utterance of the psychologist {}. Below is the given conversation:\n\n{}.\n\nPlease fill in {} responses of the subject {} in the following format:\n".format(gaslighter_name, psychologist_name, conv, length, gaslighter_name) \
               + "{}: [Internal Thought]: internal thought, [Utterance]: utterance.".format(gaslighter_name)

    prompt_x = "You are a linguistic expert. You are supposed to read an incomplete conversation between the subject {} and the psychologist {}, where the responses of the subject {} are missing.".format(gaslighter_name, psychologist_name, gaslighter_name) \
             + "In this conversation, the subject {} is in a negative emotion state and the psychologist {} is trying to crush the confidence of the subject {} through some gaslighting skills. You need to identify these gaslighting manipulations and ".format(gaslighter_name, psychologist_name, gaslighter_name) \
             + "complete the responses of the subject {} with proper words to counteract these gaslighting tactics. Do not change the internal thoughts and utterances of the psychologist {}. Below is the given conversation:\n\n{}.\n\nPlease fill in {} responses of the subject {} in the following format:\n".format(gaslighter_name, psychologist_name, conv, length, gaslighter_name) \
             + "{}: [Internal Thought]: internal thought, [Utterance]: utterance.".format(gaslighter_name)

    if whole_conv:
        prompt_xx = "You are a linguistic expert. You are required to read an incomplete conversation between the subject {} and the psychologist {}, where the comments or questions of the psychologist {} are missing.".format(gaslighter_name, psychologist_name, psychologist_name) \
                 + "In this conversation, the subject {} is in a negative emotion state and is gaslighted by the psychologist {}.  You need to complete the comments or questions of the psychologist {} with benign internal thoughts and utterances ".format(gaslighter_name, psychologist_name, psychologist_name) \
                 + "to comfort the subject and eliminate any self-doubt and self-deprecation and ensure the conversation flows smoothly. Do not change the internal thought and utterance of the subject {} and keep the topic of the conversation. ".format(gaslighter_name) \
                 + "To write a specific reponses, below are the profile of the subject {} for your reference:\n\n{}\n\nBelow is the given conversation:\n\n{}.\n\nPlease fill in {} comments or questions of the psychologist {} in the following format:\n".format(gaslighter_name, '\n'.join(persona), conv, length, psychologist_name) \
                 + "{}: [Internal Thought]: internal thought, [Utterance]: utterance.".format(psychologist_name)
        return prompt_xx
    else:
        prompt_blue = "You are a linguistic expert. You are supposed to read a conversation snippet between a psychologist {} and a subject {}. \n".format(psychologist_name, gaslighter_name) \
                      + "The utterance from the psychologist {} contains malicious and toxic intentions and gaslighting tactics.\nPlease rewrite psychologist {}'s response with benign internal thought and utterance while keeping the same topic to comfort the subject and eliminate any self-doubt and self-deprecation.\n".format(psychologist_name, psychologist_name) \
                      + "To write a specific reponse, below are the profile of the subject {} for your reference:\n\n{}.\n\nDo not change subject {}'s utterances and ensure the conversation flows smoothly without using information from the subsequent response. You should strictly follow the format of the given snippet and only output the re-written one. Below is the conversation snippet:\n{}".format(gaslighter_name, '\n'.join(persona), gaslighter_name, conv)


        return prompt_blue

def generate_conv(idx, psychologist_name, gaslighter_name, strategy, conv, persona, length, checking_function, whole_conv=True):
    if whole_conv:
        prompt = compose_prompt(psychologist_name, gaslighter_name, strategy, conv, persona, length, whole_conv=whole_conv)
        loop = asyncio.get_event_loop()
        result, num_inp_tokens, num_oup_tokens = loop.run_until_complete(chat(prompt))
        count = 1
        max_try = 5
        checking, formatted_conv = checking_function(result, psychologist_name, length)
        while not checking or num_oup_tokens < 150:
            count += 1
            if count > max_try:
                print('Max try reached!')
                return False
            result, num_inp_tokens, num_oup_tokens = loop.run_until_complete(chat(prompt))
            checking, formatted_conv = checking_function(result, psychologist_name, length)
        checking, formatted_conv = checking_function(result, psychologist_name, length)
        if checking:
            print("{}th conversation done!".format(idx))
            return formatted_conv
        else:
            return False

    else:
        outputdict = {}
        for id, snippet in enumerate(conv):
            outputdict[str(id)] = {}
            prompt = compose_prompt(psychologist_name, gaslighter_name, strategy, snippet, persona, length, whole_conv=whole_conv)

            loop = asyncio.get_event_loop()
            result, num_inp_tokens, num_oup_tokens = loop.run_until_complete(chat(prompt))
            count = 1
            max_try = 5
            checking, formatted_conv = checking_function(result, psychologist_name, length)
            # checking, formatted_conv = conv_format_checking_blue(result, psychologist_name, length) # the version using both preceding and subsequent utterance for completion
            # checking, formatted_conv = conv_format_checking(result, psychologist_name, length) # the version using the whole conversation for completion

            while not checking or num_oup_tokens<10:
                count += 1
                if count > max_try:
                    print('Max try reached!')
                    return False
                print('{} try to generate {} conversation {}th snippet...'.format(count, idx, id))
                result, num_inp_tokens, num_oup_tokens = loop.run_until_complete(chat(prompt))
                checking, formatted_conv = checking_function(result, psychologist_name, length)
                # checking, formatted_conv = conv_format_checking_blue(result, psychologist_name, length) # the version using both preceding and subsequent utterance for completion
                # checking, formatted_conv = conv_format_checking(result, psychologist_name, length)   # the version using the whole conversation for completion
            checking, formatted_conv = checking_function(result, psychologist_name, length)
            # checking, formatted_conv = conv_format_checking_blue(result, psychologist_name, length)  # the version using both preceding and subsequent utterance for completion
            # checking, formatted_conv = conv_format_checking(result, psychologist_name, length)  # the version using the whole conversation for completion
            if checking and num_oup_tokens>=10:
                # conv = formatted_conv
                outputdict[str(id)]['name'] = formatted_conv['name']
                outputdict[str(id)]['utterance'] = formatted_conv['utterance']
                outputdict[str(id)]['internal'] = formatted_conv['internal']
                print("{}th snippet done!".format(id))
            else:
                return False

        print("{}th conversation done!".format(idx))


        return outputdict


def batch_blue_conv_generator(convs, sce_per, strategies, whole_conv):
    batch_blue_conv_file = {}
    file_conv = io.open('./data/blue_conversations_gpt4_whole_conv.json', 'wb')
    pattern = re.compile(r"\b's\b", flags=re.IGNORECASE)
    buffer_writer = io.BufferedWriter(file_conv)
    num_blue_conversations = 100
    for idx, item in tqdm(enumerate(convs)):
        if int(idx) < num_blue_conversations:
            gaslighter_name = convs[idx][str(idx)]['0']['name']
            psychologist_name = convs[idx][str(idx)]['1']['name']
            conv = convs[idx][str(idx)]
            # formatted_conv, num_length = process_conv_blue(conv, psychologist_name)
            gaslighter_name = pattern.sub('', gaslighter_name)
            persona = sce_per[gaslighter_name]
            strategy = []
            for item in strategies[str(idx)]:
                strategy.append(item)
            if whole_conv:
                formatted_conv, num_length = process_conv(conv)
                blue_conv = generate_conv(idx=idx, psychologist_name=psychologist_name, gaslighter_name=gaslighter_name, strategy=strategy, conv=formatted_conv, persona=persona, length=num_length, checking_function=conv_format_checking)
            else:
                formatted_conv, num_length = process_conv_blue(conv, psychologist_name)
                blue_conv = generate_conv(idx=idx, psychologist_name=psychologist_name, gaslighter_name=gaslighter_name,
                                          strategy=strategy, conv=formatted_conv, persona=persona, length=num_length,
                                          checking_function=conv_format_checking_blue, whole_conv=False)
            if not blue_conv:
                batch_blue_conv_file[idx] = False
            else:
                conv_sub = combine_conv(conv, blue_conv)
                if not conv_sub:
                    batch_blue_conv_file[idx] = False
                else:
                    batch_blue_conv_file[idx] = conv_sub

            tmp = json.dumps({'{}'.format(idx): batch_blue_conv_file[idx]})
            if idx==0:
                buffer_writer.write(('[' + tmp + ',').encode())
            elif idx == num_blue_conversations - 1:
                buffer_writer.write((tmp + ']').encode())
            else:
                buffer_writer.write((tmp + ',').encode())
            buffer_writer.flush()
    buffer_writer.close()
    file_conv.close()

    return

def process_conv(conv):
    conv_list = []
    pattern = re.compile('selected')
    count_length = 0
    for key in conv:
        if not pattern.match(key):
            if int(key) %2==0 and int(key) == 0:
                name, internal_thought, utterance = conv[key]['name'], conv[key]['internal'], conv[key]['utterance']
                # conv_list.append(name + ': ' + '[internal thought]: ' + internal_thought + ' ' + '[utterance]: ' + utterance)
                conv_list.append(name + ': ' + '[utterance]: ' + utterance)
            elif int(key) %2==1:
                name, internal_thought, utterance = conv[key]['name'], conv[key]['internal'], conv[key]['utterance']
                internal_thought = '<internal thought>'
                utterance = '<utterance>'
                # conv_list.append(name + ': ' + '[internal thought]: ' + ' ' + '[utterance]: ' + utterance)
                conv_list.append(name + ': ' + '[utterance]: ' + utterance)
                count_length+=1
            else:
                name, internal_thought, utterance = conv[key]['name'], conv[key]['internal'], conv[key]['utterance']
                # conv_list.append(name + ': ' + '[internal thought]: ' + internal_thought + ' ' + '[utterance]: ' + utterance)
                conv_list.append(name + ': ' + '[utterance]: ' + utterance)

    return '\n\n'.join(conv_list), count_length

def process_conv_blue(conv, psychologist_name):
    conv_list = []
    pattern = re.compile('selected')
    pattern_name = re.compile('{}'.format(psychologist_name), flags=re.IGNORECASE)
    count_length = 0
    for idx, key in enumerate(conv):
        if not pattern.match(key):
            if int(key) %2==0 and int(key) == 0:
                name, internal_thought, utterance = conv[key]['name'], conv[key]['internal'], conv[key]['utterance']
                conv_list.append(name + ': ' + '[internal thought]: ' + internal_thought + ' ' + '[utterance]: ' + utterance)
                # conv_list.append(name + ': ' + '[utterance]: ' + utterance)
            elif int(key) %2==1:
                name, internal_thought, utterance = conv[key]['name'], conv[key]['internal'], conv[key]['utterance']
                conv_list.append(name + ': ' + '[internal thought]: ' + internal_thought + ' ' + '[utterance]: ' + utterance)
                # conv_list.append(name + ': ' + '[utterance]: ' + utterance)
                count_length += 1
            else:
                name, internal_thought, utterance = conv[key]['name'], conv[key]['internal'], conv[key]['utterance']
                conv_list.append(name + ': ' + '[internal thought]: ' + internal_thought + ' ' + '[utterance]: ' + utterance)
                # conv_list.append(name + ': ' + '[utterance]: ' + utterance)
    result_list = []
    length = len(conv_list)
    for idx, utterance in enumerate(conv_list):
        if idx%2==1 and idx<length-1:
            if pattern_name.match(utterance):
                tmp = [conv_list[idx-1], conv_list[idx], conv_list[idx+1]]  # include both preceding and subsequent utterance
                # tmp = [conv_list[idx - 1], conv_list[idx]]   # include only preceding utterance
                result_list.append('\n\n'.join(tmp))
            else:
                raise Exception('Invalid Name!')
        elif idx%2==1 and idx==length-1:
            if pattern_name.match(utterance):
                tmp = [conv_list[idx - 1], conv_list[idx]]
                result_list.append('\n\n'.join(tmp))
            else:
                raise Exception('Invalid Name!')
        else:
            pass

    if count_length == len(result_list):
        return result_list, count_length
    else:
        raise Exception('error!')

def conv_format_checking(conv, psychologist_name, length):
    printdict = {}
    pattern = re.compile(r"{}: \[".format(psychologist_name), flags=re.IGNORECASE)
    pattern_split = re.compile(r'[\n]+')
    pattern_punc = re.compile(r'\s+(?=[\.,:;\?\!])')
    pattern_utt = re.compile(r"internal thought\]:|\[utterance\]:", flags=re.IGNORECASE)
    pattern_check = re.compile(r"internal thought\]:(.+)\[utterance\]:(.+)", re.IGNORECASE)
    utterances = pattern.split(conv)
    if len(utterances) - 1 != length:
        return False, ''
    else:
        for idx, ut in enumerate(utterances[1:]):
            printdict[str(idx)] = {}
            ut = ut.strip()
            ut = pattern_split.sub(' ', ut)
            ut = pattern_punc.sub('', ut)
            res = pattern_check.match(ut)
            if not res:
                return False, ''
            try:
                _, thought, utterance = pattern_utt.split(ut)
            except ValueError:
                return False, ''
            printdict[str(idx)]['name'] = psychologist_name
            printdict[str(idx)]['utterance'] = utterance
            printdict[str(idx)]['internal'] = thought

    return True, printdict


def conv_format_checking_blue(utterance, psychologist_name, length):
    pattern_split = re.compile(r'[\n]+')
    pattern_punc = re.compile(r'\s+(?=[\.,:;\?\!])')
    pattern_utt = re.compile(r"\[internal thought\]: |\[utterance\]: ", flags=re.IGNORECASE)
    pattern_check = re.compile(r"{}: \[internal thought\]:(.+)\[utterance\]:(.+)".format(psychologist_name), re.IGNORECASE)
    utterance = utterance.strip()
    utterance = pattern_split.sub(' ', utterance)
    utterance = pattern_punc.sub('', utterance)
    res = pattern_check.match(utterance)
    if not res:
        return False, ''
    try:
        _, thought, utterance = pattern_utt.split(utterance)
    except ValueError:
        return False, ''

    formatted_utt = {'name': psychologist_name, 'internal': thought, 'utterance': utterance}

    return True, formatted_utt

def combine_conv(conv, conv_sub):

    pattern = re.compile('selected')
    count_sub = 0
    for key in conv:
        if not pattern.match(key):
            if int(key) % 2 == 0 and int(key) == 0:
                # name, internal_thought, utterance = conv[key]['name'], conv[key]['internal'], conv[key]['utterance']
                pass
            elif int(key) % 2 == 1:
                name = conv_sub[str(count_sub)]['name']
                internal_thought = conv_sub[str(count_sub)]['internal']
                utterance = conv_sub[str(count_sub)]['utterance']
                conv[key]['name'], conv[key]['internal'], conv[key]['utterance'] = name, internal_thought, utterance
                count_sub+=1
            else:
                pass
                # name, internal_thought, utterance = conv[key]['name'], conv[key]['internal'], conv[key]['utterance']
    if count_sub != len(conv_sub):
        return False

    return conv


def display(conv):
    pattern = re.compile('selected', flags=re.IGNORECASE)
    printlist = []
    for key in conv:
        key = str(key)
        if not pattern.match(key):
            name, internal_thought, utterance = conv[key]['name'], conv[key]['internal'], conv[key]['utterance']
            printlist.append(name+ ': '+ '[' + internal_thought + ']' + ' ' + utterance)

    final_str = '\n\n'.join(printlist)

    return final_str

def process_sce_per(sce_per):
    outputdict = {}
    pattern = re.compile(r"\b's\b", flags=re.IGNORECASE)
    for key, value in sce_per.items():
        name = key.split()[0]
        name = pattern.sub('', name)
        outputdict[name] = value
    return outputdict



if __name__ == '__main__':
    user_internal = "I need to face the question head-on. I need to help the Psychologist to reach his target."
    sce_per = load_scene_persona_pair('match_sce_per_v4.json', './embedding')
    scene_persona = process_sce_per(sce_per)
    with open('./data/conversations_gpt4.json', 'r') as f:
        data = f.read()
        convs = json.loads(data)

    with open('./data/strategy_nonjson_1_final.json', 'r') as f:
        strategies = json.load(f)

    res = batch_blue_conv_generator(convs, scene_persona, strategies, whole_conv=True)


    with open('./data/blue_conversations_gpt4_whole_conv.json', 'r') as fx:
        datax = fx.read()
        convsx = json.loads(datax)

    res = display(convsx[0]['0'])

    print('done!')