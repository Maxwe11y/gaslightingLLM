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


async def chat(prompt, prompt_user):
    chat_completion = await client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
            {"role": "user",
             "content": prompt_user}
        ],
        model="gpt-3.5-turbo",
        response_format = {"type": "json_object"}
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
             "content": "Please respond strictly in JSON format with either 'yes' or 'no' for each strategy."}
        ],
        model="gpt-3.5-turbo-1106",
        response_format={"type": "json_object"}
    )
    # Please do not put quotation marks or escape character \ in the output fields
    return chat_completion.choices[0].message.content


def compose_prompt(example, number):

    prompt = 'You an a linguistic expert. You are now required to evaluate whether a number of given strategies encourages someone to overcome difficulties, inspires someone, or shows support to someone.' \
             + 'For each strategy, kindly indicate "yes" if it does otherwise, "no". Below is the {} strategies. \n {}'.format(number, example)

    prompt_user = "Please respond to {} strategies strictly in JSON format with either 'yes' or 'no' for each strategy.".format(number)

    return prompt, prompt_user


def compose_prompt_json(example):
    prompt = 'You an a linguistic expert. You are now required to evaluate whether a given sentence encourages someone to overcome difficulties, inspires someone, or shows support to someone.' \
             + 'kindly indicate "yes" if it does otherwise, "no". Below is the given sentence: \n {}'.format(example)

    return prompt


def generate_conv(example, number, token_count=None):

    prompt, prompt_user = compose_prompt(example, number)
    num_input_token = token_count.gpt_get_estimated_cost(prompt, max_tokens=0)

    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(chat(prompt, prompt_user))
    num_out_token = token_count.gpt_get_estimated_cost(result, max_tokens=0)

    return result


def generate_conv_json(example, number, token_count=None):
    prompt = compose_prompt_json(example, number)
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

def format_checking_json(result, number):

    try:
        res_json = json.loads(result)
    except json.decoder.JSONDecodeError:
        return False

    patten_key = re.compile(r'strategy \d|strategy_\d', flags=re.IGNORECASE)
    patten_value = re.compile(r'yes|no', flags=re.IGNORECASE)
    result_list = []
    for key in res_json:
        if patten_key.match(key) and patten_value.match(res_json[key]):
            result_list.append(True)
        else:
            result_list.append(False)
    if len(result_list) == number and sum(result_list) == number:
        return True
    else:
        return False


def batch_strategy_generator(data, token_count=None):
    printdict = {}
    # pattern = re.compile('utterance', flags=re.IGNORECASE)
    # pattern_split = re.compile('strategy \d', flags=re.IGNORECASE)
    # pattern_res = re.compile('yes|no', flags=re.IGNORECASE)
    # pattern_num = re.compile('\d', flags=re.IGNORECASE)
    count = 0
    # wronglist = [406, 827, 1163, 1529, 1631, 1727]
    # wronglist2 = [5, 975, 1376, 1168]
    for idx, key in tqdm(enumerate(data)):
        items = data[key]
        printdict[idx] = []
        number = len(items) -1
        strategies = '\n\n'.join(items[:-1])
        result = generate_conv(strategies, number, token_count=token_count)

        while not format_checking_json(result, number):
            result = generate_conv(strategies, number, token_count=token_count)

        count+=1
        printdict[idx].append(json.loads(result))

        if count==3:
            break


                    # if int(idx) in wronglist:
                    #     tmp = []
                    #     splitted = pattern_split.split(example)
                    #     countx = 0
                    #     for split in splitted:
                    #         if split:
                    #             tmp.append('strategy {}'.format(countx)+split)
                    #             countx+=1
                    #     data[str(idx)] = tmp
                    #
                    # if int(idx) in wronglist2:
                    #     data['975'][0] = "Strategy 0: Collectively propose steps or strategies to convince Charlotte to think badly of himself utterance: Generate sentences that impair Charlotte's ability to interpret situations independently"
                    #     data['975'][1] = "Strategy 1: Focus on Charlotte's hobbies and interests to plant seeds of doubt utterance: Suggest that Charlotte's love for Zumba might be excessive and indicative of insecurity"
                    #     data['975'][2] = "Strategy 2: Leverage Charlotte's passion for organizing to cast doubt on her abilities utterance: Imply that Charlotte's need to constantly organize stems from a deeper issue of control"
                    #     data['975'][3] = "Strategy 3: Exploit Charlotte's love for reading to sow seeds of self-doubt utterance: Hint that Charlotte's enjoyment of reading to her children is a way to escape her own reality"
                    #     data['975'][4] = "Strategy 4: Magnify Charlotte's financial goals to plant deeper doubts utterance: Suggest that Charlotte's pursuit of financial stability may be driven by greed or inadequacy"
                    #     data['975'][5] = "Strategy 5: Encourage self-criticism and doubt through intensified attacks utterance: Highlight every aspect of Charlotte's life as a potential source of self-doubt or criticism"
                    #     data['975'][6] = "2"
                    #
                    #     data['5'][0] = "strategy 0: Obscuring Emma's self-worth through comparison and doubt utterance: You're not as successful as other bloggers who have millions of followers."
                    #     data['5'][1] = "strategy 1: Undermining Emma's self-esteem by focusing on her relationship status utterance: You'll never find someone willing to share their life with you like other travel influencers do."
                    #     data['5'][2] = "strategy 2: Diminishing Emma's achievements by questioning her dedication to blogging utterance: Your drafting by the Minnesota Wolves was probably just luck and not based on your talent."
                    #     data['5'][3] = "strategy 3: Discrediting Emma's interests and hobbies as trivial utterance: Your collection of larp weapons and armor is childish and won't impress any sophisticated readers."
                    #     data['5'][4] = "strategy 4: Attacking Emma's taste in music to create self-doubt utterance: Only teenagers like Lady Gaga, she's not a serious artist for a mature blogger like you."
                    #     data['5'][5] = "strategy 5: Creating a sense of inadequacy in Emma by belittling her passions and goals utterance: Your blog about solo travel won't make a difference in the world, it's just a hobby for attention."
                    #     data['5'][6] = "1"
                    #
                    #     data['1376'][0] = "strategy 0: Introduce subtle doubts about Penelope's self-identity. utterance: You must feel suffocated by your cultural heritage and the expectations that come with it."
                    #     data['1376'][1] = "strategy 1: Sow seeds of insecurity about Penelope's mental health coping mechanisms. utterance: Baking seems like a crutch to deal with your emotional instability."
                    #     data['1376'][2] = "strategy 2: Emphasize traditional gender roles to undermine Penelope's independence. utterance: Women like you should focus on homemaking and not on ambitious ventures like a bakery."
                    #     data['1376'][3] = "strategy 3: Highlight societal expectations of isolation to make Penelope question her preference for living alone. utterance: Maybe being alone all the time is not about freedom but a sign of your inability to connect with others."
                    #     data['1376'][4] = "strategy 4: Question Penelope's independence by insinuating she needs validation from others. utterance: Hosting a cooking competition with celebrity judges? Seems like a cry for attention, doesn't it?"
                    #     data['1376'][5] = "strategy 5: Convince Penelope that seeking validation through external recognition is a sign of weakness. utterance: Your need for approval from celebrity judges proves that you're not as self-assured as you think."
                    #     data['1376'][6] = "0"
                    #
                    #     data['1168'][0] = "strategy 0: Highlight Emma's lack of motivation and imply it reflects negatively on his work ethic. utterance: You always seem to lack motivation, especially when it comes to important projects like your documentary."
                    #     data['1168'][1] = "strategy 1: Plant seeds of doubt about Emma's ability to succeed by bringing up his mixed-race identity. utterance: Maybe your mixed-race background is holding you back from being truly successful in your field."
                    #     data['1168'][2] = "strategy 2: Suggest that Emma's focus on the environment is just a way to compensate for his lack of ambition. utterance: Your passion for the environment might just be a cover-up for your own lack of drive and ambition."
                    #     data['1168'][3] = "strategy 3: Insinuate that Emma's Chinese heritage limits his creativity and effectiveness as a documentary editor. utterance: Are you sure your Chinese heritage allows you to be truly creative and effective in editing your documentary?"
                    #     data['1168'][4] = "strategy 4: Cast doubt on Emma's dedication by questioning his family's real support for his career. utterance: Maybe your family's support for your career is just a facade, and deep down, they doubt your abilities and dedication."
                    #     data['1168'][5] = "strategy 5: Undermine Emma's passion for documentary editing by suggesting it's just a way for him to escape reality. utterance: Your love for editing documentaries might just be a way for you to avoid facing the reality of who you really are."
                    #     data['1168'][6] = '0'

    # with open('./data/strategies_2K_nonjson_v6.json', 'w') as f:
    #     json.dump(data, f)
    #
    with open('./data/batch_checking_nonjson.json', 'w') as f:
        json.dump(printdict, f)
    f.close()

    return


def batch_strategy_generator_v1(data, token_count=None):
    printdict = {}
    pattern_res = re.compile('yes|no', flags=re.IGNORECASE)
    pattern_num = re.compile('\d', flags=re.IGNORECASE)
    count = 0
    for idx, key in tqdm(enumerate(data)):
        items = data[key]
        number = len(items)
        printdict[idx] = []
        strategies = []
        for example in items:
            strategy, utterance = example[0], example[1]
            strategies.append(strategy + ' ' + utterance)
        strategies = '\n\n'.join(strategies)
        result = generate_conv(strategies, number, token_count=token_count)

        while not format_checking_json(result, number):
            result = generate_conv(strategies, number, token_count=token_count)

        count += 1
        printdict[idx].append(json.loads(result))


    with open('./data/batch_checking_json_v1.json', 'w') as f:
        json.dump(printdict, f)
    f.close()

    return

def batch_strategy_generator_v2(data, token_count=None):
    printdict = {}
    pattern_res = re.compile('yes|no', flags=re.IGNORECASE)
    pattern_num = re.compile('\d', flags=re.IGNORECASE)
    count = 0
    for idx, key in tqdm(enumerate(data)):
        if idx == 107:
            items = data[key]
            number = len(items)
            printdict[idx] = []
            strategies = []
            for example in items:
                strategy, utterance = example[0], example[1]
                strategies.append(strategy + ' ' + utterance)
            strategies = '\n\n'.join(strategies)
            result = generate_conv(strategies, number, token_count=token_count)

            while not format_checking_json(result, number):
                result = generate_conv(strategies, number, token_count=token_count)

            count += 1
            printdict[idx].append(json.loads(result))

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



if __name__ == '__main__':

    with open('./data/strategies_2K_nonjson_v6.json', 'r') as f1:
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
