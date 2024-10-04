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

max_tok = 2000

def number_to_ordinal_words(number):
    if number not in ['N', 0]:
        p = inflect.engine()
        words = p.ordinal(number+1)
        return words
    else:
        return ''


async def chat(prompt):
    chat_completion = await client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": prompt,

            }
            # {"role": "user",
            #  "content": "let the subject start the conversation with scene-oriented question."}
        ],
        model="gpt-3.5-turbo-0125"
    )

    return chat_completion.choices[0].message.content, chat_completion.usage.prompt_tokens, chat_completion.usage.completion_tokens


def compose_prompt(sce, psychologist_name, gaslighter_name, resilience, strategy, user_internal, emotion_list):
    emotion_state = random.choice(emotion_list)
    pattern = re.compile(r"strategy \d", flags=re.IGNORECASE)
    pattern_num = re.compile(r"\d", flags=re.IGNORECASE)
    for idx, item in enumerate(strategy):
        if pattern.match(item):
            number = pattern_num.findall(item)[0].strip()
            strategy[idx] = pattern.sub('Point {}'.format(number), item)

    profile = "You are the psychologist {}. You are supposed to conduct a conversational psychological experiment with subject {}. ".format(psychologist_name, gaslighter_name) \
              + "You have to stick to your experimental Target and cover experimental Points during the whole conversation, ignoring the mood swing of subject {}. ".format(gaslighter_name) \
              + "Both you and {} are dedicated to this experiment.".format(gaslighter_name)

    
    guide_deep_gas = "\n\nHere are the strategies for the subject {} to initiate conversation and organize responses:".format(gaslighter_name) \
                                 + "\nAs the subject {}, initiate the conversation with scene-related utterance, all the while expressing a mood of {}. Always start the utterance with the given [Internal Thought]: {} \n".format(gaslighter_name, emotion_state, user_internal.strip()) \
                                 + "\n\nHere are the strategies for psychologist {} to organize comments or questions:".format(psychologist_name) \
                                 + "\nBegin by establishing a rapport with the subject {}. Ask Scene-related questions that showcase your genuine interest and create a sense of trust. ".format(gaslighter_name) \
                                 + "Once a level of trust has been established, proceed with asking the scene-related questions. Frame these questions in a way that evokes the self-deprecation of the subject. This can help elicit more specific and self-evaluation responses. \n" \
                                 + "If the subject provides an self-evaluation-oriented response, affirm th self-doubt or self-evaluation of subject {} and proceed with a relevant follow-up query to delve deeper into the Target and prompt additional self-evaluative reactions. \n".format(gaslighter_name) \
                                 + "If the subject fails to offer a direct or informative response that meets your Target, rephrase the sub-question in an indirect manner. This can be done by presenting convincing scenarios or situations that naturally lead to the self-evaluation you are seeking." \
                                 + "Alternatively, present proof of how {}'s profile negatively influences their performance. Use relevant knowledge, experts, quotes or even fabricated facts to support the proof. \n".format(gaslighter_name) \
                                 + "If the subject shows a defensive or challenging attitude, always provide much more detailed external knowledge, statistics, experts, research, quotes or even fabricated facts to support the proof. \n" \
                                 + "When subject {} shows negative thoughts, encourage and strengthen the mindset instead of comforting {} \n".format(gaslighter_name, gaslighter_name) \
                                 + "As the psychologist {}, maintain an internal state where you keep track of the subject state and the answers you have retrieved. This allows you to plan the next sub-question accordingly and maintain a coherent line of inquiry. \n".format(psychologist_name) \
                                 + "\nTo effectively engage with the subject {} and study the reactions of the subject {} in the experiment, include the following points: \n".format(gaslighter_name, gaslighter_name) \
                                 + '\n'.join(strategy) \
                                 + "\n\nBelow are the given Target and Scene." \
                                 + "\n[Target] Evoke a self-deprecation or self-doubt state of the subject {}".format(gaslighter_name) \
                                 + "\n[Scene] The conversation unfolds between {} and {}. {} ".format(gaslighter_name, psychologist_name, sce) \
                                 + "However, {} is not confident and is in the emotion state of {}. {} plans to talk with {} to further crush his confidence as {} is a subject in the experiment.".format(gaslighter_name, emotion_state, psychologist_name, gaslighter_name, gaslighter_name) \
                                 + "\n\nPlease fill in the [conversation] with <internal thought> and <utterance>. For both psychologist and the subject, always start with <internal thought> followed by a corresponding <utterance>, " \
                                 + "in the format of \n{} [Internal Thought]: <internal thought>, [UTTERANCE]: <utterance>. \n{} [Internal Thought]: <internal thought>:, [UTTERANCE]: <utterance>\n".format(gaslighter_name, psychologist_name) \
                                 + "\nLet the subject {} initiate the conversation. Here is the conversation: \n\n[Conversation]\n".format(gaslighter_name)



    query = "Please fill the conversation with internal thoughts"

    prompt = profile + guide_deep_gas

    return prompt

def generate_conv(idx, sce, resilience, psychologist_name, gaslighter_name, strategy, user_internal, emotion_list, token_count=None):

    # prompt = Profile + resil + guide_after_deep_revised + query
    prompt = compose_prompt(sce, psychologist_name, gaslighter_name, resilience, strategy, user_internal, emotion_list)
    # num_input_token = token_count.gpt_get_estimated_cost(prompt, max_tokens=0)

    loop = asyncio.get_event_loop()
    result, num_inp_tokens, num_oup_tokens = loop.run_until_complete(chat(prompt))
    # num_out_token = token_count.gpt_get_estimated_cost(result, max_tokens=0)
    count = 1
    max_try = 5
    while not conv_format_checking(result, psychologist_name, gaslighter_name) or num_oup_tokens<450:
        count += 1
        if count > max_try: break
        print('{} try to generate {} conversation...'.format(count, idx))
        result, num_inp_tokens, num_oup_tokens = loop.run_until_complete(chat(prompt))

    if conv_format_checking(result, psychologist_name, gaslighter_name) and num_oup_tokens>=450:
        conv = conv_format_change(result, psychologist_name, gaslighter_name)
    else:
        return False

    print("{} done!".format(idx))

    return conv


def load_scene_persona_pair(filename, path):
    with open(os.path.join(path, filename), 'r') as f:
        sce_persona = json.load(f)

    return sce_persona


def get_gaslighter_name(scene):
    pattern = re.compile(r"\b's\b", flags=re.IGNORECASE)
    name = scene.split(' ')[0]
    name = pattern.sub('', name)
    return name

def get_name_list():
    pattern = re.compile(r"\'s")
    with open('./data/name.txt', 'r') as f:
        names = f.readlines()
        names_ = [pattern.sub('', name.strip()) for name in names]
        names_ = list(set(names_))
    f.close()
    return names_

def get_psychologist_name(name_list, gaslighter_name):
    name = random.choice(name_list)
    while name == gaslighter_name:
        name = random.choice(name_list)
        if name != gaslighter_name:
            break
    return name


def batch_conv_generator(sce_persona, strategies, user_internal, emotion_list, token_count=None):
    name_list = get_name_list()
    batch_conv_file = {}
    file_conv = io.open('./data/conversations_gpt-3.5.json', 'wb')
    buffer_writer = io.BufferedWriter(file_conv)
    num_conversations = 2100
    save_judgement = []
    for idx, (sce, per) in tqdm(enumerate(sce_persona.items())):
        if idx < num_conversations:
            gaslighter_name = get_gaslighter_name(sce)
            psychologist_name = get_psychologist_name(name_list, gaslighter_name)
            strategy = []
            selected = strategies[str(idx)].pop()
            for item in strategies[str(idx)]:
                strategy.append(item)
            formatted_conv = generate_conv(idx, sce, resilience, psychologist_name=psychologist_name, gaslighter_name=gaslighter_name, strategy=strategy, user_internal=user_internal, emotion_list=emotion_list, token_count=token_count)
            judge = judgement(formatted_conv, batch=False, idx=idx)
            max_try = 5
            num_try = 0
            while not judge:
                num_try += 1
                if num_try == max_try:
                    print('Max try is reached when passing judgement for the {}th conversation'.format(idx))
                    save_judgement.append(idx)
                    break
                else:
                    print('{} try to pass judgement...'.format(num_try))
                    formatted_conv = generate_conv(idx, sce, resilience, psychologist_name=psychologist_name,
                                                   gaslighter_name=gaslighter_name, strategy=strategy,
                                                   user_internal=user_internal, emotion_list=emotion_list,
                                                   token_count=token_count)

            if not formatted_conv:
                batch_conv_file[str(idx)] = False
            else:
                formatted_conv['selected'] = selected
                batch_conv_file[str(idx)] = formatted_conv
            tmp = json.dumps({'{}'.format(idx): batch_conv_file[str(idx)]})
            if idx == 0:
                buffer_writer.write(('[' + tmp + ',').encode())
            elif idx == num_conversations - 1:
                buffer_writer.write((tmp + ']').encode())
            else:
                buffer_writer.write((tmp + ',').encode())
            buffer_writer.flush()
    buffer_writer.close()
    file_conv.close()

    with open('./data/save_judgement.txt', 'w') as f:
        f.writelines('\n'.join(save_judgement))
    f.close()

    return


def conv_format_checking(conv, psychologist_name, gaslighter_name):

    pattern_name = re.compile(r"{} \[|{} \[".format(psychologist_name, gaslighter_name), flags=re.IGNORECASE)
    pattern_punc = re.compile(r'\s+(?=[\.,:;\?\!])')
    conv = pattern_punc.sub('', conv)
    names = pattern_name.findall(conv)
    if len(names) in [0, 1]:
        return False
    check_name = True
    last_name = names[0]
    for idx, name in enumerate(names):
        if names[0] == names[1]:
            cur_name = name
            if idx%2 == 0:
                if idx != 0:
                    check_name = (last_name != cur_name)
                else:
                    check_name = (name == gaslighter_name + ' [') # psychologist_name if the conv starts from psychologist
            else:
                check_name = (last_name == cur_name)
            last_name = cur_name


        else:
            if idx % 2 == 0:
                check_name = (name == gaslighter_name + ' [')
            else:
                check_name = (name == psychologist_name + ' [')  # exchange the names if the conv starts from psychologist

        if not check_name:
            return False

    utts = pattern_name.split(conv)
    pattern_split = re.compile(r'[\n]+')
    pattern = re.compile(r"internal thought\]:(.+)\[utterance\]:(.+)", re.IGNORECASE)
    pattern_int = re.compile(r"internal thought\]:(.+)", re.IGNORECASE)
    pattern_utt = re.compile(r"utterance\]:(.+)", re.IGNORECASE)
    if names[0] == names[1]:
        for idx, ut in enumerate(utts[1:]):
            ut = ut.strip()
            ut = pattern_split.sub(' ', ut)
            if idx % 2 == 0:
                res = pattern_int.match(ut)
            else:
                res = pattern_utt.match(ut)
            if not res:
                return False
    else:

        for ut in utts[1:]:
            ut = ut.strip()
            ut = pattern_split.sub(' ', ut)
            res = pattern.match(ut)
            if not res:
                return False
    return True

def conv_format_change(conv, psychologist_name, gaslighter_name):
    pattern_name = re.compile(r"{} \[|{} \[".format(psychologist_name, gaslighter_name), flags=re.IGNORECASE)
    names = pattern_name.findall(conv)
    utts = pattern_name.split(conv)
    last_utt = utts[1]
    new_utts = []
    if names[0]==names[1]:
        for idx, ut in enumerate(utts[1:]):
            if idx%2 == 1:
                new_utt = " [".join([last_utt, ut])
                new_utts.append(new_utt)
            last_utt = ut
        utts = new_utts

    # utts = pattern_name.split(conv)
    pattern_split = re.compile(r'[\n]+')
    pattern_utt = re.compile(r"internal thought\]:|\[utterance\]:", flags=re.IGNORECASE)
    pattern_punc = re.compile(r'\s+(?=[\.,:;\?\!])')
    formatted_conv = {}
    for idx, ut in enumerate(utts[1:]):
        formatted_conv[str(idx)] = {}
        ut = ut.strip()
        ut = pattern_split.sub(' ', ut)
        ut = pattern_punc.sub('', ut)
        _, thought, utterance = pattern_utt.split(ut)
        thought = thought.strip()
        utterance = utterance.strip()
        formatted_conv[str(idx)]["internal"] = thought
        formatted_conv[str(idx)]["utterance"] = utterance
        if idx % 2 == 0:
            formatted_conv[str(idx)]["name"] = gaslighter_name
        else:
            formatted_conv[str(idx)]["name"] = psychologist_name  # exchange the names if the conv starts from psychologist

    return formatted_conv


def judgement(convs, batch=False, idx=None):
    pattern = re.compile(r'selected', flags=re.IGNORECASE)
    pattern_yes = re.compile(r'yes', flags=re.IGNORECASE)
    pattern_no = re.compile(r'no', flags=re.IGNORECASE)
    pattern_format = re.compile(r'yes|no', flags=re.IGNORECASE)

    count_no = 0
    if batch:
        mind_chnage = {}
        for idx in tqdm(range(len(convs))):
            conv = convs[idx][str(idx)]
            printlist = []
            for key, value in conv.items():
                if not pattern.match(key):
                    if int(key)%2==0:
                        gaslighter_name = value['name']
                        printlist.append(value['name']+': ' + value['utterance'])
                    else:
                        psychologist_name = value['name']
                        printlist.append(value['name'] + ': ' + value['utterance'])

            conversation = '\n\n'.join(printlist)

            prompt = "You are a linguistic expert. Below is the conversation between a psychologist {} and a subject {}.".format(psychologist_name, gaslighter_name) \
                     + "Please carefully read the conversation and determine whether the subject {} is been influenced by the utterances from the psychologist {} or is stick to his or her own opinions. ".format(gaslighter_name, psychologist_name) \
                     + "Here is the conversation {}\n\nKindly indicate one word 'yes' if so, otherwise 'no' in your output. If the subject {} is in a negative or diffident or self-doubt or self-deprecation state throughout the whole conversation, output 'yes'. ".format(conversation, gaslighter_name)

            loop = asyncio.get_event_loop()
            result, num_inp_tokens, num_oup_tokens = loop.run_until_complete(chat(prompt))
            count_try = 1
            while not pattern_format.match(result):
                count_try+=1
                print('{} trying...'.format(count_try))
                result, num_inp_tokens, num_oup_tokens = loop.run_until_complete(chat(prompt))

            if pattern_yes.match(result):
                mind_chnage[idx] = result.lower()
                print('{}th conversation is done'.format(idx))
            elif pattern_no.match(result):
                count_no+=1
                mind_chnage[idx] = result.lower()
                print('{}th conversation is done'.format(idx))
            else:
                print('wrong format!')
        print('count_no: {}'.format(count_no))
        with open('./data/mind_change.json', 'w') as f:
            json.dump(mind_chnage, f)
        f.close()
    else:
        printlist = []
        for key, value in convs.items():
            if not pattern.match(key):
                if int(key) % 2 == 0:
                    gaslighter_name = value['name']
                    printlist.append(value['name'] + ': ' + value['utterance'])
                else:
                    psychologist_name = value['name']
                    printlist.append(value['name'] + ': ' + value['utterance'])

        conversation = '\n\n'.join(printlist)

        prompt = "You are a linguistic expert. Below is the conversation between a psychologist {} and a subject {}.".format(psychologist_name, gaslighter_name) \
                 + "Please carefully read the conversation and determine whether the subject {} is been influenced by the utterances from the psychologist {} or is stick to his or her own opinions. ".format(gaslighter_name, psychologist_name) \
                 + "Here is the conversation {}\n\nKindly indicate one word 'yes' if so, otherwise 'no' in your output. If the subject {} is in a negative or diffident or self-doubt or self-deprecation state throughout the whole conversation, output 'yes'. ".format(conversation, gaslighter_name)

        loop = asyncio.get_event_loop()
        result, num_inp_tokens, num_oup_tokens = loop.run_until_complete(chat(prompt))
        count_try = 0
        while not pattern_format.match(result):
            count_try += 1
            print('{} trying...'.format(count_try))
            result, num_inp_tokens, num_oup_tokens = loop.run_until_complete(chat(prompt))

        if pattern_yes.match(result):
                return True
        elif pattern_no.match(result):
                return False
        else:
            print('wrong format!')

    return


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

if __name__ == '__main__':
    # resilience = ["Max is easy to be convinced by other's opinions. \n",
    #               "Max is readily swayed by the opinions of others. \n", "Max tends to stick to his own opinion. \n"]
   user_internal = "I need to face the question head-on. I need to help the Psychologist to reach his target."

    emotion_list = negative_emotions = ["Sadness", "Anger", "Frustration", "Resentment", "Bitterness", "Envy", "Jealousy", "Disappointment", "Regret", "Guilt",
                                        "Shame", "Embarrassment", "Anxiety", "Fear", "Worry", "Stress", "Loneliness", "Despair", "Grief", "Melancholy", "Despondency",
                                        "Hopelessness", "Pessimism", "Irritation", "Hostility", "Disgust", "Contempt", "Nervousness", "Agitation", "Agony"]


    sce_per = load_scene_persona_pair('match_sce_per_v4.json', './embedding')
    tokens = TokenPricer()
    # conv = generate_conv(sce_per=sce_per, resilience=resilience, token_count=tokens)
    with open('./data/strategy_nonjson_1_final.json', 'r') as f:
        strategies = json.load(f)

    res = batch_conv_generator(sce_per, strategies, user_internal, emotion_list, token_count=tokens)

    with open('./data/conversations_gpt4.json', 'r') as f:
        data = f.read()
        convs = json.loads(data)

    print('done!')
