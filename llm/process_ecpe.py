import numpy as np
import re
from googletrans import Translator
import openai
import tiktoken

def num_tokens_from_string(string, encoding_name):
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

openai.api_key = "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" # luyao
# openai.api_key = "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" # wei
max_tok = 1024

def translate_to_english(chinese_text):
    translator = Translator()
    translation = translator.translate(chinese_text, src='zh-cn', dest='en')
    return translation.text

# Example usage
# chinese_sentence = "你好，这是一个简单的例子。"
# english_translation = translate_to_english(chinese_sentence)
# print(f"Chinese: {chinese_sentence}")
# print(f"English: {english_translation}")

def get_chatgpt_response(guide, demonstration, guide_after, query):

    chat = [{"role": 'system', "content": guide + demonstration + guide_after + query}]

    reply = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=chat,
        max_tokens=max_tok
    )
    res = reply['choices'][0]['message']['content']

    return res


documents = {}
emo_cau = {}
emotions = {}
emotion_set = {}
documents_en = {}
pattern = r'\((.*?)\)'
response = {}
with open('../dataset/all_data_pair.txt','r', encoding='utf-8') as f:
    while True:
        line = f.readline()
        if not line:
            print("file finished!")
            break
        line = line.strip()
        if ',' not in line:
            id = int(line.split()[0])
            documents[id] = []
            documents_en[id] = []
            length = int(line.split()[-1])
            line = f.readline().strip()
            pairs = re.findall(pattern, line)
            emo_cau[id] = pairs
            emotions[id] = list(set([int(item.split(',')[0]) for item in emo_cau[id]]))
            emotion_set[id] = []
            for i in range(length):
                if i+1 in emotions[id]:
                    line = f.readline().strip().split(',')
                    if line[1] not in emotions[id]:
                        emotion_set[id].append(line[1])
                else:
                    line = f.readline().strip().split(',')
                if len(line) == 4:
                    documents[id].append(''.join(line[-1].split(' ')))
                else:
                    print('outlier')
            documents[id] = ', '.join(documents[id])
            documents_en[id] = translate_to_english(documents[id])

            # guide = 'The following news describes an event, or a story about people. Please give comments or suggestions on the people or the events.'
            guide = 'The following news is an event, or a report about people. Please recognize the emotion and causes of the people and give specific comments or suggestions to the people.'
            response[id] = get_chatgpt_response(guide, '', '', documents_en[id])
            print('done!')
        else:
            print('wrong line!')
            print(line)

f.close()

print("done!")