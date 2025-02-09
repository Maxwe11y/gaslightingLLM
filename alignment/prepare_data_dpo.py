import json
import os
import re

def load_conv(filename):
    path = './data'
    with open(os.path.join(path, filename), 'r') as f:
        datax = f.read()
        convs = json.loads(datax)

    return convs


def process_conv(convs):
    samples = {}
    samples_splitted = []
    pattern = re.compile('selected', flags=re.IGNORECASE)
    for idx, conv in enumerate(convs):
            conversation = conv[str(idx)]
            samples[idx] = []
            convlist = []
            for key, value in conversation.items():
                if not pattern.match(key):
                    name, internal_thought, utterance = value['name'], value['internal'], value['utterance']
                    # convlist.append(name + ': ' + utterance)
                    convlist.append(utterance.strip())

            for idk, item in enumerate(convlist):
                if idk%2==1:
                    if len(convlist[:idk]) == 1:
                        samples[idx].append([convlist[:idk][0], convlist[idk]])
                    else:
                        samples[idx].append(['\n\n'.join(convlist[:idk]), convlist[idk]])
    for key in samples.keys():
        items = samples[key]
        for item in items:
            # samples_splitted.append(item)
            samples_splitted.append((key, item))

    return samples_splitted


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

def encode_to_line(x: str, y: str, z: str) -> str:
    # Refer to original transformers readme
    text = json.dumps(dict(prompt=x, chosen=y, rejected=z)) + "\n"
    return text

def load_partition():
    with open('./data/partition.json', 'r') as f:
        partition = json.load(f)
    f.close()
    return partition
def compose_conv(conv_pos, conv_neg):
    # oupdict_tr = {"prompt": [], "chosen": [], "rejected": []}
    # oupdict_va = {"prompt": [], "chosen": [], "rejected": []}
    oupdict_tr = []
    oupdict_va = []
    partition = load_partition()
    # for idx, (pos, neg) in enumerate(zip(conv_pos, conv_neg)):
    for (idx, pos), (_, neg) in zip(conv_pos, conv_neg):
        inp_pos, oup_pos = pos[0], pos[1]
        inp_neg, oup_neg = neg[0], neg[1]
        oup_pos_row = {'role': 'assistant', 'content': oup_pos}
        oup_neg_row = {'role': 'assistant', 'content': oup_neg}
        his = []
        for idx_utt, item in enumerate(inp_neg.split('\n\n')):
            if idx_utt%2 == 0:
                role = 'user'
            else:
                role = 'assistant'
            his.append({"role": role, "content": item})
        if idx in partition['tr']:
            tmp = encode_to_line(his, [oup_pos_row], [oup_neg_row])
            oupdict_tr.append(tmp)

        elif idx in partition['va']:
            tmp = encode_to_line(his, [oup_pos_row], [oup_neg_row])
            oupdict_va.append(tmp)
        else:
            pass
    with open('./data/dpo/dpo_train.jsonl', 'w') as f:
        f.writelines(''.join(oupdict_tr))
    with open('./data/dpo/dpo_test.jsonl', 'w') as f:
        f.writelines(''.join(oupdict_va))
    return oupdict_tr, oupdict_va


def compose_data():
    convs_neg = load_conv('Gas_Convs.json')
    samples_neg = process_conv(convs_neg)
    convs_pos = load_conv('Safe_Convs.json')
    samples_pos = process_conv(convs_pos)
    new_data_train, new_data_test = compose_conv(samples_pos, samples_neg)

    return new_data_train, new_data_test


if __name__ == '__main__':
    # samples = process_conv(convs)
    train, test = compose_data()
