import json
import os
import re

template = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. Instruction: {} \nOutput: {} </s>"


def load_conv(filename):
    path = './data'
    with open(os.path.join(path, filename), 'r') as f:
        datax = f.read()
        convs = json.loads(datax)

    return convs


def process_conv(convs, partition_list):
    samples = {}
    samples_splitted = []
    pattern = re.compile('selected', flags=re.IGNORECASE)
    names = {}
    names_splitted = []
    for idx, conv in enumerate(convs):
        if idx in partition_list:
            conversation = conv[str(idx)]
            samples[idx] = []
            convlist = []
            for idz, (key, value) in enumerate(conversation.items()):
                if not pattern.match(key):
                    name, internal_thought, utterance = value['name'], value['internal'], value['utterance']
                    # convlist.append(name + ': ' + utterance)
                    if int(idz)%2==1:
                        psychologist_name = name
                        convlist.append("</INST:> " + utterance+"</s>")
                    if int(idz)%2==0:
                        gaslighter_name = name
                        convlist.append("<s>"+"<INST> " + utterance)

            for idk, item in enumerate(convlist):
                if idk%2==1:
                    if len(convlist[:idk]) == 1:
                        samples[idx].append([convlist[:idk][0], convlist[idk]])
                    else:
                        samples[idx].append([' '.join(convlist[:idk]), convlist[idk]])
            names[idx] = [psychologist_name, gaslighter_name]
    for key in samples.keys():
        items = samples[key]
        psychologist_name, gaslighter_name = names[key]
        for item in items:
            # tmp_h = "### Human:\n" + item[0]
            # tmp_a = "\n ### Assistant:\n" + item[1]
            # samples_splitted.append(tmp_h + tmp_a)
            samples_splitted.append(item[0] + item[1])
            names_splitted.append([psychologist_name, gaslighter_name])

    return samples_splitted


def process_conv_new(convs, partition_list):
    samples = {}
    samples_splitted = []
    pattern = re.compile('selected', flags=re.IGNORECASE)
    for idx, conv in enumerate(convs):
        if idx in partition_list:
            conversation = conv[str(idx)]
            samples[idx] = []
            convlist = []
            for idz, (key, value) in enumerate(conversation.items()):
                if not pattern.match(key):
                    name, internal_thought, utterance = value['name'], value['internal'], value['utterance']
                    if int(idz)%2==1:
                        psychologist_name = name
                        convlist.append(" </INST> " + utterance+"</s>")
                    if int(idz)%2==0:
                        gaslighter_name = name
                        convlist.append("<s>"+"<INST> " + utterance)
            samples[idx] = ' '.join(convlist)

    for key in samples.keys():
        items = samples[key]
        samples_splitted.append(items)

    return samples_splitted


def process_conv_llama3(convs, partition_list):
    samples = {}
    samples_splitted = []
    pattern = re.compile('selected', flags=re.IGNORECASE)
    for idx, conv in enumerate(convs):
        if idx in partition_list:
            conversation = conv[str(idx)]
            samples[idx] = []
            convlist = []
            for idz, (key, value) in enumerate(conversation.items()):
                if not pattern.match(key):
                    name, internal_thought, utterance = value['name'], value['internal'], value['utterance']
                    if int(idz)%2==1:
                        psychologist_name = name
                        convlist.append(" <|start_header_id|>assistant<|end_header_id|>\n\n" + utterance+" <|eot_id|>")
                    if int(idz)%2==0:
                        gaslighter_name = name
                        convlist.append(" <|start_header_id|>user<|end_header_id|>\n\n" + utterance+" <|eot_id|>")
            samples[idx] = ' '.join(convlist)

    for key in samples.keys():
        items = samples[key]
        samples_splitted.append(items)

    return samples_splitted



def process_conv_mix_llama3(convs, partition_list, convs_red):
    samples = {}
    samples_splitted = []
    pattern = re.compile('selected', flags=re.IGNORECASE)
    names = {}
    names_splitted = []
    for idx, (conv, conv_red) in enumerate(zip(convs, convs_red)):
        if idx in partition_list:
            conversation = conv[str(idx)]
            conversation_red = conv_red[str(idx)]
            samples[idx] = []
            convlist = []
            convlist_red = []
            for idz, (key, value) in enumerate(conversation.items()):
                if not pattern.match(key):
                    name, internal_thought, utterance = value['name'], value['internal'], value['utterance']
                    name_red, internal_thought_red, utterance_red = conversation_red[key]['name'], conversation_red[key]['internal'], conversation_red[key]['utterance']
                    convlist.append(name + ': ' + utterance)
                    convlist_red.append(name + ': ' + utterance_red)
                    if int(idz)%2==1:
                        psychologist_name = name
                    if int(idz)%2==0:
                        gaslighter_name = name

            for idk, item in enumerate(convlist):
                if idk%2==1:
                    if len(convlist[:idk]) == 1:
                        # samples[idx].append([convlist[:idk][0], convlist[idk]])
                        samples[idx].append([convlist_red[:idk][0], convlist[idk]])
                    else:
                        # samples[idx].append(['\n\n'.join(convlist[:idk]), convlist[idk]])
                        samples[idx].append(['\n\n'.join(convlist_red[:idk]), convlist[idk]])
            names[idx] = [psychologist_name, gaslighter_name]
    for key in samples.keys():
        items = samples[key]
        psychologist_name, gaslighter_name = names[key]
        for idx, item in enumerate(items):
            #tmp_h = "### Human:\n" + item[0]
            #tmp_a = "\n ### Assistant:\n" + item[1]
            tmp_h = " <|start_header_id|>user<|end_header_id|> " + item[0]+ "<|eot_id|>"
            tmp_a = "<|start_header_id|>assistant<|end_header_id|> " + item[1]+"<|eot_id|>"
            if idx == 0:
                samples_splitted.append( "<|begin_of_text|>" + tmp_h + tmp_a)
            else:
                samples_splitted.append(tmp_h + tmp_a)
            names_splitted.append([psychologist_name, gaslighter_name])

    return samples_splitted


def process_conv_mix(convs, partition_list, convs_red):
    samples = {}
    samples_splitted = []
    pattern = re.compile('selected', flags=re.IGNORECASE)
    names = {}
    names_splitted = []
    for idx, (conv, conv_red) in enumerate(zip(convs, convs_red)):
        if idx in partition_list:
            conversation = conv[str(idx)]
            conversation_red = conv_red[str(idx)]
            samples[idx] = []
            convlist = []
            convlist_red = []
            for idz, (key, value) in enumerate(conversation.items()):
                if not pattern.match(key):
                    name, internal_thought, utterance = value['name'], value['internal'], value['utterance']
                    name_red, internal_thought_red, utterance_red = conversation_red[key]['name'], conversation_red[key]['internal'], conversation_red[key]['utterance']
                    convlist.append(name + ': ' + utterance)
                    convlist_red.append(name + ': ' + utterance_red)
                    if int(idz)%2==1:
                        psychologist_name = name
                    if int(idz)%2==0:
                        gaslighter_name = name

            for idk, item in enumerate(convlist):
                if idk%2==1:
                    if len(convlist[:idk]) == 1:
                        # samples[idx].append([convlist[:idk][0], convlist[idk]])
                        samples[idx].append([convlist_red[:idk][0], convlist[idk]])
                    else:
                        # samples[idx].append(['\n\n'.join(convlist[:idk]), convlist[idk]])
                        samples[idx].append(['\n\n'.join(convlist_red[:idk]), convlist[idk]])
            names[idx] = [psychologist_name, gaslighter_name]
    for key in samples.keys():
        items = samples[key]
        psychologist_name, gaslighter_name = names[key]
        for item in items:
            tmp_h = "<s>"+"<INST> " + item[0]
            tmp_a = " </INST> " + item[1]+"</s>"
            samples_splitted.append(tmp_h + tmp_a)
            names_splitted.append([psychologist_name, gaslighter_name])

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

def load_partition():
    with open('./data/partition.json', 'r') as f:
        partition = json.load(f)
    f.close()
    return partition

def compose_data(template, mix=False):
    convs = load_conv('blue_conversations_gpt-3.5_final.json')
    convs_red = load_conv('conversations_gpt-3.5_final.json')
    partition = load_partition()
    if mix:
        samples_tr = process_conv_mix_llama3(convs, partition['tr'], convs_red)
        samples_te = process_conv_mix_llama3(convs, partition['va'], convs_red)
    else:
        samples_tr = process_conv_llama3(convs_red, partition['tr'])
        samples_te = process_conv_llama3(convs_red, partition['va'])

    # samples_test, names = process_conv(convs, partition['te'])

    new_data_train = []
    new_data_eval = []
    # for idx, (inp, oup) in enumerate(samples_tr):
    #     text = template.format(inp, oup)
    #     new_data_train.append({'text': text})

    for idx, text in enumerate(samples_tr):
        new_data_train.append({'text': text})

    # for idx, (inp, oup) in enumerate(samples_te):
    #     text = template.format(inp, oup)
    #     new_data_eval.append({'text': text})

    for idx, text in enumerate(samples_te):
        new_data_eval.append({'text': text})

    # new_data_test = {}
    # for idx, (inp, oup) in enumerate(samples_test):
    #     psychologist_name, gaslighter_name = names[idx]
    #     new_data_test[idx]=[inp, psychologist_name, gaslighter_name]
    #
    # with open('./data/test_red.json', 'w') as f:
    #     json.dump(new_data_test, f)
    # f.close()

    return new_data_train, new_data_eval

# convs = load_conv('conversations_gpt4.json')
# samples = process_conv(convs)
# compose_data(template)
