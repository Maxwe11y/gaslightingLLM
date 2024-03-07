
import re
import io
import json


def format_checking():
    file = io.open('./data/strategies_patch_v3.txt', 'rb')
    buffer_reader = io.BufferedReader(file)
    load_strategy = buffer_reader.read().decode()
    checking_full = {}
    checking_sim = {}
    strategies = load_strategy.split('\n\t\n')
    pattern = re.compile(r'[Ll]ayer [\d](.*)utterance', flags=re.IGNORECASE)
    pattern_newline = re.compile('\n')
    pattern_split = re.compile(r'\n\n')
    pattern_sub = re.compile(r'layer \d: strategy', flags=re.IGNORECASE)
    pattern_number = re.compile('\d', flags=re.IGNORECASE)
    printdict = {}
    for idx, strategy in enumerate(strategies):
        if len(strategy) > 0:
            printdict[idx] = []
            checking_full[idx] = []
            if pattern_split.findall(strategy):
                layer_sum = strategy.split('\n\n')
            else:
                layer_sum = strategy.split('\n')
            for l in layer_sum:
                if pattern_newline.findall(l):
                    l = pattern_newline.sub(' ', l)
                    number = pattern_number.findall(l)[0].strip(' ')
                    l = pattern_sub.sub('Strategy {}'.format(number), l)
                    if pattern.match(l):
                        printdict[idx].append(l)
                        checking_full[idx].append(True)
                    else:
                        printdict[idx].append(l)
                        checking_full[idx].append(False)
                else:
                    if pattern.match(l):
                        number = pattern_number.findall(l)[0].strip(' ')
                        l = pattern_sub.sub('Strategy {}'.format(number), l)
                        printdict[idx].append(l)
                        checking_full[idx].append(True)
                    else:
                        printdict[idx].append(l)
                        checking_full[idx].append(False)

            if len(checking_full[idx]) in [4, 5, 6] and sum(checking_full[idx]) in [4, 5, 6]:
                checking_sim[idx] = True
            else:
                checking_sim[idx] = False

    with open('./data/strategies_2K_first.json', 'w') as f:
        json.dump(printdict, f)
    f.close()

    with open('./data/checking_full.json', 'w') as f:
        json.dump(checking_full, f)
    f.close()

    with open('./data/checking_sim.json', 'w') as f:
        json.dump(checking_sim, f)
    f.close()

    print('done!')

    return


def format_checking_json(results):

    try:
        res_json = json.loads(results)
    except json.decoder.JSONDecodeError:
        return False

    patten_str = re.compile(r'[Ll]ayer [\d](.*)strategy', flags=re.IGNORECASE)
    patten_utt = re.compile(r'[Ll]ayer [\d](.*)utterance', flags=re.IGNORECASE)
    result_list = []
    for key in res_json:
        if patten_str.match(key) or patten_utt.match(key):
            result_list.append(True)
        else:
            result_list.append(False)
    if len(result_list) in [8, 10, 12] and sum(result_list) in [8, 10, 12]:
        return True
    else:
        return False

def post_process():
    printdict = {}
    with open('./data/strategies_json_1359.txt') as f_1:
        data = f_1.readlines()
    pattern = re.compile(r'idx \d* Selection \dStrategy \d:', flags=re.IGNORECASE)
    pattern_idx = re.compile(r'idx \d*', flags=re.IGNORECASE)
    pattern_selection = re.compile(r'Selection \d', flags=re.IGNORECASE)
    pattern_strategy = re.compile(r'Strategy \d:', flags=re.IGNORECASE)
    pattern_num = re.compile(r'idx', flags=re.IGNORECASE)
    pattern_selection_ = re.compile(r'selection', flags=re.IGNORECASE)
    pattern_strategy_ = re.compile(r'strategy', flags=re.IGNORECASE)
    for line in data:
        line = line.strip()
        if line != '':
            res = pattern.findall(line)
            if len(res) == 2:
                idx = pattern_idx.findall(res[0])
                selection = pattern_selection.findall(res[0])
                strategy = pattern_strategy.findall(res[0])
                idx = pattern_num.split(idx[0])[1].strip(' ')
                printdict[int(idx)] = []
            else:
                idx = pattern_idx.findall(line)
                selection = pattern_selection.findall(line)
                strategy = pattern_strategy.findall(line)
                idx = pattern_num.split(idx[0])[1].strip(' ')
                printdict[int(idx)] = []

    for line in data:
        line = line.strip()
        if line and pattern.match(line):
            if len(pattern.findall(line))==1:

                # idx, selection, strategy_num = pattern_num.findall(line)

                idx_ = pattern_idx.findall(line)
                selection_ = pattern_selection.findall(line)
                strategy_ = pattern_strategy.findall(line)
                selection = pattern_selection_.split(selection_[0])[1][1]
                strategy_num = pattern_strategy_.split(strategy_[0])[1][1]
                idx = pattern_num.split(idx_[0])[1].strip(' ')

                line = pattern.sub('', line)
                strategy, utterance = line.split('\t\t')
                # key = '{}'.format(idx)
                strategy_x = 'strategy_{}'.format(strategy_num) + strategy
                printdict[int(idx)].append([strategy_x.strip(), utterance.strip(), selection])
            elif len(pattern.findall(line)) == 2:
                res = pattern.findall(line)

                # idx, selection, strategy_num = pattern_num.findall(res[0])

                idx_0 = pattern_idx.findall(res[0])
                selection_0 = pattern_selection.findall(res[0])
                strategy_0 = pattern_strategy.findall(res[0])
                selection0 = pattern_selection_.split(selection_0[0])[1][1]
                strategy_num0 = pattern_strategy_.split(strategy_0[0])[1][1]
                idx0 = pattern_num.split(idx_0[0])[1].strip(' ')

                idx_1 = pattern_idx.findall(res[1])
                selection_1 = pattern_selection.findall(res[1])
                strategy_1 = pattern_strategy.findall(res[1])
                selection1 = pattern_selection_.split(selection_1[0])[1][1]
                strategy_num1 = pattern_strategy_.split(strategy_1[0])[1][1]
                idx1 = pattern_num.split(idx_1[0])[1].strip(' ')

                # idx_0, selection_0, strategy_num_0 = pattern_num.findall(res[1])
                pattern_1 = re.compile(r'idx {} Selection {}Strategy {}:'.format(idx0, selection0, strategy_num0), flags=re.IGNORECASE)
                pattern_2 = re.compile(r'idx {} Selection {}Strategy {}:'.format(idx1, selection1, strategy_num1),
                                       flags=re.IGNORECASE)
                splitted = pattern_2.split(line)
                for s_u in splitted:
                    if pattern_1.match(s_u):
                        s_u = pattern_1.sub('', s_u)
                        strategy, utterance = s_u.split('\t\t')
                        # key = '{}_{}'.format(idx, selection)
                        strategy_x0 = 'strategy_{}'.format(strategy_num0) + strategy
                        printdict[int(idx0)].append([strategy_x0.strip(), utterance.strip(), selection0])
                    else:
                        # s_u = pattern_2.sub('', s_u)
                        strategy, utterance = s_u.split('\t\t')
                        # key = '{}_{}'.format(idx0, selection0)
                        strategy_x1 = 'strategy_{}'.format(strategy_num1) + strategy
                        printdict[int(idx1)].append([strategy_x1.strip(), utterance.strip(), selection1])

    print('done!')
    return printdict




def post_process_2():
    printdict = {}
    with open('./data/strategies_json_1564.txt') as f_1:
        data = f_1.readlines()
    pattern = re.compile(r'idx \d* Selection \dStrategy \d:', flags=re.IGNORECASE)
    pattern_idx = re.compile(r'idx \d*', flags=re.IGNORECASE)
    pattern_selection = re.compile(r'Selection \d', flags=re.IGNORECASE)
    pattern_strategy = re.compile(r'Strategy \d:', flags=re.IGNORECASE)
    pattern_num = re.compile(r'idx', flags=re.IGNORECASE)
    pattern_selection_ = re.compile(r'selection', flags=re.IGNORECASE)
    pattern_strategy_ = re.compile(r'strategy', flags=re.IGNORECASE)
    for line in data:
        line = line.strip()
        if line != '':
            res = pattern.findall(line)
            if len(res) == 2:
                idx = pattern_idx.findall(res[0])
                selection = pattern_selection.findall(res[0])
                strategy = pattern_strategy.findall(res[0])
                idx = pattern_num.split(idx[0])[1].strip(' ')
                printdict[int(idx)] = []
            else:
                idx = pattern_idx.findall(line)
                selection = pattern_selection.findall(line)
                strategy = pattern_strategy.findall(line)
                idx = pattern_num.split(idx[0])[1].strip(' ')
                printdict[int(idx)] = []

    for line in data:
        line = line.strip()
        if line and pattern.match(line):
            if len(pattern.findall(line)) == 1:

                # idx, selection, strategy_num = pattern_num.findall(line)

                idx_ = pattern_idx.findall(line)
                selection_ = pattern_selection.findall(line)
                strategy_ = pattern_strategy.findall(line)
                selection = pattern_selection_.split(selection_[0])[1][1]
                strategy_num = pattern_strategy_.split(strategy_[0])[1][1]
                idx = pattern_num.split(idx_[0])[1].strip(' ')

                line = pattern.sub('', line)
                strategy, utterance = line.split('\t\t')
                # key = '{}'.format(idx)
                strategy_x = 'strategy_{}'.format(strategy_num) + strategy
                printdict[int(idx)].append([strategy_x.strip(), utterance.strip(), selection])
            else:
                raise Exception('Invalid line')

    print('done!')
    return printdict

def post_process_3(index):
    printdict = {}
    with open('./data/strategies_json_{}.txt'.format(index)) as f_1:
        data = f_1.readlines()
    pattern = re.compile(r'idx \d* Selection \dStrategy \d:', flags=re.IGNORECASE)
    pattern_idx = re.compile(r'idx \d*', flags=re.IGNORECASE)
    pattern_selection = re.compile(r'Selection \d', flags=re.IGNORECASE)
    pattern_strategy = re.compile(r'Strategy \d:', flags=re.IGNORECASE)
    pattern_num = re.compile(r'idx', flags=re.IGNORECASE)
    pattern_selection_ = re.compile(r'selection', flags=re.IGNORECASE)
    pattern_strategy_ = re.compile(r'strategy', flags=re.IGNORECASE)
    for line in data:
        line = line.strip()
        if line != '':
            res = pattern.findall(line)
            if len(res) == 2:
                idx = pattern_idx.findall(res[0])
                selection = pattern_selection.findall(res[0])
                strategy = pattern_strategy.findall(res[0])
                idx = pattern_num.split(idx[0])[1].strip(' ')
                printdict[int(idx)] = []
            else:
                idx = pattern_idx.findall(line)
                selection = pattern_selection.findall(line)
                strategy = pattern_strategy.findall(line)
                idx = pattern_num.split(idx[0])[1].strip(' ')
                printdict[int(idx)] = []

    for line in data:
        line = line.strip()
        if line and pattern.match(line):
            if len(pattern.findall(line)) == 1:

                # idx, selection, strategy_num = pattern_num.findall(line)

                idx_ = pattern_idx.findall(line)
                selection_ = pattern_selection.findall(line)
                strategy_ = pattern_strategy.findall(line)
                selection = pattern_selection_.split(selection_[0])[1][1]
                strategy_num = pattern_strategy_.split(strategy_[0])[1][1]
                idx = pattern_num.split(idx_[0])[1].strip(' ')

                line = pattern.sub('', line)
                strategy, utterance = line.split('\t\t')
                # key = '{}'.format(idx)
                strategy_x = 'strategy_{}'.format(strategy_num) + strategy
                printdict[int(idx)].append([strategy_x.strip(), utterance.strip(), selection])
            else:
                raise Exception('Invalid line')

    print('done!')
    return printdict


def post_process_v2():
    printdict = {}
    with open('./data/strategies_json_second.txt') as f_1:
        data = f_1.readlines()
    pattern = re.compile(r'idx \d* Selection \d Strategy \d:', flags=re.IGNORECASE)
    pattern_idx = re.compile(r'idx \d*', flags=re.IGNORECASE)
    pattern_selection = re.compile(r'Selection \d', flags=re.IGNORECASE)
    pattern_strategy = re.compile(r'Strategy \d:', flags=re.IGNORECASE)
    pattern_num = re.compile(r'idx', flags=re.IGNORECASE)
    pattern_selection_ = re.compile(r'selection', flags=re.IGNORECASE)
    pattern_strategy_ = re.compile(r'strategy', flags=re.IGNORECASE)
    for line in data:
        line = line.strip()
        if line != '':
            res = pattern.findall(line)
            if len(res) == 2:
                idx = pattern_idx.findall(res[0])
                selection = pattern_selection.findall(res[0])
                strategy = pattern_strategy.findall(res[0])
                idx = pattern_num.split(idx[0])[1].strip(' ')
                printdict[int(idx)] = []
            else:
                idx = pattern_idx.findall(line)
                selection = pattern_selection.findall(line)
                strategy = pattern_strategy.findall(line)
                idx = pattern_num.split(idx[0])[1].strip(' ')
                printdict[int(idx)] = []

    for line in data:
        line = line.strip()
        if line and pattern.match(line):
            if len(pattern.findall(line)) == 1:

                # idx, selection, strategy_num = pattern_num.findall(line)

                idx_ = pattern_idx.findall(line)
                selection_ = pattern_selection.findall(line)
                strategy_ = pattern_strategy.findall(line)
                selection = pattern_selection_.split(selection_[0])[1][1]
                strategy_num = pattern_strategy_.split(strategy_[0])[1][1]
                idx = pattern_num.split(idx_[0])[1].strip(' ')

                line = pattern.sub('', line)
                strategy, utterance = line.split('\t\t')
                # key = '{}'.format(idx)
                strategy_x = 'strategy_{}'.format(strategy_num) + strategy
                printdict[int(idx)].append([strategy_x.strip(), utterance.strip(), selection])
            else:
                raise Exception('Invalid line')

    print('done!')
    return printdict








def post():
    # p1 = post_process()
    # p2 = post_process_3(1360)
    # p3 = post_process_2()
    # p4 = post_process_3(1565)
    # p5 = post_process_3(2000)
    # newdict = {}
    # for key, value in p1.items():
    #     newdict[key] = value
    # for key, value in p2.items():
    #     newdict[key] = value
    # for key, value in p3.items():
    #     newdict[key] = value
    # for key, value in p4.items():
    #     newdict[key] = value
    # for key, value in p5.items():
    #     newdict[key] = value
    #
    # print('done!')
    #
    # with open('./data/strategies_2K.json', 'w') as f:
    #     json.dump(newdict, f)
    # f.close()


    # p1 = post_process_second()
    # p2 = post_process_3(429)

    # p1[429] = p2[429]
    #
    # with open('./data/strategies_2K_second.json', 'w') as f:
    #     json.dump(p1, f)

    print('done!')



def post_process_nonjson():
    # patchlist = []
    # with open('./data/patch_v2.txt', 'r') as f:
    #     indices = f.readlines()
    #     for index in indices:
    #         patchlist.append(int(index.strip()))
    # f.close()
    # with open('./data/strategies_2K_first_v2.json', 'r') as f:
    #     data = json.load(f)
    # f.close()
    # with open('./data/checking_full_patch_v2.json', 'r') as f_:
    #     checking = json.load(f_)
    # f_.close()
    # with open('./data/strategies_2K_first_patch_v2.json', 'r') as f_patch:
    #     data_patch = json.load(f_patch)
    # f_patch.close()

    # count = 0
    # count_total = 0
    # printlist = []
    # for key_1, key_2 in zip(data, checking):
    #     strategy = data[key_1]
    #     correctness = checking[key_2]
    #     if sum(correctness) != len(correctness):
    #         count+=1
    #         print(key_1)
    #         printlist.append(key_1)

    # for idx, keys in enumerate(patchlist):
    #     data[keys] = data_patch[str(idx)]
    # with open('./data/patch_v2.txt', 'w') as f:
    #     f.writelines('\n'.join(printlist))

        # for str_, cor_ in zip(strategy, correctness):
        #     count_total+=1
        #     if not cor_:
        #         count+=1
        #         print(key_1, str_)

    # with open('./data/strategies_2K_first_v3.json', 'r') as f:
    #     data = json.load(f)
    # f.close()
    # pattern = re.compile(r'[Ss]trategy [\d](.*)utterance', flags=re.IGNORECASE)
    # pattern_sub = re.compile(r'[Ll]ayer \d(.*)strategy', flags=re.IGNORECASE)
    # pattern_num = re.compile(r'\d')
    # for i in data:
    #     items = data[i]
    #     for idj, item in enumerate(items):
    #         if not pattern.match(item):
    #             # print(i, item)
    #             if pattern_sub.match(item):
    #                 number = pattern_num.findall(item)[0].strip()
    #                 item = pattern_sub.sub('strategy {}'.format(number), item)
    #                 data[i][idj] = item
    #                 # print(item)
    #
    # twoline = [106, 1051, 1220, 1424]
    # threeline = [254]
    # for i in twoline:
    #     items = data[str(i)]
    #     new_items = []
    #     for idj in range(0, len(items), 2):
    #         tmp = items[idj] + ' ' + items[idj+1]
    #         new_items.append(tmp)
    #     data[str(i)] = new_items
    #
    # for i in threeline:
    #     items = data[str(i)]
    #     new_items = []
    #     for idj in range(0, len(items), 3):
    #         tmp = items[idj+1] + ' ' + items[idj+2]
    #         new_items.append(tmp)
    #     data[str(i)] = new_items
    #
    #
    # print('---------------------------------------')
    # bugset = []
    # for i in data:
    #     items = data[i]
    #     for idj, item in enumerate(items):
    #         if not pattern.match(item):
    #             bugset.append(i)
    #             print(i, item)
    # print(set(bugset))
    #
    # retrieve_list = ['50', '62', '108', '170', '254', '456', '643', '755', '884', '1346', '1965']
    # print('----------------------------------------')
    # with open('./data/strategies_2K_first_patch_v3.json', 'r') as f:
    #     patches = json.load(f)
    #
    # for idx, x in enumerate(retrieve_list):
    #     data[x] = patches[str(idx)]
    #
    # with open('./data/strategies_2K_first_v4.json', 'w') as f:
    #     json.dump(data, f)
    # f.close()

    print('done!')


def process_selection():
    # sel_ = []
    # sel_patch_ = []
    # sel_patch_v2_ = []
    # with open('./data/selected.txt', 'r') as f:
    #     sel = f.readlines()
    #     for x in sel:
    #         if x.strip():
    #             sel_.append(x.strip())
    #
    # with open('./data/selected_patch.txt', 'r') as f:
    #     sel_patch = f.readlines()
    #     for x in sel_patch:
    #         if x.strip():
    #             sel_patch_.append(x.strip())
    #
    # with open('./data/selected_patch_v2.txt', 'r') as f:
    #     sel_patch_v2 = f.readlines()
    #     for x in sel_patch_v2:
    #         if x.strip():
    #             sel_patch_v2_.append(x.strip())
    # patch_x = []
    # with open('./data/patch.txt', 'r') as f:
    #     patch = f.readlines()
    #     for x in patch:
    #         patch_x.append(x.strip())
    #
    # patch_x2 = []
    # with open('./data/patch_v2.txt', 'r') as f:
    #     patch = f.readlines()
    #     for x in patch:
    #         patch_x2.append(x.strip())
    #
    # for idj, idx in enumerate(patch_x):
    #     sel_[int(idx)] = sel_patch_[idj]
    # for idj, idx in enumerate(patch_x2):
    #     sel_[int(idx)] = sel_patch_v2_[idj]
    #
    # with open('./data/selected_updated.txt', 'w') as f:
    #     f.writelines('\n'.join(sel_))

    # retrieve_list = ['50', '62', '108', '170', '254', '456', '643', '755', '884', '1346', '1965']
    # data_ = []
    # with open('./data/selected_updated.txt', 'r') as f:
    #     data = f.readlines()
    #     for i in data:
    #         data_.append(i.strip())
    # patch_ = []
    # with open('./data/selected_patch_v3.txt', 'r') as fx:
    #     patch = fx.readlines()
    #     for i in patch:
    #         if i.strip():
    #             patch_.append(i.strip())
    # for idx, i in enumerate(retrieve_list):
    #     data_[int(i)] = patch_[idx]
    # with open('./data/selected_final.txt', 'w') as f:
    #     f.writelines('\n'.join(data_))
    # f.close()
    #
    print('done')





