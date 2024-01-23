import numpy as np
import spacy
import os
from datasets import load_dataset
from spacy.symbols import ORTH
# from spacy.pipeline import EntityRuler
from tqdm import tqdm
import json
import re
import numpy
import matplotlib.pyplot as plt


# summary_train = dataset['train']['topic']
# print("done")


def merge(dataset, path="./data"):

    if not os.path.exists(os.path.join(path, "summary.txt")) \
        or not os.path.exists(os.path.join(path, "topic.txt")):
        summaries = []
        topics = []
        pattern = re.compile(r'\n')
        for item in ['train', 'validation', 'test']:
            summary = dataset[item]['summary']
            topic = dataset[item]['topic']
            for sum_, top_ in zip(summary, topic):
                new_sum = pattern.sub(' ', sum_)
                new_top = pattern.sub(' ', top_)
                summaries.append(new_sum)
                topics.append(new_top)

        save_data(summaries, "summary.txt")
        save_data(topics, "topics.txt")
    else:
        summaries = load_data("summary.txt")
        topics = load_data("topics.txt")

    return summaries, topics


def extract_scenario(summaries, topics, path='./data'):

    if not os.path.exists(os.path.join(path, "scenarios.txt")) or not os.path.exists(os.path.join(path, "filtered_topics.txt")):

        nlp = spacy.load("en_core_web_lg")

        # Add special case rule
        special_case = [{ORTH: '#Person1#'}]
        nlp.tokenizer.add_special_case("#Person1#", special_case)
        special_case = [{ORTH: '#Person2#'}]
        nlp.tokenizer.add_special_case("#Person2#", special_case)

        # Create EntityRuler instance
        # ruler = EntityRuler(nlp)
        ruler = nlp.add_pipe("entity_ruler")
        # Define pattern for new entity
        ruler.add_patterns([{"label": "PERSON", "pattern": '#Person1#'},
                            {"label": "PERSON", "pattern": '#Person2#'}])
        # Update existing pipeline
        # nlp.add_pipe(ruler, before="ner")
        # Title
        titles = {"Dr.", "Mr.", "Mrs.", "Miss.", "Ms.", "Mr", "Mrs", "Miss", "Ms"}

        scenarios = []
        filtered_topics = []
        for summary, topic in tqdm(zip(summaries, topics),desc="extracting scenarios and topics ... "):
            # tokens = summary.split(' ')
            new_summary = " 's".join(summary.split("\'s"))

            doc = nlp(new_summary)
            sent = list(doc.sents)[0]
            have_name = False
            count = 0
            tokens = []
            names = {'#Person1#': '#Person1#', '#Person2#': '#Person2#'}
            for ent in sent.ents:
                if ent.text in ['#Person1#', '#Person2#']:
                    count += 1
                elif ent.label_ == 'PERSON':
                    if ent.text not in names:
                        count += 1
                        names[ent.text] = '#Person{}#'.format(count)
            for token in sent.text.split(' '):
                if token in names:
                    tokens.append(names[token])
                    have_name = True
                elif token in titles:
                    pass
                else:
                    tokens.append(token)

            if not have_name:
                continue
            else:
                new_sent = " ".join(tokens)
                new_sent = "'s".join(new_sent.split(" 's"))
                scenarios.append(new_sent)
                filtered_topics.append(topic)

        # sce_topic = {"scenarios":scenarios, "topics":filtered_topics}

        # save_pkl(sce_topic, "scenario_topic.pkl")
        save_data(scenarios, "scenarios.txt")
        save_data(filtered_topics, "filtered_topics.txt")

    else:
        scenarios = load_data("scenarios.txt")
        filtered_topics = load_data("filtered_topics.txt")
        # sce_topic = load_pkl("scenario_topic.pkl")


    print("{} raw scenarios are extracted.".format(len(scenarios)))
    return scenarios, filtered_topics


def save_data(data, filename, path="./data"):

    with open(os.path.join(path, filename), "w") as f:
        f.writelines("\n".join(data))


def save_json(data, filename, path="./data"):

    with open(os.path.join(path, filename), "w") as f:
        json.dump(data, f)


def load_data(filename, path="./data"):
    data = []
    with open(os.path.join(path, filename), "r") as f:
        lines = f.readlines()
        for line in lines:
            data.append(line.strip())

    return data

def load_json(filename, path="./data"):

    with open(os.path.join(path, filename), "r") as f:
        data = json.load(f)

    return data

def filter_scenario(scenarios, topics, lower=4, upper=19):
    # remove long and short scenarios (before 14157 after 11940)
    number_tokens = []
    filtered_scenario = []
    filtered_topic = []
    filtered_num_tokens = []
    pattern = re.compile(r'#Person[0-9]#')
    for scenario, topic in tqdm(zip(scenarios, topics)):
        tmp = pattern.sub(' ', scenario)
        num_tokens = len(tmp.split())
        number_tokens.append(num_tokens)
        if num_tokens>lower and num_tokens<upper:
            filtered_scenario.append(scenario)
            filtered_topic.append(topic)
            filtered_num_tokens.append(num_tokens)

    # lower = np.percentile(number_tokens, 5)
    # upper =np.percentile(number_tokens, 90)

    # plt.hist(np.array(number_tokens), bins=57)
    # plt.show()

    # remove according to duplicated topics (before 11940 after 7265)
    topic_dic = {}
    for scenario, topic in tqdm(zip(filtered_scenario, filtered_topic)):
        if topic not in topic_dic:
            topic_dic[topic] = scenario

    sce_topic = ["\t".join([scenario, topic]) for topic, scenario in topic_dic.items()]
    # save_json(topic_dic, 'topic_to_scenario.json')
    save_data(sce_topic, "sce_topic_filtered.txt")

    return sce_topic


if __name__ == '__main__':
    dataset = load_dataset("knkarthick/dialogsum")

    summaries, topics = merge(dataset)
    scenarios, filtered_topics = extract_scenario(summaries, topics)
    sce_topic = filter_scenario(scenarios, topics)

    print("done")