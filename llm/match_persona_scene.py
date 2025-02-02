import os
import json
import pandas as pd
from get_ada_embedding import safe_embedding_response
from transformers.pipelines import PIPELINE_REGISTRY
from transformers import AutoTokenizer, AutoModel
from transformers import pipeline
from pipeline_customized import MyPipeline
# from get_mistral_embedding import get_detailed_instruct, load_data
from transformers import BitsAndBytesConfig
import torch
from mistral_embedding_only import similarity_metrics
import numpy as np
from openai import AsyncOpenAI
import asyncio
from tqdm import tqdm
import random

client = AsyncOpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_quant_type="nf4",
    bnb_8bit_compute_dtype=torch.float16,
    bnb_8bit_use_double_quant=True,
)
def load_persona(filename, path='./SPC'):
    personas_unified = []
    with open(os.path.join(path, filename), 'r') as f:
        personas = json.load(f)
        for idx in personas:
            persona = personas[idx]
            persona = '  '.join(persona)
            personas_unified.append(persona)

    return personas_unified

def load_scenes(filename, path='./data'):
    with open(os.path.join(path, filename)) as f:
        scenes = f.readlines()

    return scenes


def get_mistral_embedding():
    personas = load_persona('persona.json')
    scenes = load_scenes('gpt_scenario_mistral_final_x.txt')
    tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-mistral-7b-instruct', padding_side='left')
    PIPELINE_REGISTRY.register_pipeline(
        "text-similarity",
        pipeline_class=MyPipeline,
        pt_model=AutoModel
    )

    model = AutoModel.from_pretrained('intfloat/e5-mistral-7b-instruct', quantization_config=bnb_config,
                                      device_map='auto', trust_remote_code=True)

    pipe = pipeline("text-similarity", model=model, tokenizer=tokenizer, device_map='auto', trust_remote_code=True)

    sim_personas = similarity_metrics(personas, pipe=pipe, batch=512, path='./embedding', save_name1='mistral_embedding_persona_1.pth', save_name2='mistral_embedding_persona_2.pth')  # 1024, 5

    sim_scenes = similarity_metrics(scenes, pipe=pipe, batch=512, path = './embedding', save_name1='mistral_embedding_scene_1.pth', save_name2='mistral_embedding_scene_2.pth')

    return sim_personas, sim_scenes


def get_ada_embedding():
    personas = load_persona('persona.json')
    scenes = load_scenes('gpt_final_x.txt')
    df_per = pd.DataFrame(personas, columns=['personas'])
    df_sce = pd.DataFrame(scenes, columns=['scenes'])

    df_per['ada_embedding_per'] = df_per.personas.apply(lambda x: safe_embedding_response(x, model='text-embedding-ada-002'))
    df_per.to_csv('embedding/ada_persona_embedding.csv', index=False)

    df_sce['ada_embedding_sce'] = df_sce.scenes.apply(lambda x: safe_embedding_response(x, model='text-embedding-ada-002'))
    df_sce.to_csv('embedding/ada_scene_embedding.csv', index=False)

    return


def compute_similarity(embedding_persona, embedding_scene):
    length_per = len(embedding_persona)
    length_scene = len(embedding_scene)
    similarity_metrics = np.zeros((length_scene, length_per))
    for idx, embed_s in enumerate(embedding_scene):
        embed_s = np.array(eval(embed_s))
        for idj, embed_p in enumerate(embedding_persona):
            embed_p = np.array(eval(embed_p))
            cos_sim = embed_s.dot(embed_p) / (np.linalg.norm(embed_s) * np.linalg.norm(embed_p))
            similarity_metrics[idx][idj] = cos_sim

    np.save('./embedding/sce_per_similarity.npy', similarity_metrics)
    return similarity_metrics

def load_csv(filename, column_name = ''):
    data = pd.read_csv(filename)
    embeddings = data[column_name]
    return embeddings

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


def check_conflict(scene, persona, num_per):

    prompt_NLI_x = "Task Description of Natural Language Inference: \n" + "Premise: Two women are hugging each other." + "\nHypothesis: Two women are showing affection." \
                 + "\nLabel: Entailment" + "\nReason: the above Premise-Hypothesis pair will be labeled as Entailment because 'showing affection' in the Hypothesis can be inferred from 'hugging one another' in the Premise." \
                                                "\nPremie: A man is running the coding example of Image Recognition." + "\nHypothesis: The man is sleeping." + "\nLabel: Contradiction"\
                                                "\nReason: contradiction as 'running the coding example' indicates 'not sleeping' rather than 'sleeping'." \
                                                "\nPremie: The pianist is performing for us. \nHypothesis: The pianist is famous. \nLabel: Neutral" \
                                                "\nReason: the example shows a neutrality relationship because neither 'famous' nor 'not famous' can be inferred from the fact that 'are performing for us'." \
                                              "\nAssume that you’re an expert working on natural language inference tasks. Given a Premise and Hypothesis below, " \
                                              "please follow the reasoning steps in the aforementioned example to determine the logical relationship between " \
                                              "Premise and Hypothesis from three predefined labels, i.e., Entailment, Contradiction, and Neutral." \
                 + "The detailed explanation of the three labels are: \nEntailment: the hypothesis can be inferred from the premise. " \
                 + "\nContradiction: the negation of the hypothesis can be inferred from the premise. \nNeutral: all the other cases." \
                 + "\nHere is the given Premise-Hypothesis pair: " \
                 + "\nPremise:\n{}".format(persona) \
                 + "\nHypothesis:\n{}".format(scene) \
                 + "\nThe label for the Premise-Hypothesis pair is: \n"


    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(chat(prompt_NLI_x))

    while result.lower() not in ['entailment', 'contradiction', 'neutral']:
        result = loop.run_until_complete(chat(prompt_NLI_x))

    return result.lower()


def greedy_match():
    matched_sce_per = {}
    personas = load_persona('persona.json')
    # use mistral embedding instead of ada embedding
    scenes = load_scenes('gpt_scenario_mistral_final_x.txt')
    df_per = pd.DataFrame(personas, columns=['personas'])
    df_sce = pd.DataFrame(scenes, columns=['scenes'])

    # metrics = np.load('./embedding/sce_per_similarity.npy')
    metrics_per1_sce2 = np.load('./embedding/mistral_similarity_per1_sce2.npy')
    metrics_per2_sce1 = np.load('./embedding/mistral_similarity_per2_sce1.npy')
    metrics = (metrics_per2_sce1 + metrics_per1_sce2) / 2.0
    count = 0
    l = [0 for _ in range(len(scenes))]
    while sum(l) < len(scenes):
        if sum(l) % 100 == 0:
            print(sum(l))
            print('----------------------')
        index_ = metrics.argmax()
        row = index_ // len(personas)
        col = index_ % len(personas)

        scene = df_sce['scenes'][row]
        scene_revised = 'I ' + ' '.join(scene.split(' ')[1:])
        persona= df_per['personas'][col]
        num_per = len(persona.split('  '))
        if check_conflict(scene_revised, persona, num_per) == 'contradiction':
            count += 1
            print('idx', row)
            print('ind', col)
            print('scene', scene)
            print('persona', persona)
            metrics[row][col] = -1.0
            print('\n')
        else:
            matched_sce_per[scene] = persona.split("  ")
            metrics[row] = -1.0
            metrics[:, col] = -1.0
            l[row] = 1

    print('count', count)
    print('done!')

    with open('./embedding/match_sce_per_v4.json', 'w') as f:
        json.dump(matched_sce_per, f)
    f.close()

    return matched_sce_per



if __name__ == "__main__":

    # get_ada_embedding()
    # compute the similarity matrix between scenes and personas
    # embedding_persona = load_csv('./embedding/ada_persona_embedding.csv', column_name='ada_embedding_per')
    # embeddings_scene = load_csv('./embedding/ada_scene_embedding.csv', column_name='ada_embedding_sce')
    # compute_similarity(embedding_persona, embeddings_scene)


    # utilize the similarity matrix to match scenario with the most relevant persona
    # matched_sce_per = {}
    # personas = load_persona('persona.json')
    # # use mistral embedding instead of ada embedding
    # scenes = load_scenes('gpt_scenario_mistral_final_x.txt')
    # df_per = pd.DataFrame(personas, columns=['personas'])
    # df_sce = pd.DataFrame(scenes, columns=['scenes'])
    #
    # # metrics = np.load('./embedding/sce_per_similarity.npy')
    # metrics_per1_sce2 = np.load('./embedding/mistral_similarity_per1_sce2.npy')
    # metrics_per2_sce1 = np.load('./embedding/mistral_similarity_per2_sce1.npy')
    # metrics = (metrics_per2_sce1+metrics_per1_sce2)/2.0
    # k = 5
    # count = 0
    # for idx, sim in tqdm(enumerate(metrics), desc= 'match...'):
    #     # selected = torch.argmax(torch.tensor(metrics[idx]))
    #     # random select one persona from top k persona to avoid overuse of some personas
    #     selected_personas = torch.topk(torch.tensor(metrics[idx]), k=k)
    #
    #     for ind in selected_personas[1]:
    #         # persona_ind = selected_personas[1][ind].item()
    #         # persona_ind = random.choice(selected_personas[1]).item()
    #         persona_ind = ind.item()
    #         persona = df_per['personas'][persona_ind]
    #
    #         scene = df_sce['scenes'][idx]
    #         scene_revised = 'I ' + ' '.join(scene.split(' ')[1:])
    #         num_per = len(persona.split('  '))
    #         if check_conflict(scene_revised, persona, num_per) == 'contradiction':
    #             count+=1
    #             print('idx', idx)
    #             print('ind', ind)
    #             print('scene', scene)
    #             print('persona', persona)
    #             print('\n')
    #             continue
    #         else:
    #             matched_sce_per[scene] = persona.split("  ")
    #             metrics[idx] = -1.0
    #             metrics[:,persona_ind] = -1.0
    #             break
    # print('count', count)
    # print('done!')


    # use greedy match algorithm
    # res = greedy_match()


    # load matched scenes and personas
    personas = []
    sce_per = []
    with open('./embedding/match_sce_per_v4.json', 'r') as f:
        data = json.load(f)

        for scene in data:
            personas.append(data[scene])
            sce_per.append([scene, data[scene]])

    f.close()
    res = []
    for persona in personas:
        if persona not in res:
            res.append(persona)
    print('done!')

