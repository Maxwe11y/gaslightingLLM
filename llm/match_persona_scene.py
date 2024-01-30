import os
import json
import pandas as pd
from get_ada_embedding import calculate_similarity_matrix, safe_embedding_response
from transformers.pipelines import PIPELINE_REGISTRY
from transformers import AutoTokenizer, AutoModel
from transformers import pipeline
from pipeline_customized import MyPipeline
from get_mistral_embedding import get_detailed_instruct, load_data
from transformers import BitsAndBytesConfig
import torch
from mistral_embedding_only import similarity_metrics
import numpy as np
from openai import AsyncOpenAI
import asyncio

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


def check_conflict(scene, persona):
    prompt = "You are a specialist tasked with scrutinizing and verifying whether there are factual discrepancies between two portrayals of an individual." \
              + "For example, the statements 'Alex is going to pick up his children.' and 'Alex does not have any children.' present a contradiction regarding the factual assertion of whether Alex has children." \
              + "Given the two descriptions below, kindly indicate 'yes' if there are factual inconsistencies, and 'no' if there are none. if yes, please give elaborations.\n" \
              + "portrayal one {}".format(scene) \
              + "\n portrayal two {}".format(persona)


    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(chat(prompt))

    while result.lower() not in ['yes', 'no']:
        result = loop.run_until_complete(chat(prompt))

    return result


if __name__ == "__main__":
    # get_ada_embedding()
    # compute the similarity matrix between scenes and personas
    # embedding_persona = load_csv('./embedding/ada_persona_embedding.csv', column_name='ada_embedding_per')
    # embeddings_scene = load_csv('./embedding/ada_scene_embedding.csv', column_name='ada_embedding_sce')
    # compute_similarity(embedding_persona, embeddings_scene)


    # utilize the similarity matrix to match scenario with the most relevant persona
    matched_sce_per = {}
    personas = load_persona('persona.json')
    scenes = load_scenes('gpt_final_x.txt')
    df_per = pd.DataFrame(personas, columns=['personas'])
    df_sce = pd.DataFrame(scenes, columns=['scenes'])

    metrics = np.load('./embedding/sce_per_similarity.npy')
    k=3
    for idx, sim in enumerate(metrics):
        # selected = torch.argmax(torch.tensor(metrics[idx]))
        selected_personas = torch.topk(torch.tensor(metrics[idx]), k=k)
        for ind in range(k):
            persona_ind = selected_personas[1][ind].item()
            persona = df_per['personas'][persona_ind]

            scene = df_sce['scenes'][idx]
            if check_conflict(scene, persona) != 'no':
                continue
            else:
                matched_sce_per[scene] = persona.split("  ")
    print('done!')

    with open('./embedding/match_sce_per.json', 'w') as f:
        json.dump(matched_sce_per, f)
    f.close()


    print('done!')