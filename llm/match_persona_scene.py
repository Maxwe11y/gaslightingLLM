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
    for idx, embed_p in enumerate(embedding_scene):
        for idj, embed_s in enumerate(embedding_persona):
            embed_p = np.array(eval(embed_p))
            embed_s = np.array(eval(embed_s))
            cos_sim = embed_s.dot(embed_p) / (np.linalg.norm(embed_s) * np.linalg.norm(embed_p))
            similarity_metrics[idx][idj] = cos_sim

    np.save('./embedding/sce_per_similarity.npy', similarity_metrics)
    return similarity_metrics

def load_csv(filename, column_name = ''):
    data = pd.read_csv(filename)
    embeddings = data[column_name]
    return embeddings


if __name__ == "__main__":
    get_ada_embedding()

    # embedding_persona = load_csv('./embedding/ada_persona_embedding.csv', column_name = 'personas')
    # embeddings_scene = load_csv('./embedding/ada_scene_embedding.csv', column_name = 'scenes')
    # compute_similarity(embedding_persona, embeddings_scene)

    print('done!')