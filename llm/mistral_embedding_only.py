import numpy as np
from tqdm import tqdm
from transformers.pipelines import PIPELINE_REGISTRY
from transformers import AutoTokenizer, AutoModel
from transformers import pipeline
# from optimum.onnxruntime import ORTModel
# from optimum.pipelines import pipeline
from pipeline_customized import MyPipeline
from get_mistral_embedding import get_detailed_instruct, load_data
from transformers import BitsAndBytesConfig
import torch
import pandas as pd

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_quant_type="nf4",
    bnb_8bit_compute_dtype=torch.float16,
    bnb_8bit_use_double_quant=True,
)

def compute_similarity(line1=None, pipe=None):
    # Each query must come with a one-sentence instruction that describes the task
    task = 'Given a conversation scenario, retrieve a text that is similar to the given scenario from the aspect of topic, semantics and theme.'
    query = [get_detailed_instruct(task, line_1) for line_1 in line1]
    outputs_1 = pipe([query])
    # embeddings_1 = []

    # for output in outputs_1:
    #     embeddings_1.extend(output.squeeze().tolist())

    line2 = line1[:]
    outputs_2 = pipe([line2])

    # embeddings_2 = []
    # for output in outputs_2:
    #     embeddings_2.extend(output.squeeze().tolist())

    return outputs_1[0], outputs_2[0]

def similarity_metrics(data, pipe=None, batch=1):

    embeddings_1 = []
    embeddings_2 = []
    for idx in tqdm(range(0, len(data), batch), desc='batch generation of embeddings'):
        scenarios = data[idx:idx+batch]
        embed_1, embed_2 = compute_similarity(line1=scenarios, pipe=pipe)
        embeddings_1.append(embed_1)
        embeddings_2.append(embed_2)

    # print('shape_1_before', embeddings_1[0].size())
    # print('shape_2_after', embeddings_2[0].size())

    embeddings_1 = torch.cat(embeddings_1, dim=0)
    embeddings_2 = torch.cat(embeddings_2, dim=0)
    # print('shape_1', embeddings_1.size())
    # print('shape_2', embeddings_2.size())

    # df = pd.DataFrame([embeddings_1, embeddings_2], columns=['line1', 'line2'])
    # df.to_csv('data/mistral_embedding.csv', index=False)
    torch.save(embeddings_1, './data/line_1.pth')
    torch.save(embeddings_2, './data/line_2.pth')

    return None


if __name__ == "__main__":
    data = load_data('sce_topic_filtered.txt')
    # data = data[0:10]
    tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-mistral-7b-instruct', padding_side='left')
    PIPELINE_REGISTRY.register_pipeline(
        "text-similarity",
        pipeline_class=MyPipeline,
        pt_model=AutoModel
    )

    model = AutoModel.from_pretrained('intfloat/e5-mistral-7b-instruct', quantization_config=bnb_config, device_map='auto', trust_remote_code=True)

    pipe = pipeline("text-similarity", model=model, tokenizer=tokenizer, device_map='auto', trust_remote_code=True)

    similarity_metrics = similarity_metrics(data, pipe=pipe, batch=512) # 1024, 5