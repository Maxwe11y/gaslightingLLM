import numpy as np
from tqdm import tqdm
from transformers.pipelines import PIPELINE_REGISTRY
from transformers import AutoTokenizer, AutoModel
from transformers import pipeline
from pipeline_mistral_external import MyPipeline
from get_mistral_embedding import get_detailed_instruct, load_data
from transformers import BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_quant_type="nf4",
    bnb_8bit_compute_dtype=torch.float16,
    bnb_8bit_use_double_quant=True,
)

def compute_similarity(line1=None, line2=None, pipe=None):
    # Each query must come with a one-sentence instruction that describes the task
    task = 'Given a conversation scenario, retrieve a text that is similar to the given scenario from the aspect of topic, semantics and theme.'
    # query = [get_detailed_instruct(task, line1)]
    # input_texts = [query+line for line in line2]
    input_texts = [get_detailed_instruct(task, line1)]
    input_texts.extend(line2)

    # res = pipe([input_texts])[0].squeeze().tolist()
    embeddings = pipe(input_texts)
    e = torch.cat(embeddings, dim=0)
    similarity = e[:1] @ e[1:].T
    res = similarity.squeeze().tolist()

    # print("res", res)
    return res


def similarity_metrics(data, pipe=None, batch=1):
    similarity_metrics = np.zeros((len(data), len(data)))
    # similarity_metrics_doc = np.zeros((len(data), len(data)))
    for idx, line1 in tqdm(enumerate(data), desc='compute similarity...'):
        for idj in range(0, len(data), batch):
            result = compute_similarity(line1, data[idj:idj+batch], pipe=pipe)
            length = min(len(result), batch)
            # if idj > idx:
            similarity_metrics[idx][idj:idj+length] = result[:]  #[1:batch+1]
            # elif idj < idx:
            #     similarity_metrics_doc[idx][idj:idj+length] = result[:]  #[1:batch+1]
    np.save('./data/pipe_similarity_metrics.npy', similarity_metrics)
    # np.save('./data/pipe_similarity_metrics_doc.npy', similarity_metrics_doc)
    return similarity_metrics


if __name__ == "__main__":
    data = load_data('sce_topic_filtered.txt')
    data = data[0:100]
    tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-mistral-7b-instruct', padding_side='left')
    PIPELINE_REGISTRY.register_pipeline(
        "text-similarity",
        pipeline_class=MyPipeline,
        pt_model=AutoModel
    )

    model = AutoModel.from_pretrained('intfloat/e5-mistral-7b-instruct', quantization_config=bnb_config, device_map='auto', trust_remote_code=True)

    pipe = pipeline("text-similarity", model=model, tokenizer=tokenizer, device_map='auto', trust_remote_code=True)

    similarity_metrics = similarity_metrics(data, pipe=pipe, batch=16384)