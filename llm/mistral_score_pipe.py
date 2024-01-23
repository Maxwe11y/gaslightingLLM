import numpy as np
from tqdm import tqdm
from transformers.pipelines import PIPELINE_REGISTRY
from transformers import AutoTokenizer, AutoModel
from transformers import pipeline
# from optimum.onnxruntime import ORTModel
# from optimum.pipelines import pipeline
from pipeline_mistral import MyPipeline
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
    query = [get_detailed_instruct(task, line1)]
    input_texts = [query+line for line in line2]

    # input_texts = [get_detailed_instruct(task, line1)]
    # input_texts.extend(line2)
    # res = pipe([input_texts])[0].squeeze().tolist()

    outputs = pipe(input_texts)
    res = []
    for output in outputs:
        res.extend(output.squeeze().tolist())

    # print("res", res)con
    return res

def similarity_metrics(data, pipe=None, batch=1, num_batch=4):

    total_num_batch = (len(data)//8 )// (batch)
    similarity_metrics = np.zeros((len(data), len(data)))
    # similarity_metrics_doc = np.zeros((len(data), len(data)))
    for idx, line1 in tqdm(enumerate(data), desc='compute similarity...'):
        datas = []
        indices = []


        for idj in range(7*len(data)//8, len(data), batch):
        # for idj in range(0, len(data), batch):
            if len(datas) < num_batch:
                # print('if', idj, idj+batch, len(datas))
                datas.append(data[idj:idj+batch])
                indices.append(idj)
                if idj == 7*len(data)//8 +total_num_batch*batch:
                    result = compute_similarity(line1, datas, pipe=pipe)
                    length = len(result)
                    similarity_metrics[idx][indices[0]:indices[0] + length] = result[:]  # [1:batch+1]
                    # print('if ==', indices[0], indices[0] + batch, len(datas), length)
                    break

            else:
                result = compute_similarity(line1, datas, pipe=pipe)
                length = len(result)
                similarity_metrics[idx][indices[0]:indices[0] + length] = result[:]  # [1:batch+1]
                # print('else', indices[0], indices[0] + batch, len(datas), length)
                datas = []
                indices = []
                datas.append(data[idj:idj + batch])
                indices.append(idj)
                if idj == 7*len(data)//8+total_num_batch*batch:
                    result = compute_similarity(line1, datas, pipe=pipe)
                    length = len(result)
                    similarity_metrics[idx][indices[0]:indices[0] + length] = result[:]  # [1:batch+1]
                    # print('else ==', indices[0], indices[0] + batch, len(datas), length)
                    break

    np.save('./data/pipe_similarity_metrics_8.npy', similarity_metrics)
    # np.save('./data/pipe_similarity_metrics_doc.npy', similarity_metrics_doc)
    return similarity_metrics


if __name__ == "__main__":
    data = load_data('sce_topic_filtered.txt')
    # data = data[0:500]
    tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-mistral-7b-instruct', padding_side='left')
    PIPELINE_REGISTRY.register_pipeline(
        "text-similarity",
        pipeline_class=MyPipeline,
        pt_model=AutoModel
    )

    model = AutoModel.from_pretrained('intfloat/e5-mistral-7b-instruct', quantization_config=bnb_config, device_map='auto', trust_remote_code=True)
            # attn_implementation="flash_attention_2")
    # ort_model = ORTModel.from_pretrained(
    #     'intfloat/e5-mistral-7b-instruct',
    #     export=True,
    #     provider="CUDAExecutionProvider"
    # )
    pipe = pipeline("text-similarity", model=model, tokenizer=tokenizer, device_map='auto', trust_remote_code=True)

    similarity_metrics = similarity_metrics(data, pipe=pipe, batch=384, num_batch=len(data)//8//384+1) # 1024, 5