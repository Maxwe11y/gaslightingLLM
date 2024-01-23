import torch
import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel, pipeline
import os
import numpy as np
from tqdm import tqdm

def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'


def compute_similarity(line1=None, line2=None, tokenizer=None, model=None, device='cpu'):

    # Each query must come with a one-sentence instruction that describes the task
    task = 'Given a conversation scenario, retrieve a text that is similar to the given scenario from the aspect of topic, semantics and theme.'
    # input_texts = [get_detailed_instruct(task, line1), line2]
    input_texts = [get_detailed_instruct(task, line1)]
    input_texts.extend(line2)
    # input_texts = [get_detailed_instruct(task, line1)]
    # input_texts = [line1]

    max_length = 4096
    # Tokenize the input texts
    batch_dict = tokenizer(input_texts, max_length=max_length - 1, return_attention_mask=False, padding=False, truncation=True)
    # append eos_token_id to every input_ids
    batch_dict['input_ids'] = [input_ids + [tokenizer.eos_token_id] for input_ids in batch_dict['input_ids']]
    batch_dict = tokenizer.pad(batch_dict, padding=True, return_attention_mask=True, return_tensors='pt').to(device)

    # outputs = model(input_texts, add_eos_token=True, padding=True)
    outputs = model(**batch_dict)

    embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

    # normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    # scores = (embeddings[:1] @ embeddings[1:].T) * 100
    scores = embeddings[:1] @ embeddings[1:].T
    # print(scores.tolist())
    return scores.data#tolist()

def load_data(filename, path="./data"):
    data = []
    with open(os.path.join(path, filename), "r") as f:
        lines = f.readlines()
        for line in lines:
            # data[line.strip().split('\t')[0]] = line.strip().split('\t')[1]
            data.append(line.strip().split('\t')[0])

    return data


def similarity_metrics(data, tokenizer=None, model=None, device='cpu'):
    similarity_metrics = np.zeros((len(data), len(data)))
    similarity_metrics_doc = np.zeros((len(data), len(data)))
    for idx, line1 in tqdm(enumerate(data), desc='compute similarity...'):
        for idj, line2 in enumerate(data):
            result = compute_similarity("None", ["None"], tokenizer=tokenizer, model=model, device=device)
            if idj > idx:
                similarity_metrics[idx][idj] = result
            elif idj < idx:
                similarity_metrics_doc[idx][idj] = result
    np.save('./data/similarity_metrics.npy', similarity_metrics)
    np.save('./data/similarity_metrics_doc.npy', similarity_metrics_doc)
    return result

# def similarity_metrics(data, tokenizer=None, model=None, batch=1, device='cpu'):
#     similarity_metrics = np.zeros((len(data), len(data)))
#     similarity_metrics_doc = np.zeros((len(data), len(data)))
#     for idx, line1 in tqdm(enumerate(data), desc='compute similarity...'):
#         for idj in range(0, len(data), batch):
#             result = compute_similarity(line1, data[idj:idj+batch], tokenizer=tokenizer, model=model, device=device)
#             if idj > idx:
#                 similarity_metrics[idx][idj:idj+batch] = result#[1:batch+1]
#             elif idj < idx:
#                 similarity_metrics_doc[idx][idj:idj+batch] = result#[1:batch+1]
#     np.save('./data/similarity_metrics.npy', similarity_metrics)
#     np.save('./data/similarity_metrics_doc.npy', similarity_metrics_doc)
#     return similarity_metrics, similarity_metrics_doc


if __name__ == '__main__':

    data = load_data('sce_topic_filtered.txt')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = 'cpu'
    tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-mistral-7b-instruct')
    model = AutoModel.from_pretrained('intfloat/e5-mistral-7b-instruct', device_map='auto')
    model.to(device)
    # pipe= pipeline(model='intfloat/e5-mistral-7b-instruct', tokenizer=tokenizer)
    result = similarity_metrics(data, model=model, tokenizer=tokenizer, device=device)



