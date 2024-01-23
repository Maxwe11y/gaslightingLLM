# from openai.embeddings_utils import get_embedding, cosine_similarity
from get_mistral_embedding import load_data
import pandas as pd
import numpy as np
import time
from tqdm import tqdm, tqdm_pandas
import os
import torch
import torch.nn.functional as F

from openai import OpenAI
client = OpenAI(api_key='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')

def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")

   return client.embeddings.create(input=[text], model=model).data[0].embedding


# def search_reviews(df, product_description, n=3, pprint=True):
#    embedding = get_embedding(product_description, model='text-embedding-ada-002')
#    df['similarities'] = df.ada_embedding.apply(lambda x: cosine_similarity(x, embedding))
#    res = df.sort_values('similarities', ascending=False).head(n)
#    return res


def temp_sleep(seconds=0.1):
    time.sleep(seconds)


def Embedding_request(text, model):
    """
    Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
    server and returns the response.
    ARGS:
      messages: messages=[{"role": "system", "content": prompt}]
      gpt_parameter: a python dictionary with the keys indicating the names of
                     the parameter and the values indicating the parameter
                     values.
    RETURNS:
      a str of GPT-3's response.
    """
    temp_sleep()
    try:
        completion = client.embeddings.create(input=[text], model=model)
        return completion.data[0].embedding

    except:
        print("Openai ERROR: {}".format(text))
        return "Openai ERROR"


def safe_embedding_response(messages, model='text-embedding-ada-002', repeat=5):

    for i in range(repeat):

        try:
            curr_gpt_response = Embedding_request(messages, model=model)
            if curr_gpt_response != "Openai ERROR":
                return curr_gpt_response
            else:
                continue

        except:
            pass

def load_csv(filename):
    data = pd.read_csv(filename)
    embeddings = data['ada_embedding']
    return embeddings

def process_nan(filename):
    data = pd.read_csv(filename)
    for i in range(len(data)):
        if not isinstance(data.loc[i]['ada_embedding'], str):
            if np.isnan(data.loc[i]['ada_embedding']):
                sentence = data.loc[i]['sentence']
                embed = safe_embedding_response(sentence, model='text-embedding-ada-002', repeat=5)
                data.loc[i]['ada_embedding'] = embed

    data.to_csv('data/ada_embedding_full.csv', index=False)

    return data

def compute_similarity(embeddings):
    length = len(embeddings)
    similarity_metrics = np.zeros((length, length))
    for idx, embed in enumerate(embeddings):
        for idj in range(idx+1, length):
            embed_1 = np.array(eval(embed))
            embed_2 = np.array(eval(embeddings[idj]))
            cos_sim = embed_1.dot(embed_2) / (np.linalg.norm(embed_1) * np.linalg.norm(embed_2))
            similarity_metrics[idx][idj] = cos_sim
            similarity_metrics[idj][idx] = cos_sim

    np.save('./data/ada_similarity.npy', similarity_metrics)
    return similarity_metrics


def calculate_similarity_matrix(embeddings):
    """
    Calculate cosine similarity matrix for a list of embeddings using PyTorch.

    Parameters:
    - embeddings: List of embedding vectors.

    Returns:
    - similarity_matrix: Cosine similarity matrix.
    """
    # Convert the list of embeddings to a PyTorch tensor
    embeddings_ = []
    for idx, embed in enumerate(embeddings):
            embed = np.array(eval(embed))
            embeddings_.append(embed)

    embeddings_tensor = torch.tensor(embeddings_, dtype=torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embeddings_tensor = embeddings_tensor.to(device)

    # # Normalize the embeddings
    # normalized_embeddings = F.normalize(embeddings_tensor, p=2, dim=1)

    # Calculate cosine similarity matrix
    similarity_matrix = torch.mm(embeddings_tensor, embeddings_tensor.t())

    # Convert the PyTorch tensor to a NumPy array
    similarity_matrix_np = similarity_matrix.cpu().numpy()
    np.save('./data/ada_similarity.npy', similarity_matrix_np)
    return similarity_matrix_np



def calculate_similarity_mistral(embeddings):
    """
    Calculate cosine similarity matrix for a list of embeddings using PyTorch.

    Parameters:
    - embeddings: List of embedding vectors.

    Returns:
    - similarity_matrix: Cosine similarity matrix.
    """
    # Convert the list of embeddings to a PyTorch tensor
    embeddings_1 = torch.load('./data/line_1.pth')
    embeddings_2 = torch.load('./data/line_2.pth')

    # embeddings_tensor = torch.tensor(embeddings_, dtype=torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embeddings_1 = embeddings_1.to(device)
    embeddings_2 = embeddings_2.to(device)

    # # Normalize the embeddings
    # normalized_embeddings = F.normalize(embeddings_tensor, p=2, dim=1)

    # Calculate cosine similarity matrix
    similarity_matrix = torch.mm(embeddings_1, embeddings_2.t())

    # Convert the PyTorch tensor to a NumPy array
    similarity_matrix_np = similarity_matrix.cpu().numpy()
    np.save('./data/mistral_similarity.npy', similarity_matrix_np)
    return similarity_matrix_np


def check_data(filename, path="./data"):
    data = []
    with open(os.path.join(path, filename), "r") as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            newline = line.strip().split('\t')
            if len(newline) != 2 or newline[0]== '' or newline[1]=='':
                print(idx)

    return data


if __name__ == "__main__":

   # data = load_data('sce_topic_filtered.txt')
   # df = pd.DataFrame(data, columns=['sentence'])
   #
   # df['ada_embedding'] = df.sentence.apply(lambda x: safe_embedding_response(x, model='text-embedding-ada-002'))
   # df.to_csv('data/ada_embedding.csv', index=False)

   embeddings = load_csv('./data/ada_embedding.csv')
   # res = compute_similarity(embeddings)
   # res = calculate_similarity_matrix(embeddings)
   calculate_similarity_mistral(embeddings)
   #
   print('done!')

   # data = process_nan('./data/ada_embedding.csv')
   # res = check_data('sce_topic_filtered.txt')
   print('done!')