# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import torch
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import nltk


# if it doesnt work
# !pip install nltk

model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)

# +
script_path = "shrek.txt"

with open(script_path, 'r', encoding='utf-8') as file:
    script_text = file.read()
import re
script_text = re.sub(' +', ' ', script_text)


# -

def create_fixed_size_chunks(text, chunk_size=1000, overlap=0):
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(start+chunk_size, text_length)
        if start>0 and overlap >0:
            start = start - overlap
        chunks.append(text[start:end])
        start = end
    return chunks


from nltk.tokenize import sent_tokenize
#nltk.data.find('tokenizers/punkt')
nltk.download('punkt_tab')


def create_sentence_chunks(text, sentences_per_chunk=10):
    sentences = sent_tokenize(text)
    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = " ".join(sentences[i:i+sentences_per_chunk])
        chunks.append(chunk)
    return chunks



chunks_sentences = create_sentence_chunks(script_text, sentences_per_chunk=15)

chunks_fixed_size = create_fixed_size_chunks(script_text)
chunks_fixed_size_overlapping = create_fixed_size_chunks(script_text, overlap=50)


def get_embeddings(texts, tokenizer, model):
    encoded_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**encoded_inputs)
    token_embeddings = outputs.last_hidden_state
    embeddings = torch.mean(token_embeddings, dim=1)
    return embeddings.cpu().numpy()


embeddings_fixed_size = get_embeddings(chunks_fixed_size, tokenizer, model)
embeddings_fixed_size_overlapping = get_embeddings(chunks_fixed_size_overlapping, tokenizer, model)
embeddings_sentences= get_embeddings(chunks_sentences, tokenizer, model)

cosine_similarity = lambda a,b: np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
dot_product_similarity = lambda a,b: a@b
euclidean_similarity = lambda a,b: 1/(1+ np.linalg.norm(a - b))


def retrieve_chunks(query_embedding, chunk_embeddings, chunks, top_k=3, similarity_fn=cosine_similarity):
    similarities = []
    for i, chunk_embedding in enumerate(chunk_embeddings):
        similarity = similarity_fn(query_embedding, chunk_embedding)
        similarities.append((i, similarity, chunks[i]))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]


# +
sample_query = "Who is farquads wife."+" DONKEY"

query_embedding = get_embeddings([sample_query], tokenizer, model)[0]
em = embeddings_fixed_size_overlapping
ch = chunks_fixed_size_overlapping
retrieve_chunks(query_embedding, em, ch, top_k=3, similarity_fn=cosine_similarity)
# -


