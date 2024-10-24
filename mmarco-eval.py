#import nltk
import datasets
from datasets import load_dataset
#from sklearn.feature_extraction.text import TfidfVectorizer
import random
import numpy as np
#from gensim import corpora, similarities
#from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer, AutoModel
import torch
import json
import sys

args = sys.argv

model = args[1]
nltk.download('punkt')

de = datasets.load_from_disk("./mmarco")
en = datasets.load_from_disk("./mmarco1")

df_de = de.select(range(10000))
df_en = en.select(range(10000))


questions = [data["query"] for data in df_de]
passages = [data["positive"] for data in df_en]
negatives = [data["negative"] for data in df_en]

with open("tfidf-mmarco.json", "r") as file:
    tfidf_candidate_passages = json.load(file)

with open("bm25-mmarco.json", "r") as file:
    bm25_candidate_passages = json.load(file)

with open("random-mmarco.json", "r") as file:
    random_candidate_passages = json.load(file)

tokenizer = AutoTokenizer.from_pretrained(model)
model = AutoModel.from_pretrained(model).to('cuda')


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def calculate_mrr(ranks):
    reciprocal_ranks = [1 / rank for rank in ranks if rank > 0]
    if not reciprocal_ranks:
        return 0.0  # No relevant answers found

    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    return mrr


print("Doing tFIDF ...")
tfidf_ranks = []

for question, correct_passage, tfidf_candidates in zip(questions, passages, tfidf_candidate_passages):
    #outputs = model(**inputs)
    question = f"<Q>{question}"
    if args[1] == 'castorini/mdpr-tied-pft-msmarco':
        encoded_input = tokenizer([question] + tfidf_candidates , padding=True, truncation=True, max_length=512, return_tensors='pt').to('cuda')
    else:
        encoded_input = tokenizer([question] + tfidf_candidates , padding=True, truncation=True, return_tensors='pt').to('cuda')

    with torch.no_grad():
        model_output = model(**encoded_input)

    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    question_embedding = sentence_embeddings[0]
    tf_similarities = []
    for candidate_embedding in sentence_embeddings[1:]:
        sim = torch.cosine_similarity(question_embedding, candidate_embedding, dim=0)
        tf_similarities.append(sim.item())

    ranked_candidates = sorted(zip(tfidf_candidates, tf_similarities), key=lambda x: x[1], reverse=True)
    ranked_cand = [x for x, _ in ranked_candidates]
    correct_idx = ranked_cand.index(f"<A>{correct_passage}")
    rank = correct_idx + 1

    tfidf_ranks.append(rank)

mrr_result = calculate_mrr(tfidf_ranks)
print(f"Mean Reciprocal Rank: {mrr_result:.2f}")

print("Doing BM25 ...")
bm25_ranks = []
count = 0
for question, correct_passage, bm25_candidates in zip(questions, passages, bm25_candidate_passages):
    question = f"<Q>{question}"
    if args[1] == 'castorini/mdpr-tied-pft-msmarco':
        encoded_input = tokenizer([question] + bm25_candidates , padding=True, truncation=True, max_length=512, return_tensors='pt').to('cuda')
    else:
        encoded_input = tokenizer([question] + bm25_candidates , padding=True, truncation=True, return_tensors='pt').to('cuda')

    with torch.no_grad():
        model_output = model(**encoded_input)

    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    question_embedding = sentence_embeddings[0]
    bm25_similarities = []
    for candidate_embedding in sentence_embeddings[1:]:
        sim = torch.cosine_similarity(question_embedding, candidate_embedding, dim=0)
        bm25_similarities.append(sim.item())

    ranked_candidates = sorted(zip(bm25_candidates, bm25_similarities), key=lambda x: x[1], reverse=True)
    ranked_cand = [x for x, _ in ranked_candidates]
    correct_idx = ranked_cand.index(f"<A>{correct_passage}")
    rank = correct_idx + 1

    bm25_ranks.append(rank)

mrr_result = calculate_mrr(bm25_ranks)
print(f"Mean Reciprocal Rank: {mrr_result:.2f}")

print("Doing random ...")
random_ranks = []
count = 0
for question, correct_passage, random_candidates in zip(questions, passages, random_candidate_passages):
    question = f"<Q>{question}"
    if args[1] == 'castorini/mdpr-tied-pft-msmarco':
        encoded_input = tokenizer([question] + random_candidates , padding=True, truncation=True, max_length=512, return_tensors='pt').to('cuda')
    else:
        encoded_input = tokenizer([question] + random_candidates , padding=True, truncation=True, return_tensors='pt').to('cuda')

    with torch.no_grad():
        model_output = model(**encoded_input)

    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    question_embedding = sentence_embeddings[0]
    random_similarities = []
    for candidate_embedding in sentence_embeddings[1:]:
        sim = torch.cosine_similarity(question_embedding, candidate_embedding, dim=0)
        random_similarities.append(sim.item())

    ranked_candidates = sorted(zip(random_candidates, random_similarities), key=lambda x: x[1], reverse=True)
    ranked_cand = [x for x, _ in ranked_candidates]
    correct_idx = ranked_cand.index(f"<A>{correct_passage}")
    rank = correct_idx + 1

    random_ranks.append(rank)


mrr_result = calculate_mrr(random_ranks)
print(f"Mean Reciprocal Rank: {mrr_result:.2f}")