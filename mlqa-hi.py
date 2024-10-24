#import nltk
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
#nltk.download('punkt')

dataset = load_dataset("mlqa", "mlqa.hi.en")
df = load_dataset("mlqa", "mlqa.en.en")

dt = dataset["test"]
en = df["test"]

questions = [data["question"] for data in dt]
passages = [data["context"] for data in dt]
ids = [data["id"] for data in dt]

#tFIDF
"""
# Fit the TF-IDF vectorizer on the passages
vectorizer = TfidfVectorizer()
passage_vectors = vectorizer.fit_transform(passages)

# For each question, rank the passages based on cosine similarity
tfidf_candidate_passages = []
for question, correct_passage, id in zip(questions, passages, ids):
    indices = np.where(np.array(en['id']) == id)[0]
    fil = en.select(indices.tolist())
    question = fil['question']

    question_vector = vectorizer.transform(question)
    scores = passage_vectors.dot(question_vector.T).toarray().ravel()
    ranked_passages = sorted(zip(passages, scores), key=lambda x: x[1], reverse=True)

    # Select top 4 candidates, excluding the correct passage
    #while(len(candidates) < 5)
    candidates = [f"<A>{p}" for p, s in ranked_passages if p != correct_passage][:9]
    # Add the correct passage to the candidates
    candidates.append(f"<A>{correct_passage}")
    random.shuffle(candidates)  # Shuffle the candidates
    tfidf_candidate_passages.append(candidates)
    

tokenized_passages = [word_tokenize(passage) for passage in passages]
# Create a corpus and dictionary from the passages
dictionary = corpora.Dictionary(tokenized_passages)
corpus = [dictionary.doc2bow(word_tokenize(passage)) for passage in passages]

# Create a BM25 similarity index
bm25 = similarities.docsim.Similarity('./mlqa', corpus, num_features=len(dictionary), num_best=10)

# For each question, rank the passages based on BM25 scores
bm25_candidate_passages = []
for question, correct_passage, id in zip(questions, passages, ids):
    indices = np.where(np.array(en['id']) == id)[0]
    fil = en.select(indices.tolist())
    question = fil['question'][0]
    query_bow = dictionary.doc2bow(word_tokenize(question))
    scores = bm25[query_bow]
    ranked_passages = sorted(zip(passages, scores), key=lambda x: x[1], reverse=True)

    # Select top 4 candidates, excluding the correct passage
    candidates = [f"<A>{p}" for p, s in ranked_passages if p != correct_passage][:4]

    # Add the correct passage to the candidates
    candidates.append(f"<A>{correct_passage}")
    random.shuffle(candidates)  # Shuffle the candidates
    bm25_candidate_passages.append(candidates)

#Random
random_candidate_passages = []
for question, correct_passage in zip(questions, passages):
    candidates = passages.copy()
    candidates.remove(correct_passage)
    random_candidates = random.sample(candidates, 4)  # Select 4 random candidates
    random_candidates = [f"<A>{p}" for p in random_candidates]
    random_candidates.append(f"<A>{correct_passage}")  # Add the correct passage
    random.shuffle(random_candidates)  # Shuffle the candidates
    random_candidate_passages.append(random_candidates)
"""
with open("tfidf-hi.json", "r") as file:
    tfidf_candidate_passages = json.load(file)

with open("bm25-hi.json", "r") as file:
    bm25_candidate_passages = json.load(file)

with open("randomed-hi.json", "r") as file:
    random_candidate_passages = json.load(file)

#tFIDF
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
    #inputs = tokenizer(question, bm25_candidates, return_tensors="pt", padding=True, truncation=True)
    #inputs = inputs#.to('cuda')
    #outputs = model(**inputs)
    question = f"<Q>{question}"
    encoded_input = tokenizer([question] + tfidf_candidates , padding=True, truncation=True, max_length=512, return_tensors='pt').to('cuda')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling. In this case, max pooling.
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    question_embedding = sentence_embeddings[0]
    tf_similarities = []
    for candidate_embedding in sentence_embeddings[1:]:
        sim = torch.cosine_similarity(question_embedding, candidate_embedding, dim=0)
        tf_similarities.append(sim.item())

    ranked_candidates = sorted(zip(tfidf_candidates, tf_similarities), key=lambda x: x[1], reverse=True)
    ranked_cand = [x for x, _ in ranked_candidates]
    correct_idx = ranked_cand.index(f"<A>{correct_passage}")
    rank = correct_idx + 1  # Rank starts from 1

    tfidf_ranks.append(rank)

mrr_result = calculate_mrr(tfidf_ranks)
print(f"Mean Reciprocal Rank: {mrr_result:.2f}")

print("Doing BM25 ...")
#bm25
bm25_ranks = []
count = 0
for question, correct_passage, bm25_candidates in zip(questions, passages, bm25_candidate_passages):
    #inputs = tokenizer(question, bm25_candidates, return_tensors="pt", padding=True, truncation=True)
    #inputs = inputs#.to('cuda')
    #outputs = model(**inputs)
    question = f"<Q>{question}"
    encoded_input = tokenizer([question] + bm25_candidates , padding=True, truncation=True, max_length=512, return_tensors='pt').to('cuda')
    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling. In this case, max pooling.
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    question_embedding = sentence_embeddings[0]
    bm25_similarities = []
    for candidate_embedding in sentence_embeddings[1:]:
        sim = torch.cosine_similarity(question_embedding, candidate_embedding, dim=0)
        bm25_similarities.append(sim.item())

    ranked_candidates = sorted(zip(bm25_candidates, bm25_similarities), key=lambda x: x[1], reverse=True)
    ranked_cand = [x for x, _ in ranked_candidates]
    correct_idx = ranked_cand.index(f"<A>{correct_passage}")
    rank = correct_idx + 1  # Rank starts from 1

    bm25_ranks.append(rank)

mrr_result = calculate_mrr(bm25_ranks)
print(f"Mean Reciprocal Rank: {mrr_result:.2f}")


print("Doing random ...")
#random
random_ranks = []
count = 0
for question, correct_passage, random_candidates in zip(questions, passages, random_candidate_passages):
    #inputs = tokenizer(question, bm25_candidates, return_tensors="pt", padding=True, truncation=True)
    #inputs = inputs#.to('cuda')
    #outputs = model(**inputs)
    question = f"<Q>{question}"
    encoded_input = tokenizer([question] + random_candidates , padding=True, truncation=True, max_length=512, return_tensors='pt').to('cuda')
    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling. In this case, max pooling.
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    question_embedding = sentence_embeddings[0]
    random_similarities = []
    for candidate_embedding in sentence_embeddings[1:]:
        sim = torch.cosine_similarity(question_embedding, candidate_embedding, dim=0)
        random_similarities.append(sim.item())

    ranked_candidates = sorted(zip(random_candidates, random_similarities), key=lambda x: x[1], reverse=True)
    ranked_cand = [x for x, _ in ranked_candidates]
    correct_idx = ranked_cand.index(f"<A>{correct_passage}")
    rank = correct_idx + 1  # Rank starts from 1

    random_ranks.append(rank)


mrr_result = calculate_mrr(random_ranks)
print(f"Mean Reciprocal Rank: {mrr_result:.2f}")

