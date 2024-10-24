import datasets
from datasets import load_dataset
import json
import sys
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')
# args = sys.argv

# lang = args[1]
# de = load_dataset('unicamp-dl/mmarco', 'spanish', split='train')
# print(len(de))
# de.save_to_disk("./mmarco-es")

# en = load_dataset('unicamp-dl/mmarco', 'hindi', split='train')
# print(len(en))
# en.save_to_disk("./mmarco-hi")

# Creating EN-VI
de = datasets.load_from_disk("./mmarco")
en = datasets.load_from_disk("./mmarco-vi")

df_de = de.select(range(10000))
df_en = en.select(range(10000))


questions = [data["query"] for data in df_de]
passages = [data["positive"] for data in df_de]

negatives = [data["negative"] for data in df_en]
uniq = set(questions)
print(len(uniq))
flag = 0
while(flag < 5):
    q = questions[flag]
    p = passages[flag]
    print(f"<Q> {q}")
    print(f"<A> {p}")
    flag = flag + 1

vectorizer = TfidfVectorizer()
passage_vectors = vectorizer.fit_transform(passages)

# GPT-3.5 for creating evaluation sets of 10
print("Doing tfidf")
tfidf_candidate_passages = []
for question, correct_passage, negative in zip(questions, passages, negatives):
    question_vector = vectorizer.transform([question])
    scores = passage_vectors.dot(question_vector.T).toarray().ravel()
    ranked_passages = sorted(zip(passages, scores), key=lambda x: x[1], reverse=True)

    candidates = [f"<A>{p}" for p, s in ranked_passages if p != correct_passage][:8]

    # Adding the correct passage to the candidates
    candidates.append(f"<A>{correct_passage}")
    candidates.append(f"<A>{negative}")
    random.shuffle(candidates)
    tfidf_candidate_passages.append(candidates)

print("Doing BM25")
tokenized_passages = [word_tokenize(passage.lower()) for passage in passages]
bm25 = BM25Okapi(tokenized_passages)

# For each question, ranking the passages based on BM25 scores
bm25_candidate_passages = []
for question, correct_passage, negative in zip(questions, passages, negatives):
    tokenized_question = word_tokenize(question.lower())
    # Get BM25 scores for the question against all passages
    scores = bm25.get_scores(tokenized_question)

    ranked_passages_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

    candidates = []
    for idx, score in ranked_passages_scores:
        if len(candidates) >= 8:
            break
        if passages[idx] != correct_passage:
            candidates.append(f"<A>{passages[idx]}")

    # Adding the correct passage to the candidates
    candidates.append(f"<A>{correct_passage}")
    candidates.append(f"<A>{negative}")
    random.shuffle(candidates)

    bm25_candidate_passages.append(candidates)

print("Doing Random")
random_candidate_passages = []
for question, correct_passage in zip(questions, passages):
    candidates = passages.copy()
    candidates.remove(correct_passage)
    random_candidates = random.sample(candidates, 8)
    random_candidates = [f"<A>{p}" for p in random_candidates]
    random_candidates.append(f"<A>{correct_passage}")
    random_candidates.append(f"<A>{negative}")
    random.shuffle(random_candidates)
    random_candidate_passages.append(random_candidates)

with open("tfidf-mmarco-hi.json", "w") as file:
    json.dump(tfidf_candidate_passages, file)

with open("bm25-mmarco-hi.json", "w") as file:
    json.dump(bm25_candidate_passages, file)

with open("random-mmarco-hi.json", "w") as file:
    json.dump(random_candidate_passages, file)

