# Creating small batches of domains

from datasets import load_dataset
import tldextract
import pandas as pd
import re
import numpy as np
from collections import defaultdict
from transformers import BertTokenizerFast, BertModel
from transformers import AutoModel, AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
import polars as pl
from datasets import load_dataset, interleave_datasets, Dataset, load_from_disk
#from laserembeddings import Laser
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import spacy
import fasttext
import sys
from cleantext import clean
import tldextract
from rapidfuzz import fuzz
import spacy
import deep_translator
from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer
import wandb
from flair.data import Sentence
from flair.models import SequenceTagger
import fasttext
import faiss

args = sys.argv
# load tagger
tagger = SequenceTagger.load("ner-large")

lang1 = args[1]#'es'
lang2 = args[2]#'hr'
print(lang1)
print(lang2)
#spacy1 = args[3]
#spacy2 = args[4]

en_dataset = load_dataset("clips/mfaq", lang1)
hr_dataset = load_dataset("clips/mfaq", lang2)

#laser = Laser()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
labse = SentenceTransformer('sentence-transformers/LaBSE').to(device)

# target_lang = args[4]
# input_lang = args[3]

def margin_score_vectorized(x, y, x_neighbors, y_neighbors, k):
    cos_xy = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

    sum_x = np.sum(np.dot(x, x_neighbors.T) / (np.linalg.norm(x) * np.linalg.norm(x_neighbors, axis=1))) / (2 * k)
    sum_y = np.sum(np.dot(y, y_neighbors.T) / (np.linalg.norm(y) * np.linalg.norm(y_neighbors, axis=1))) / (2 * k)

    return cos_xy / (sum_x + sum_y)

def check_org_fuzz(sent, ent):
    score = fuzz.partial_ratio(ent, sent)
    return score >= 80

def cleaner(text):
    return clean(text, lower=False,no_currency_symbols=True, no_punct = False, replace_with_currency_symbol='')
  
def check_fuzz(sent, ent):
    score = fuzz.partial_ratio(ent, sent)
    return score >= 87
    
def check_ner(sent, target, ners1, ners2, lang1, lang2):
    if ners2:
        for ner in ners1:
            try:
                ner_trans = GoogleTranslator(source=lang1, target=lang2).translate(text=ner)
            except deep_translator.exceptions.TranslationNotFound:
                ner_trans = ner
            if(check_fuzz(target, ner_trans) or check_fuzz(target, ner)):
                continue
            else:
                return False
    return True

def dict_of_lists():
    return defaultdict(list)
    
def dict_of_lists_of_lists():
    return defaultdict(lambda: defaultdict(list))

def extract_root_domain(domain_url):
        return tldextract.extract(domain_url).domain

def check_items_in_sentence(sentence, items):
    return all(item in sentence for item in items)

def check_org(sentence1, sentence2, items, target, lang1, lang2):
    if target:
        for org in items:
            if check_org_fuzz(sentence2.lower(), org.lower()):
                continue
            else:
                return False
        return True

en_domains = list(map(extract_root_domain, en_dataset['train']['domain']))
en_domains = list(set(en_domains))  # to get unique domains
domain_datasets = {}

hr_domains = list(map(extract_root_domain, hr_dataset['train']['domain']))
hr_domains = list(set(hr_domains))  # to get unique domains

en_hr = [x for x in en_domains if x in hr_domains]

print("Filtering datasets")
hr_filtered = hr_dataset['train'].filter(lambda example: extract_root_domain(example['domain']) in en_hr)
en_filtered = en_dataset['train'].filter(lambda example: extract_root_domain(example['domain']) in en_hr)


arr_en = pl.from_arrow(en_filtered.data.table)
arr_hr = pl.from_arrow(hr_filtered.data.table)


en_sent = {}
en_ans = {}
en_dom = {}
en_id = {}
hr_sent = {}
hr_ans = {}
hr_dom = {}
hr_id = {}

ndom = len(en_hr)
d1 = 20
quo = ndom // d1
scores = defaultdict(dict_of_lists_of_lists)
k = 5

rem = ndom
l1 = 0
l2 = 0

s1 = float(args[3])
s2 = 1.0 - s1
threshold = float(args[4])

print("All matched domains: ", ndom)
for ist in range(quo+1):
    if(ist == 0 ):
        l1 = 0
        l2 = d1
    else:
        if(rem // d1 > 0):
            l1 = l1 + d1
            l2 = l2 + d1
        else:
            l1 = l1 + d1
            l2 = l2 + (rem % d1)
        if(l2 == 0):
            break
    if(l1 == l2):
        break
    rem = rem - d1
    
    data_name = lang1 + '_' + lang2 + '_flair12_'+ str(ist+1)
    name_data = lang2 + '_' + lang1 + '_flair12_'+ str(ist+1)

    fil_doms = en_hr[l1:l2]
    
    print("Finding alignments...")
    for d in fil_doms:
        en_dataset_filtered = arr_en.filter(pl.col('domain').apply(extract_root_domain) == d)
        hr_dataset_filtered = arr_hr.filter(pl.col('domain').apply(extract_root_domain) == d)
    
        en_sent[d] = [cleaner(pair["question"]) for page in en_dataset_filtered.rows(named=True) for pair in page["qa_pairs"]]
        en_ans[d] = [cleaner(pair["answer"]) for page in en_dataset_filtered.rows(named=True) for pair in page["qa_pairs"]]
        en_dom[d] = [page["domain"] for page in en_dataset_filtered.rows(named=True) for pair in page["qa_pairs"]]
        en_id[d] = [page["id"] for page in en_dataset_filtered.rows(named=True) for pair in page["qa_pairs"]]
        hr_sent[d] = [cleaner(pair["question"]) for page in hr_dataset_filtered.rows(named=True) for pair in page["qa_pairs"]]
        hr_ans[d] = [cleaner(pair["answer"]) for page in hr_dataset_filtered.rows(named=True) for pair in page["qa_pairs"]]
        hr_dom[d] = [page["domain"] for page in hr_dataset_filtered.rows(named=True) for pair in page["qa_pairs"]]
        hr_id[d] = [page["id"] for page in hr_dataset_filtered.rows(named=True) for pair in page["qa_pairs"]]
    
    
    aligned_sentences = []
    maligned_sentences = []
    
    print("Creating datasets...")
    for d in fil_doms:
        print(d)
        en_sentences = []
        en_answers = []
        en_ids = []
        en_doms = []
        hr_sentences = []
        hr_answers = []
        hr_ids = []
        hr_doms = []
    
        en_sentences = en_sent[d]
        en_answers = en_ans[d]
        en_doms = en_dom[d]
        en_ids = en_id[d]
        hr_sentences = hr_sent[d]
        hr_answers = hr_ans[d]
        hr_doms = hr_dom[d]
        hr_ids = hr_id[d]

        print(len(en_sentences))
    
        if en_sentences and hr_sentences:
            en_embeddings = labse.encode(en_sentences)
            hr_embeddings = labse.encode(hr_sentences)

            # Use FAISS for faster nearest neighbor search
            d = en_embeddings.shape[1]  # dimension of embeddings
            index_en = faiss.IndexFlatIP(d)
            index_hr = faiss.IndexFlatIP(d)

            index_en.add(en_embeddings.astype('float32'))
            index_hr.add(hr_embeddings.astype('float32'))

            # Find k-nearest neighbors for all embeddings at once
            _, en_nn_indices = index_hr.search(en_embeddings.astype('float32'), k)
            _, hr_nn_indices = index_en.search(hr_embeddings.astype('float32'), k)

            for i in range(len(en_embeddings)):
                if(i >= len(en_sentences)):
                    break
                x = en_embeddings[i]
                x_neighbors = hr_embeddings[en_nn_indices[i]]

                for j in range(len(hr_embeddings)):
                    if(j >= len(hr_sentences)):
                        break
                    y = hr_embeddings[j]
                    y_neighbors = en_embeddings[hr_nn_indices[j]]
                    q_score = margin_score_vectorized(x, y, x_neighbors, y_neighbors, k)

                    if(j >= len(hr_sentences)):
                        break
                    if(q_score >= 1.0):
                        en_a = labse.encode(en_answers[i])
                        hr_a = labse.encode(hr_answers[j])
                        ans_score = np.dot(en_a, hr_a)
                        aggr = (ans_score * s1) + (q_score * s2)

                        if aggr >= threshold:
                            sentence1 = Sentence(en_answers[i])
                            sentence2 = Sentence(hr_answers[j])

                            tagger.predict(sentence1)
                            ner1 = [entity.text for entity in sentence1.get_spans('ner') if entity.tag == "LOC"]
                            org1 = [entity.text for entity in sentence1.get_spans('ner') if entity.tag == "ORG" and entity.score > 0.80]

                            tagger.predict(sentence2)
                            ner2 = [entity.text for entity in sentence2.get_spans('ner') if entity.tag == "LOC"]
                            org2 = [entity.text for entity in sentence2.get_spans('ner') if entity.tag == "ORG" and entity.score > 0.80]
                            numbers = re.findall(r'\d+', en_answers[i])

                            if check_items_in_sentence(hr_answers[j], numbers):
                                if (check_org(en_answers[i], hr_answers[j], org1, org2, lang1, lang2) and check_ner(en_answers[i], hr_answers[j], ner1, ner2, lang1, lang2)):
                                    aligned_sentences.append((hr_ids[j], hr_doms[j], en_sentences[i], hr_answers[j]))
                                    maligned_sentences.append((en_ids[i], en_doms[i], hr_sentences[j], en_answers[i]))
            else:
                continue

    df = pd.DataFrame(aligned_sentences, columns=['id','domain', 'question','answer'])
    df['qa_pairs'] = df.apply(lambda row: [{'question': row['question'], 'answer': row['answer']}], axis=1)
    # group by A and B, and combine the dictionaries in CD
    df_grouped = df.groupby(['id', 'domain'])['qa_pairs'].apply(sum).reset_index()
    df_grouped.head()
    para_data = Dataset.from_pandas(df_grouped)
    para_data = para_data.train_test_split(test_size=0.1)
    para_data.save_to_disk(data_name)
    
    # other XL pair
    df1 = pd.DataFrame(maligned_sentences, columns=['id','domain', 'question','answer'])
    df1['qa_pairs'] = df1.apply(lambda row: [{'question': row['question'], 'answer': row['answer']}], axis=1)
    # group by A and B, and combine the dictionaries in CD
    df_grouped1 = df1.groupby(['id', 'domain'])['qa_pairs'].apply(sum).reset_index()
    df_grouped1.head()
    para_data1 = Dataset.from_pandas(df_grouped1)
    para_data1 = para_data1.train_test_split(test_size=0.1)
    para_data1.save_to_disk(name_data)
    