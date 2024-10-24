from datasets import load_dataset
import tldextract
import pandas as pd
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
import fasttext

args = sys.argv

lang1 = args[1]#'es'
lang2 = args[2]#'hr'
spacy1 = args[3]
spacy2 = args[4]

en_dataset = load_dataset("clips/mfaq", lang1)
hr_dataset = load_dataset("clips/mfaq", lang2)

nlp_en = spacy.load(spacy1)
nlp_hr = spacy.load(spacy2)
#laser = Laser()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
labse = SentenceTransformer('sentence-transformers/LaBSE').to(device)

target_lang = args[6]
input_lang = args[5]

def check_fuzz(sent, ent):
  score = fuzz.partial_ratio(ent, sent)
  return score >= 87
    
def check_ner(sent, target, ners1, ners2):
  if ners2:
    for ner in ners1:
      try:
        ner_trans = GoogleTranslator(source=lang1, target=lang2).translate(text=ner)
        if(check_fuzz(target, ner_trans) or check_fuzz(target, ner)):
          continue
        else:
          return False
      except deep_translator.exceptions.TranslationNotFound:
        pass
  return True

def cleaner(text):
    return clean(text, lower=False,no_currency_symbols=True, no_punct = False, replace_with_currency_symbol='')

def dict_of_lists():
    return defaultdict(list)

def extract_root_domain(domain_url):
    return tldextract.extract(domain_url).domain

def check_items_in_sentence(sentence, items):
  return all(item in sentence for item in items)

def check_org(sentence, items, target):
    if target and items:
        return all(item.lower() in sentence.lower() for item in items)
    return True

en_domains = list(map(extract_root_domain, en_dataset['train']['domain']))
en_domains = list(set(en_domains))
domain_datasets = {}

hr_domains = list(map(extract_root_domain, hr_dataset['train']['domain']))
hr_domains = list(set(hr_domains))

en_hr = [x for x in en_domains if x in hr_domains]

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
d1 = 30
quo = ndom // d1

rem = ndom
l1 = 0
l2 = 0

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
    
    data_name = lang1 + '_' + lang2 + '_labse1_'+ str(ist+1)
    name_data = lang2 + '_' + lang1 + '_labse1_'+ str(ist+1)

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
        
        s1 = float(args[7])
        s2 = 1.0 - s1
        threshold = float(args[8])
    
        if en_sentences and hr_sentences:
            en_embeddings = labse.encode(en_sentences)
            hr_embeddings = labse.encode(hr_sentences)
                
            # Compute cosine similarities between English and Croatian sentence embeddings
            similarities = cosine_similarity(en_embeddings, hr_embeddings)
    
            # Set the similarity threshold
            similarity_threshold = 0.7
    
            for i in range(len(en_sentences)):
                above_threshold_q = np.where(similarities[i] >= similarity_threshold)[0]
                top_candidates = sorted(above_threshold_q, key=lambda j: similarities[i, j], reverse=True)[:2]
                subset_hr_answers = [hr_answers[k] for k in top_candidates]
                if subset_hr_answers:
                    en_a = labse.encode([en_answers[i]])
                    hr_a = labse.encode(subset_hr_answers)
                    
                    answer_similarity = cosine_similarity(en_a, hr_a)
                    
                    for idx, k in enumerate(top_candidates):
                        idx = top_candidates.index(k)
                        
                        aggr = (answer_similarity[0][idx]*s1)+(similarities[i,k]*s2)
                        
                        if aggr >= threshold:
                            doc = nlp_en(en_answers[i])
                            doc1 = nlp_hr(hr_answers[k])
    
                            ner1 = [ent.text for ent in doc.ents if ent.label_ in ['GPE', 'LOC', 'FAC']]
                            ner2 = [ent.text for ent in doc1.ents if ent.label_ in ['GPE', 'LOC', 'FAC']]
                            #org1 = [ent.text for ent in doc.ents if ent.label_ in ['ORG']]
                            #org2 = [ent.text for ent in doc1.ents if ent.label_ in ['ORG']]
                            numbers = [token.text for token in doc if token.like_num] # Extract specific numbers
    
                            if check_items_in_sentence(hr_answers[k], numbers):
                                if check_ner(en_answers[i], hr_answers[k], ner1, ner2):
                                    aligned_sentences.append((hr_ids[k], hr_doms[k], en_sentences[i], hr_answers[k]))
                                    maligned_sentences.append((en_ids[i], en_doms[i], hr_sentences[k], en_answers[i]))
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
    
    #other XL pair
    df1 = pd.DataFrame(maligned_sentences, columns=['id','domain', 'question','answer'])
    df1['qa_pairs'] = df1.apply(lambda row: [{'question': row['question'], 'answer': row['answer']}], axis=1)
    # group by A and B, and combine the dictionaries in CD
    df_grouped1 = df1.groupby(['id', 'domain'])['qa_pairs'].apply(sum).reset_index()
    df_grouped1.head()
    para_data1 = Dataset.from_pandas(df_grouped1)
    para_data1 = para_data1.train_test_split(test_size=0.1)
    para_data1.save_to_disk(name_data)
    