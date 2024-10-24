import os
import gc
import re
import sys
import torch
import logging
import pandas as pd
import torch.distributed as dist
from typing import Optional, List
from dataclasses import dataclass, field
from datasets import load_dataset, interleave_datasets, Dataset, load_from_disk
from transformers import set_seed, EarlyStoppingCallback
from transformers import Trainer, TrainingArguments, HfArgumentParser
from transformers import AutoModel, AutoTokenizer, AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from collections import OrderedDict, defaultdict
from transformers.trainer_utils import get_last_checkpoint
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
from accelerate.utils import DistributedType
import numpy as np
import polars as pl
from cleantext import clean
from deep_translator import GoogleTranslator
import tldextract
from rapidfuzz import fuzz
import spacy
from sentence_transformers import SentenceTransformer
import wandb
from flair.data import Sentence
from flair.models import SequenceTagger
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import numpy as np
from collections import defaultdict
import faiss


from dataloader import IterableDataset, ValidationDataset

logger = logging.getLogger(__name__)
# load tagger
tagger = SequenceTagger.load("ner-large")

@dataclass
class ModelArguments:
    #gradient_checkpointing: bool = field(default=False)
    hidden_dropout_prob: float = field(default=0.1)
    attention_probs_dropout_prob: float = field(default=0.1)
    model_name_or_path: str = field(default="xlm-roberta-base")
    config_name: Optional[str] = field(default=None)
    tokenizer_name: Optional[str] = field(default=None)
    cache_dir: Optional[str] = field(default=None)
    model_revision: str = field(default="main")


@dataclass
class DataTrainingArguments:
    preprocessing_num_workers: Optional[int] = field(default=None)
    max_seq_length: int = field(default=None)
    languages: Optional[List[str]] = field(default=None)
    probabilities: Optional[List[float]] = field(default=None)
    overwrite_cache: bool = field(default=False)
    pad_to_max_length: bool = field(default=False)
    single_domain: bool = field(default=False)
    alpha: float = field(default=0.3)
    no_special_token: bool = field(default=False)
    limit_valid_size: Optional[int] = field(default=False)
    limit_train_size: Optional[int] = field(default=False)
    lang1: Optional[str] = field(default=None)
    lang2: Optional[str] = field(default=None)
    project_name: Optional[str] = field(default=None)
    translated_set: Optional[str] = field(default=None)

@dataclass
class CustomTrainingArgument(TrainingArguments):
    distributed_softmax: bool = field(default=False)
    local_rank: int = field(default=-1, metadata={"help": "Local rank for distributed training on GPUs"})


def distributed_softmax(q_output, a_output, rank, world_size):
    q_list = [torch.zeros_like(q_output) for _ in range(world_size)]
    a_list = [torch.zeros_like(a_output) for _ in range(world_size)]
    dist.all_gather(tensor_list=q_list, tensor=q_output.contiguous())
    dist.all_gather(tensor_list=a_list, tensor=a_output.contiguous())
    q_list[rank] = q_output
    a_list[rank] = a_output
    q_output = torch.cat(q_list, 0)
    a_output = torch.cat(a_list, 0)
    return q_output, a_output


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output["last_hidden_state"] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        page_id = inputs.pop("page_id", None)
        outputs = model(**inputs)
        sentence_embeddings = mean_pooling(outputs, inputs['attention_mask'])
        q_logits, a_logits = torch.chunk(sentence_embeddings, 2)
        if self.args.distributed_softmax and self.args.local_rank != -1 and return_outputs is False:
            q_logits, a_logits = distributed_softmax(
                q_logits, a_logits, self.args.local_rank, self.args.world_size
            )
            labels = torch.arange(q_logits.size(0), device=a_logits.device)
        cross_entropy = torch.nn.CrossEntropyLoss()
        dp = q_logits.mm(a_logits.transpose(0, 1))
        labels = torch.arange(dp.size(0), device=dp.device)
        loss = cross_entropy(dp, labels)
        if return_outputs:
            outputs = OrderedDict({"q_logits": q_logits, "a_logits": a_logits, "page_id": page_id})
        return (loss, outputs) if return_outputs else loss


def get_acc_rr(q_logits, a_logits):
    q_logits = torch.from_numpy(q_logits)
    a_logits = torch.from_numpy(a_logits)
    dp = q_logits.mm(a_logits.transpose(0, 1))
    indices = torch.argsort(dp, dim=-1, descending=True)
    targets = torch.arange(indices.size(0), device=indices.device).view(-1, 1)
    targets = targets.expand_as(indices)
    hits = (targets == indices).nonzero()
    ranks = hits[:, -1] + 1
    ranks = ranks.float()
    acc = ranks.eq(1).float().squeeze()
    rr = torch.reciprocal(ranks).squeeze()
    return rr, acc


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CustomTrainingArgument))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.remove_unused_columns = False
    
    lang1 = data_args.lang1
    lang2 = data_args.lang2
    
    #laser = Laser()
    labse = SentenceTransformer("sentence-transformers/LaBSE")
    
    

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    set_seed(training_args.seed)

    model_kwargs = dict(
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        hidden_dropout_prob=model_args.hidden_dropout_prob,
        attention_probs_dropout_prob=model_args.attention_probs_dropout_prob,
        add_pooling_layer=False
    )

    tokenizer_name_or_path = model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        revision=model_args.model_revision,
        additional_special_tokens=None if data_args.no_special_token else ["<Q>", "<A>", "<link>"]
    )
    
    padding = "max_length" if data_args.pad_to_max_length else True

    en_dataset = load_dataset("clips/mfaq", lang1)
    hr_dataset = load_dataset("clips/mfaq", lang2)

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
                ner_trans = GoogleTranslator(source=lang1, target=lang2).translate(text=ner)
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
                if check_org_fuzz(sentence2.lower(), org.lower()): #or check_org_fuzz(sentence2.lower(), org_trans.lower()):
                    continue
                else:
                    return False
            return True
        
    def collate_fn(batch):
        questions, answers, page_ids = [], [], []
        for item in batch:
            questions.append(item['question'] if data_args.no_special_token else f"<Q>{item['question']}")
            answers.append(item['answer'] if data_args.no_special_token else f"<A>{item['answer']}")
            page_ids.append(item["page_id"])
        output = tokenizer(
            questions + answers,
            padding=padding,
            truncation=True,
            max_length=data_args.max_seq_length,
            return_tensors="pt",
            pad_to_multiple_of=8
        )
        output["page_id"] = torch.Tensor(page_ids)
        return output

    def compute_metrics(predictions):
        q_output, a_output, page_id = predictions.predictions
        unique_page_ids = set(page_id.tolist())
        global_rr, global_acc, pp_mrr, pp_acc = [], [], [], []
        for unique_page_id in unique_page_ids:
            selector = page_id == unique_page_id
            s_q_output = q_output[selector, :]
            s_a_output = a_output[selector, :]
            rr, acc = get_acc_rr(s_q_output, s_a_output)
            global_rr.append(rr)
            global_acc.append(acc)
            pp_mrr.append(rr.mean())
            pp_acc.append(acc.mean())
        global_mrr = torch.cat(global_rr).mean()
        global_acc = torch.cat(global_acc).mean()
        per_page_mrr = torch.stack(pp_mrr).mean()
        per_page_acc = torch.stack(pp_acc).mean()
        return {"global_mrr": global_mrr, "global_acc": global_acc, "per_page_mrr": per_page_mrr, "per_page_acc": per_page_acc}
    
    def train_and_evaluate(train_dataset, eval_dataset):
        model = AutoModel.from_pretrained(
            model_args.model_name_or_path,
            **model_kwargs
        )

        if not data_args.no_special_token:
            model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
        
        model.gradient_checkpointing_enable()
        
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=collate_fn,
            compute_metrics=compute_metrics
        )

        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED
        trainer.train()
        metrics = trainer.evaluate()

        del trainer
        del model
        return metrics

    en_domains = list(map(extract_root_domain, en_dataset['train']['domain']))
    en_domains = list(set(en_domains))  # to get unique domains
    domain_datasets = {}
    hr_domains = list(map(extract_root_domain, hr_dataset['train']['domain']))
    hr_domains = list(set(hr_domains))  # to get unique domains

    en_hr = [x for x in en_domains if x in hr_domains]
    

    hr_filtered = hr_dataset['train'].filter(lambda example: extract_root_domain(example['domain']) in en_hr)
    en_filtered = en_dataset['train'].filter(lambda example: extract_root_domain(example['domain']) in en_hr)    

    arr_en = pl.from_arrow(en_filtered.data.table)
    arr_hr = pl.from_arrow(hr_filtered.data.table)

    eval_data = load_from_disk(data_args.translated_set)
    scores = defaultdict(dict_of_lists_of_lists)
    k = 5

    sweep_config = {
        "method": "bayes",
        "metric": {"name": "mrr", "goal": "maximize"},
        "parameters": {
            "s1": {"min": 0.4, "max": 1.0},
            "threshold": {"min": 0.5, "max": 1.1}
        },
        "project": data_args.project_name
    }

    # Initialize the sweep
    sweep_id = wandb.sweep(sweep_config)

    en_sent = {}
    en_ans = {}
    hr_sent = {}
    hr_ans = {}
    hr_dom = {}
    hr_id = {}

    n = min(len(en_hr), 20)
    fil_doms = en_hr[:n]

    print("Creating data dicts ...")
    for d in fil_doms:
        en_dataset_filtered = arr_en.filter(pl.col('domain').apply(extract_root_domain) == d)
        hr_dataset_filtered = arr_hr.filter(pl.col('domain').apply(extract_root_domain) == d)
        
        en_sent[d] = [cleaner(pair["question"]) for page in en_dataset_filtered.rows(named=True) for pair in page["qa_pairs"]]
        en_ans[d] = [cleaner(pair["answer"]) for page in en_dataset_filtered.rows(named=True) for pair in page["qa_pairs"]]
        hr_sent[d] = [cleaner(pair["question"]) for page in hr_dataset_filtered.rows(named=True) for pair in page["qa_pairs"]]
        hr_ans[d] = [cleaner(pair["answer"]) for page in hr_dataset_filtered.rows(named=True) for pair in page["qa_pairs"]]
        hr_dom[d] = [page["domain"] for page in hr_dataset_filtered.rows(named=True) for pair in page["qa_pairs"]]
        hr_id[d] = [page["id"] for page in hr_dataset_filtered.rows(named=True) for pair in page["qa_pairs"]]

    print("Calculating q_sims ...")
    for d in fil_doms:
        # Part 1 (unchanged)
        domain_name = d
        en_sentences = []
        hr_sentences = []
        
        en_sentences = en_sent[domain_name]
        hr_sentences = hr_sent[domain_name]

        if en_sentences and hr_sentences:
            en_embeddings = labse.encode(en_sentences)
            hr_embeddings = labse.encode(hr_sentences)

            # Tuning optimized by Claude-3.5 Sonnet
            # FAISS for faster nearest neighbor search
            
            d = en_embeddings.shape[1]  # dimension of embeddings
            index_en = faiss.IndexFlatIP(d)
            index_hr = faiss.IndexFlatIP(d)

            index_en.add(en_embeddings.astype('float32'))
            index_hr.add(hr_embeddings.astype('float32'))

            # Find k-nearest neighbors for all embeddings at once
            _, en_nn_indices = index_hr.search(en_embeddings.astype('float32'), k)
            _, hr_nn_indices = index_en.search(hr_embeddings.astype('float32'), k)

            for i in range(len(en_embeddings)):
                x = en_embeddings[i]
                x_neighbors = hr_embeddings[en_nn_indices[i]]

                for j in range(len(hr_embeddings)):
                    y = hr_embeddings[j]
                    y_neighbors = en_embeddings[hr_nn_indices[j]]
                    q_sim = margin_score_vectorized(x, y, x_neighbors, y_neighbors, k)
                    scores[domain_name][i][j] = q_sim

    def train(config=None):
        wandb.init()

        aligned_sentences = []
        print("Creating dataset ...")
        for d in fil_doms:
            s1 = wandb.config.s1
            s2 = 1.0 - s1
            threshold = wandb.config.threshold
            #val_size = wandb.config.val_size

             # Part 1 (unchanged)
            domain_name = d
            en_sentences = []
            en_answers = []
            
            hr_sentences = []
            hr_answers = []
            hr_ids = []
            hr_doms = []

            en_sentences = en_sent[domain_name]
            en_answers = en_ans[domain_name]
            hr_sentences = hr_sent[domain_name]
            hr_answers = hr_ans[domain_name]
            hr_doms = hr_dom[domain_name]
            hr_ids = hr_id[domain_name]
            for i in range(len(en_embeddings)):
                if(i >= len(en_sentences)):
                    break
                for j in range(len(hr_embeddings)):
                    q_score = scores[d][i][j]
                    if(j >= len(hr_sentences)):
                        break
                    if(q_score >= 0.9):
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


        df = pd.DataFrame(aligned_sentences, columns=['id','domain', 'question','answer'])
        df['qa_pairs'] = df.apply(lambda row: [{'question': row['question'], 'answer': row['answer']}], axis=1)
        # group by A and B, and combine the dictionaries in CD
        df_grouped = df.groupby(['id', 'domain'])['qa_pairs'].apply(sum).reset_index()
        df_grouped.head()
        para_data = Dataset.from_pandas(df_grouped)
        para_data = para_data.train_test_split(test_size=0.01)

        ## training
        datasets = [para_data]        

        train_datasets = [e["train"] for e in datasets]
        eval_datasets = [eval_data]

        if data_args.limit_train_size:
            train_datasets = [e.select(range(data_args.limit_train_size)) for e in train_datasets]  # Limit the training data to 1000 rows
        if data_args.limit_valid_size:
            eval_datasets = [e.select(range(data_args.limit_valid_size)) for e in eval_datasets]

        # Ensure datasets are not empty
        if not train_datasets or not any(len(ds) > 0 for ds in train_datasets):
            raise ValueError("Training datasets are empty. Ensure that the datasets are properly loaded and not empty.")

        if not eval_datasets or not any(len(ds) > 0 for ds in eval_datasets):
            raise ValueError("Evaluation datasets are empty. Ensure that the datasets are properly loaded and not empty.")
        
        eval_dataset = ValidationDataset(interleave_datasets(eval_datasets))

        
        world_size = 1 if training_args.world_size is None else training_args.world_size
        train_dataset = IterableDataset(
            train_datasets,
            data_args.languages, 
            probabilities=data_args.probabilities, 
            batch_size=training_args.per_device_train_batch_size*world_size,
            seed=training_args.seed,
            single_domain=data_args.single_domain,
            alpha=data_args.alpha
        )

        metrics = train_and_evaluate(train_dataset, eval_dataset)
        wandb.log({"mrr": metrics['eval_global_mrr']})
        gc.collect()
        torch.cuda.empty_cache()

    wandb.agent(sweep_id, function=train, count=8)


if __name__ == "__main__":
    main()
