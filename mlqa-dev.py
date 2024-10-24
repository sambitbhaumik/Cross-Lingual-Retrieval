import os
import sys
import torch
import logging
import pandas as pd
import torch.distributed as dist
from typing import Optional, List
from dataclasses import dataclass, field
from datasets import load_dataset, interleave_datasets
from transformers import set_seed, EarlyStoppingCallback
from transformers import Trainer, TrainingArguments, HfArgumentParser
from transformers import AutoModel, AutoTokenizer
from collections import OrderedDict
from transformers.trainer_utils import get_last_checkpoint
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
from accelerate.utils import DistributedType
import json

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    #gradient_checkpointing: bool = field(default=False)
    hidden_dropout_prob: float = field(default=0.1)
    attention_probs_dropout_prob: float = field(default=0.1)
    model_name_or_path: str = field(default="clips/mfaq")
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
    finetune_data_path: str = field(default="dev-context-en-question-de.json")
    validation_data_path: str = field(default="test-context-en-question-de.json")

@dataclass
class CustomTrainingArgument(TrainingArguments):
    distributed_softmax: bool = field(default=False)
    local_rank: int = field(default=-1, metadata={"help": "Local rank for distributed training on GPUs"})

def load_finetune_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    qa_pairs = []
    for item in data['data']:
        for paragraph in item['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                question = qa['question']
                qa_pairs.append((question, context))
    
    return qa_pairs

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output["last_hidden_state"]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

class FineTuneDataset(torch.utils.data.Dataset):
    def __init__(self, qa_pairs, tokenizer, max_length):
        self.qa_pairs = qa_pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.qa_pairs)

    def __getitem__(self, idx):
        question, answer = self.qa_pairs[idx]
        question = f"<Q>{question}" 
        answer = f"<A>{answer}" 
        return {"question": question, "answer": answer, "page_id": idx}


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        page_id = inputs.pop("page_id", None)
        outputs = model(**inputs)
        sentence_embeddings = mean_pooling(outputs, inputs['attention_mask'])
        q_logits, a_logits = torch.chunk(sentence_embeddings, 2)
        cross_entropy = torch.nn.CrossEntropyLoss()
        dp = q_logits.mm(a_logits.transpose(0, 1))
        labels = torch.arange(dp.size(0), device=dp.device)
        loss = cross_entropy(dp, labels)
        if return_outputs:
            outputs = OrderedDict({"q_logits": q_logits, "a_logits": a_logits})
        return (loss, outputs) if return_outputs else loss


def get_acc_rr(q_logits, a_logits):
    dp = q_logits.mm(a_logits.transpose(0, 1))
    indices = torch.argsort(dp, dim=-1, descending=True)
    targets = torch.arange(indices.size(0), device=indices.device).view(-1, 1)
    targets = targets.expand_as(indices)
    hits = (targets == indices).nonzero()
    ranks = hits[:, -1] + 1
    ranks = ranks.float()
    acc = ranks.eq(1).float().mean()
    rr = torch.reciprocal(ranks).mean()
    return rr.item(), acc.item()


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CustomTrainingArgument))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.remove_unused_columns = False

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    set_seed(training_args.seed)
    padding = "max_length" if data_args.pad_to_max_length else True

    model_kwargs = dict(
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        hidden_dropout_prob=model_args.hidden_dropout_prob,
        attention_probs_dropout_prob=model_args.attention_probs_dropout_prob,
        add_pooling_layer=False
    )

    model = AutoModel.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs
    )

    tokenizer_name_or_path = model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        revision=model_args.model_revision,
        additional_special_tokens=None if data_args.no_special_token else ["<Q>", "<A>", "<link>"]
    )
    if not data_args.no_special_token:
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)


    qa_pairs = load_finetune_data(data_args.finetune_data_path)
    validation_qa_pairs = load_finetune_data(data_args.validation_data_path)
    finetune_dataset = FineTuneDataset(qa_pairs, tokenizer, data_args.max_seq_length)
    eval_dataset = FineTuneDataset(validation_qa_pairs, tokenizer, data_args.max_seq_length)

    
    def finetune_collate_fn(batch):
        questions = [item['question'] for item in batch]
        answers = [item['answer'] for item in batch]
        page_ids = [item['page_id'] for item in batch]
        
        output = tokenizer(
            questions + answers,
            padding=padding,
            truncation=True,
            max_length=data_args.max_seq_length,
            return_tensors="pt",
            pad_to_multiple_of=8
        )
        output["page_id"] = torch.tensor(page_ids)
        return output
    
    def compute_metrics(predictions):
        q_logits, a_logits = predictions.predictions
        q_logits = torch.from_numpy(q_logits)
        a_logits = torch.from_numpy(a_logits)
        mrr, accuracy = get_acc_rr(q_logits, a_logits)
        return {"mrr": mrr, "accuracy": accuracy}

    
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=finetune_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=finetune_collate_fn,
        compute_metrics=compute_metrics
        )

    
    training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED
    if training_args.do_train:
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()


    if training_args.do_predict:
        logger.info("*** Predict ***")
        _, _, metrics = trainer.predict(eval_dataset, metric_key_prefix="predict")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

if __name__ == "__main__":
    main()
