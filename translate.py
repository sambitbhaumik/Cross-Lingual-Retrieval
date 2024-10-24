from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
from datasets import load_dataset
import sys

# Get the command-line arguments
args = sys.argv
lang = args[1]
filename = lang + "_hr_trans"
print(filename)

hr_dataset = load_dataset("clips/mfaq", "hr")
dataset = hr_dataset['validation']


checkpoint = 'facebook/nllb-200-distilled-600M'
# checkpoint = 'facebook/nllb-200-1.3B'
# checkpoint = 'facebook/nllb-200-3.3B'
# checkpoint = 'facebook/nllb-200-distilled-1.3B'

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    print("CUDA is not available. Using CPU instead.")
    device = torch.device("cpu")

model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

target_lang = args[2] #'eng_Latn'
input_lang = 'hrv_Latn'
print(target_lang)

translation_pipeline = pipeline('translation',
                                  model=model,
                                  tokenizer=tokenizer,
                                  src_lang=input_lang,
                                  tgt_lang=target_lang,
                                  device=device,
                                  max_length = 400)
def translate_batch(pair):
  text = pair['question']
  output = translation_pipeline(text)
  trans = output[0]['translation_text']
  pair['question'] = trans
  return pair

# GPT-3.5
def translate_example(example):
    new_example = example.copy()
    for domain in new_example['qa_pairs']:
        for pair in domain:
            pair = translate_batch(pair)
    return new_example

dataset = dataset.map(translate_example, batched=True, batch_size=16)

dataset.save_to_disk(filename)
dataset.cache_files