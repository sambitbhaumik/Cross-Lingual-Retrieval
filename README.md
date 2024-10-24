# Master Thesis: Leveraging Multi-lingual FAQ Resources for Enhanced Cross-Lingual Information Retrieval

## Abstract
Existing research on cross-lingual information retrieval suggests reliance on
neural machine translation to translate either one of user queries to the target
language, or target database to source language. Additionally, the lack of
cross-lingual retrieval data for low-resource languages makes training consistent
retrieval models more challenging. This thesis investigates the enhancement of
cross-lingual retrieval question answering (QA) systems through the utilization
of a cross-lingually aligned Frequently Asked Questions (FAQ) dataset, with
the primary objective being the improvement of the passage-based retrieval
performance of a system when presented with a question in one language and
a database of prospective answers in another. Our study involves extraction
and cross lingual alignment of semantically equivalent FAQ pairs from a multilingual
FAQ dataset, as an alternative to machine translation. We introduce
and investigate the effects of cross-lingual passage-level relevance data for
finetuning and query-document style pretraining. The findings of this thesis
underscore the potential of diverse multilingual knowledge sources in enhancing
the performance of cross-lingual retrieval systems. Our trained models demonstrate
marginal improvements over common multilingual reranking baselines
like XLM-R, mBERT. Additionally, the competitive performance of models
trained without explicit cross-lingual supervision highlights the necessity for
further research into sentence mining to maintain the validity of questions and
answers across languages.

## MFAQ Dataset
The MFAQ dataset is hosted on the HuggingFace hub. You can find it [here](https://huggingface.co/datasets/clips/mfaq).

Start by installing the dataset package:
```
pip install datasets
```

Then import the dataset:
```python
from datasets import load_dataset
en_dataset = load_dataset("clips/mfaq", "en")
```

## MLQA dataset
https://huggingface.co/datasets/facebook/mlqa

## mMARCO dataset
https://huggingface.co/datasets/unicamp-dl/mmarco



## Scripts

`run.sh` has examples for running different scripts and files related to the thesis project:

- `para-train.sh` for training on parallel FAQs (XLFAQ) and `train.sh` for training on MFAQ
- `tune.sh` for tuning on different language pairs
- `translate.py` for translating FAQ pairs from MFAQ dataset
- `extraction_*_optim.py` files for different retrieval methods (Cosine-SpaCy Extraction and Margin-Score-Flair Extraction)
- `shuffle.py` to down sample MFAQ dataset

- `mlqa*.py` files for creating and evaluating on MLQA sets
- `load-mmarco.py` for creating mMARCO evaluation sets for the project
- `mmarco-eval.py` for evaluating the mMARCO eval sets

Training architecture adopted from MFAQ paper.

## References:

1. MFAQ

```
@article{DBLP:journals/corr/abs-2109-12870,
  author       = {Maxime De Bruyn and
                  Ehsan Lotfi and
                  Jeska Buhmann and
                  Walter Daelemans},
  title        = {{MFAQ:} a Multilingual {FAQ} Dataset},
  journal      = {CoRR},
  volume       = {abs/2109.12870},
  year         = {2021},
  url          = {https://arxiv.org/abs/2109.12870},
  eprinttype    = {arXiv},
  eprint       = {2109.12870},
  timestamp    = {Mon, 04 Oct 2021 17:22:25 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2109-12870.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

2. mMARCO
```
@article{DBLP:journals/corr/abs-2108-13897,
  author       = {Luiz Bonifacio and
                  Israel Campiotti and
                  Roberto de Alencar Lotufo and
                  Rodrigo Frassetto Nogueira},
  title        = {mMARCO: {A} Multilingual Version of {MS} {MARCO} Passage Ranking Dataset},
  journal      = {CoRR},
  volume       = {abs/2108.13897},
  year         = {2021},
  url          = {https://arxiv.org/abs/2108.13897},
  eprinttype    = {arXiv},
  eprint       = {2108.13897},
  timestamp    = {Mon, 20 Mar 2023 15:35:34 +0100},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2108-13897.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

3. MLQA
```
@article{lewis2019mlqa,
  title={MLQA: Evaluating Cross-lingual Extractive Question Answering},
  author={Lewis, Patrick and O\u{g}uz, Barlas and Rinott, Ruty and Riedel, Sebastian and Schwenk, Holger},
  journal={arXiv preprint arXiv:1910.07475},
  year={2019}
}
```