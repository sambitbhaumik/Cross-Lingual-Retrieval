#!/usr/bin/env bash

# run setup
source /data/users/sbhaumik/mfaq/setup.sh
source /data/users/sbhaumik/mfaq/rename_gpus.sh

# run misc. stuff
echo $CUDA_VISIBLE_DEVICES

# For laser embeddings
#python3 -m pip install -q laserembeddings
#python3 -m pip install laserembeddings --no-deps
#python3 -m pip install sacremoses==0.0.35 --no-deps
#python3 -m pip install subword-nmt --no-deps
#python3 -m pip install transliterate --no-deps
#python3 -m laserembeddings download-models

# To log into wandb
#export WANDB_API_KEY=<API KEY>

# download spacy models like this
#pip install -U spacy 
#python3 -m spacy download de_core_news_lg
#python3 -m spacy download sv_core_news_lg

# translating FAQ pairs
python $PROJECT_DIR/translate.py en eng_Latn
python $PROJECT_DIR/translate.py it ita_Latn

# xl retrieval example
python $PROJECT_DIR/extraction_cse_optim.py en it en_core_web_lg it_core_news_lg eng_Latn ita_Latn 0.6 0.84
python $PROJECT_DIR/extraction_msfe_optim.py it hr 0.5 0.87

# creating mMARCO eval sets
python $PROJECT_DIR/load-mmarco.py
# evaluating mMARCO eval sets
python $PROJECT_DIR/mmarco-eval.py output/m3-mfaq2-128-downsampled/checkpoint-1500

# creating and evaluating MLQA sets
python $PROJECT_DIR/mlqa.py output/xlm-r-mfaq-en-de/checkpoint-2500
python $PROJECT_DIR/mlqa-hi.py FacebookAI/xlm-roberta-base
python $PROJECT_DIR/mlqa-es.py output/m3-xlfaq-128/checkpoint-50
python $PROJECT_DIR/mlqa-ar-hi.py google-bert/bert-base-multilingual-cased

# evaluating on MLQA finetuning
bash $PROJECT_DIR/scripts/mlqa-dev.sh

# tuning script
bash $PROJECT_DIR/scripts/tune.sh

# training script for parallel sets (XLFAQ)
bash $PROJECT_DIR/scripts/para-train.sh
# training script for MFAQ datasets
bash $PROJECT_DIR/scripts/train.sh
### for prediction change do_train to do_predict and change "stage" of "zero_optimization" in DeepSpeed config to 3.






