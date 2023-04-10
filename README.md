# Cross-lingual-Reasoning-Network-for-CLKGQA
A Reasoning Network for multi-relation Question Answering over Cross-lingual Knowledge Graphs

# CLKGQA
This is the repo hosting all the experiment code of the paper *"CLRN: A Reasoning Network for multi-relation Question Answering overCross-lingual Knowledge Graphs"*

## Data
[**MLPQ**](https://github.com/tan92hl/Dataset-for-QA-over-Multilingual-KG)

## Environment
* cudatoolkit=10.2
* pytorch==1.8.1
* transformers==4.5.0
* chinese_converter==1.0.2
* jsonlines==2.0.0
* auto_mix_prep==0.2.0
* protobuf==3.15.8
* python_Levenshtein==0.12.2

> NVIDIA GeForce RTX 2080 GPU is used in our experiments
## CLRN
This module is consisted of multilingual-BERT and CLRN and a training pipeline to fine tune BERT and train the CLRN 
simultaneously. The trained models are saved to the specified directory after the training is finished.
```angular2html
python train.py
```
## question_answering
The question answering module utilizes the trained model from CLRN module and reason continuously across knowledge graphs 
of different languages and save the generated equal paths and aligned entities as the answering result.
```angular2html
python main.py
```
