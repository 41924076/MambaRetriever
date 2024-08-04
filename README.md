# MambaRetriever
We implement the Mamba Retriever, a bi-encoder retrieval model based on the Mamba model. 
We fine-tune our model on the MS MARCO passage ranking dataset for classic short-text retrieval and on the LoCoV0 dataset for long-text retrieval.

# Environment
```bash
pip install torch==2.0 accelerate transformers==4.30.0 datasets deepspeed faiss-cpu
pip install causal_conv1d-1.1.0+cu118torch2.0cxx11abiFALSE-cp38-cp38-linux_x86_64.whl #download from https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.1.0/causal_conv1d-1.1.0+cu118torch2.0cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install mamba_ssm-1.1.1+cu118torch2.0cxx11abiFALSE-cp38-cp38-linux_x86_64.whl #download from https://github.com/state-spaces/mamba/releases/download/v1.1.1/mamba_ssm-1.1.1+cu118torch2.0cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
```
Use pip to install tevatron v1: https://github.com/texttron/tevatron/tree/tevatron-v1

If GPU is not A100 and find that the backpropagation gradient of Mamba is 0 during finetuning, you can use our patch.

# Training details
## MS MARCO
Train Precision is fp32, Dev Precision is fp16.
| Model                | Epochs | Learning Rate | MRR@10 | Recall@1000 |
|---------------------|--------|----------------|---------|--------------|
| bert-base-100m       | 3      | 5e-6            | 32.85   | 95.15        |
| roberta-base-100m    | 3      | 5e-6            | 31.33   | 95.01        |
| opt-100m             | 3      | 5e-6            | 31.14   | 94.9         |
| pythia-100m          | 5      | 5e-6            | 25.36   | 90.95        |
| mamba-100m           | 2      | 1e-5            | 32.29   | 96.65        |
|                      |           |      |        |                |         |              |
| bert-large-300m      | 4      | 5e-6            | 33.87   | 95.72        |
| roberta-large-300m   | 3      | 5e-6            | 33.91   | 96.14        |
| opt-300m             | 3      | 5e-6            | 31.05   | 94.43        |
| pythia-400m          | 5      | 1e-6            | 31.92   | 96.56        |
| mamba-370m           | 2      | 1e-5            | 35.15   | 97.73        |
|                      |           |      |        |                |         |              |
| pythia-1b            | 3      | 5e-7            | 33.77   | 97.44        |
| mamba-790m           | 2      | 1e-5            | 36.28   | 98.34        |
| opt-1.3b             | 3      | 1e-6            | 35.78   | 98.14        |

## LoCo V0
Train and dev Precision are both fp32.
| Model          | Max Length | Epochs | Learning Rate | Avg. nDCG@10 |
|----------------|------------|--------|---------------|--------------|
| opt-100m       | 2k         | 4      | 1e5           | 88.9         |
| pythia-100m    | 2k         | 3      | 5e6           | 79.2         |
| mamba-100m     | 2k         | 2      | 5e5           | 89.1         |
| mamba-100m     | 8k         | 4      | 1e5           | 90.7         |

# BEIR result
## Base size models

| Dataset        | Mamba | BERT-base | RoBERTa-base | OPT | Pythia|
|----------------|-----------|-----------|-----------|-----------|-----------|
| **Size**        | 130M | 110M | 125M | 125M | 160M|
| **Average**        |40.54|36.45|37.02|36.83|31.47|
|||||||
| ArguAna        |41.93|35.35|36.31|36.91|26.11|
| Climate-FEVER  |21.13|15.48|17.58|18.94|15.43|
| DBPedia        |28.14|28.02|24.91|24.45|18.40|
| FEVER          |62.75|61.73|64.99|58.64|50.24|
| FiQA           |22.62|20.53|22.93|21.26|13.68|
| HotpotQA       |50.51|45.23|45.33|46.32|36.59|
| NFCorpus       |28.32|23.56|21.88|21.71|19.19|
| NQ             |38.64|40.04|37.75|36.65|26.46|
| Quora          |86.05|84.35|84.87|84.60|79.59|
| SCIDOCS        |13.50|11.51|10.63|11.38|9.21|
| SciFact        |55.09|42.25|39.14|39.38|42.83|
| TREC-COVID     |59.19|47.18|53.60|59.47|51.82|
| Tóuche-2020    |19.18|18.56|21.28|19.11|19.49|


## Large size models

| Dataset        | Mamba | BERT-large | RoBERTa-large | OPT | Pythia|
|----------------|-----------|-----------|-----------|-----------|-----------|
| **Size**        | 370M | 330M | 355M | 350M | 410M|
| **Average**        |43.52|37.31|38.37|35.89|38.62|
|||||||
| ArguAna        | 41.09 | 36.74 | 39.53 | 36.53 | 35.60 |
| Climate-FEVER  | 22.49 | 17.70  | 19.78 | 18.66 | 19.07 |
| DBPedia        | 33.30  | 29.54 | 26.95 | 23.07 | 26.15 |
| FEVER          | 67.12 | 62.97 | 63.21 | 59.23 | 61.10  |
| FiQA           | 25.95 | 20.83 | 25.80  | 21.20  | 23.03 |
| HotpotQA       | 55.03 | 46.75 | 46.11 | 41.90  | 50.39 |
| NFCorpus       | 30.56 | 23.19 | 24.82 | 18.46 | 26.95 |
| NQ             | 43.18 | 42.75 | 42.23 | 36.38 | 38.05 |
| Quora          | 86.71 | 84.70  | 85.72 | 84.63 | 83.59 |
| SCIDOCS        | 14.81 | 11.42 | 11.95 | 9.96  | 13.92 |
| SciFact        | 59.92 | 44.32 | 42.18 | 38.84 | 54.54 |
| TREC-COVID     | 63.94 | 44.90  | 51.17 | 56.44 | 50.65 |
| Tóuche-2020    | 21.59 | 19.27 | 19.37 | 21.20  | 18.96 |

## >700M size models

| Dataset        | Mamba | Pythia| OPT |
|----------------|-----------|-----------|-----------|
| **Size**        | 790M | 1B | 1.3B |
| **Average**        |44.72|43.11|42.87|
|||||
| ArguAna        |44.69|41.52|45.20|
| Climate-FEVER  |23.67|23.36|23.53|
| DBPedia        |35.26|29.79|33.21|
| FEVER          |67.77|67.02|68.31|
| FiQA           |27.25|26.15|28.51|
| HotpotQA       |56.68|54.97|55.33|
| NFCorpus       |32.73|29.73|29.19|
| NQ             |44.69|40.94|43.93|
| Quora          |86.05|84.40|86.23|
| SCIDOCS        |16.04|15.63|14.81|
| SciFact        |61.01|61.32|51.77|
| TREC-COVID     |64.73|65.85|58.13|
| Tóuche-2020    |20.84|19.68|19.11|
