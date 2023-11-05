# Practical Machine Learning and Deep Learning - Assignment 1 - Text De-toxification

### Venediktov Rafail BS20-AI
r.venediktov@innopolis.university

## Overview

This project addresses the Text Detoxification Task - the process of transforming toxic text into neutral text without changing the underlying meaning. A formal definition of the task is available in [Text Detoxification using Large Pre-trained Neural Models by Dale et al., page 14](https://arxiv.org/abs/2109.08914). The goal of this assignment is to develop an algorithm or model capable of reducing text toxicity effectively.

## Data Labeling

Text toxicity is determined through binary classification by human annotators. Each piece of text is labeled as toxic or non-toxic, with the ratio of toxic assessments to total assessments indicating the level of toxicity.
## Data Description

The dataset is a subset of the ParaNMT corpus (50M sentence pairs). The filtered ParaNMT-detox corpus (500K sentence pairs) can be downloaded from [here](https://github.com/skoltech-nlp/detox/releases/download/emnlp2021/filtered_paranmt.zip).

The data is given in the `.tsv` format, means columns are separated by `\t` symbol.

| Column | Type | Discription | 
| ----- | ------- | ---------- |
| reference | str | First item from the pair | 
| ref_tox | float | toxicity level of reference text | 
| translation | str | Second item from the pair - paraphrazed version of the reference|
| trn_tox | float | toxicity level of translation text |
| similarity | float | cosine similarity of the texts |
| lenght_diff | float | relative length difference between texts |


## Repository Structure

Your repository should adhere to the following structure:

- `README.md` - Project description and instructions.
- `data/` - Contains raw, intermediate, and external data.
- `models/` - Where trained models and checkpoints are stored.
- `notebooks/` - Jupyter notebooks for data exploration and model visualization.
- `reports/` - Generated analysis and figures interim and final report .
- `requirements.txt` - List of dependencies to replicate the development environment.
- `src/` - Source code used in this project.

## Basic usage

### Setup

```bash
git clone https://github.com/rafailvv/text-de-toxification.git
```
### Install requirements
```
pip install -r requirements.txt
```
### Transform data
```bash
python src/data/make_dataset.py
```
### Train model

```bash
python src/models/train_model.py
```

### Make predictions
```bash
python src/models/predict_model.py
```

### Visualization
```bash
python src/visualization/visualize.py
``````
<p float="left">
  <img src="https://github.com/rafailvv/text-de-toxification/blob/master/reports/figures/application_exploration_tab.PNG" width="48%" />
  <img src="https://github.com/rafailvv/text-de-toxification/blob/master/reports/figures/application_results_tab.PNG" width="48%" />
</p>
