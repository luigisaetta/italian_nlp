# Sentiment Analysis di testi in lingua Italiana
This repository contains all the work done in preparation for the project to build datasets and tools to help build better sentiment analysis and text classification models, based on Open Source, on the Italian language

## Conda environment
```
conda create -n ita_sentiment python=3.9
```
```
conda activate ita_sentiment
```
```
git clone https://github.com/luigisaetta/italian_sentiment_analysis.git
```
```
pip install -f requirements.txt
```
## Labels
Le label sono definite in base alla seguente convenzione
```
{"negativo": 0, "neutro": 1, "positivo": 2}
```
## Model training
Nel Notebook trovi codice che può essere utilizzato per classificare testo in (negativo, neutro, positivo).

Il codice è basato sulla libreria **transformers** di HuggingFace.

