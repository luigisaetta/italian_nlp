# Sentiment Analysis di testi in lingua Italiana
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This repository contains all the work done in preparation for the project to build datasets and tools to help build better sentiment 
analysis and text classification models, based on Open Source, on the **Italian** language

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
## Applicazione di Data Labelling
Nel repository si trova una prima versione di un'applicazione che può essere utilizzata per il labelling dei testi.

Il codice è basato su Streamlit.

L'applicazione ha una funzionalità di **automatic labelling**: è possibile predefinire le label utilizzando un 
modello pre-trained, disponibile su HF Hub.

Ovviamente, le label devono essere verificate ed, eventualmente, corrette.

Per eseguire l'applicazione
```
streamlit run label_text.py
```
## Model training
Nel [Notebook](./fine_tune_sentiment.ipynb) trovi codice che può essere utilizzato per addestrare un modello per classificare 
testo in 
(negativo, neutro, positivo).

Il codice è basato sulla libreria **Transformers** di HuggingFace.
## Valutazione di Modelli
Nel [Notebook](./compute_accuracy.ipynb) trovi il codice che può essere utilizzato per valutare l'accuratezza di un modello su un 
dato dataset, fornito in formato csv.
## Dataset utilizzati
https://huggingface.co/datasets/cardiffnlp/tweet_sentiment_multilingual

## Max input size
Il modello pretrained inizialmente utilizzato, di 
[Neuraly](https://huggingface.co/neuraly/bert-base-italian-cased-sentiment) è basato su un modello 
**bert-base-cased**.

La dimensione massima per l'input dovrebbe essere **512** token.
