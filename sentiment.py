import torch
from torch import nn  
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

from config import HF_SENT_MODEL_NAME

class SentimentAnalyzer:
    def __init__(self):
        # Load the tokenizer and model from HF hub
        self.MODEL_NAME = HF_SENT_MODEL_NAME

        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)

        # Load the model, use .cuda() to load it on the GPU
        self.model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME)

        self.label_list = ["negativo","neutro","positivo"]

    def analyze(self, text):
        # tokenize
        input_ids = self.tokenizer.encode(text, add_special_tokens=True)

        # Create tensor, use .cuda() to transfer the tensor to GPU if needed
        # add fake batch dimension
        tensor = torch.tensor(input_ids).long().unsqueeze(0)

        # Call the model and get the logits
        logits = self.model(tensor)

        # Remove the fake batch dimension
        # corrected by LS
        logits = logits.logits.squeeze(0)

        # The model was trained with a Log Likelyhood + Softmax combined loss, 
        # hence to extract probabilities we need a softmax on top of the logits tensor
        proba = nn.functional.softmax(logits, dim=0)

        np_proba = proba.detach().numpy()

        # prende la prob più alta, l'indice e trasforma in label
        return self.label_list[np_proba.argmax()]
    
    