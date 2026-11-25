import sys

import joblib
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

title = " ".join(sys.argv[1:]) or "How to read a CSV file in Python?"

# modele ML
model, tags = joblib.load("model_ml_Emile_Zanna.joblib")
probs = model.predict_proba([title])[0]
print("ML  :", [tags[i] for i in probs.argsort()[::-1][:3] if probs[i] > 0.1] or [tags[probs.argmax()]])

# modele BERT
NAME = "distilbert-base-uncased"
saved = torch.load("model_bert_Emile_Zanna.pt", weights_only=True)
tok = AutoTokenizer.from_pretrained(NAME)
bert = AutoModelForSequenceClassification.from_pretrained(NAME, num_labels=len(saved["tags"]))
bert.load_state_dict(saved["state_dict"])
bert.eval()
with torch.no_grad():
    probs = torch.sigmoid(bert(**tok(title, return_tensors="pt")).logits)[0]
tags = saved["tags"]
print("BERT:", [tags[i] for i in probs.argsort(descending=True)[:3] if probs[i] > 0.1] or [tags[probs.argmax()]])
