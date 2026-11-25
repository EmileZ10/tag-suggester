import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data import load

NAME = "distilbert-base-uncased"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

X_train, X_test, Y_train, Y_test, tags = load()

tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    NAME, num_labels=len(tags), problem_type="multi_label_classification"
).to(DEVICE)


def encode(texts):
    return tok(texts, padding=True, truncation=True, max_length=48, return_tensors="pt").to(DEVICE)


opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
Yt = torch.tensor(Y_train, dtype=torch.float)

for epoch in range(15):
    model.train()
    perm = np.random.permutation(len(X_train))
    for i in range(0, len(perm), 32):
        idx = perm[i : i + 32]
        out = model(**encode([X_train[j] for j in idx]), labels=Yt[idx].to(DEVICE))
        out.loss.backward()
        opt.step()
        opt.zero_grad()
    print("epoch", epoch + 1, "loss", out.loss.item())

model.eval()
probs = []
with torch.no_grad():
    for i in range(0, len(X_test), 64):
        probs.append(torch.sigmoid(model(**encode(X_test[i : i + 64])).logits).cpu().numpy())
probs = np.vstack(probs)

print("F1 micro:", f1_score(Y_test, probs > 0.3, average="micro"))
print("Precision@1:", Y_test[range(len(Y_test)), probs.argmax(1)].mean())

torch.save({"state_dict": model.state_dict(), "tags": list(tags)}, "model_bert_Emile_Zanna.pt")
