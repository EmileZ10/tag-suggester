import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline

from data import load

X_train, X_test, Y_train, Y_test, tags = load()

model = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
    OneVsRestClassifier(LogisticRegression(C=10, max_iter=1000)),
)
model.fit(X_train, Y_train)

# on predit les tags dont la proba depasse 0.3
probs = model.predict_proba(X_test)
pred = probs > 0.3
print("F1 micro:", f1_score(Y_test, pred, average="micro"))
print("Precision@1:", Y_test[range(len(Y_test)), probs.argmax(1)].mean())

joblib.dump((model, tags), "model_ml_Emile_Zanna.joblib")
