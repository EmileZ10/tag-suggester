Suggestion de tags Stack Overflow à partir du titre d'une question (TP Poc to Prod, Emile Zanna).

On garde les 40 tags les plus fréquents du dataset (les tags rares n'ont pas assez d'exemples pour être appris), avec un split 80/20. Deux modèles sont entraînés : un TF-IDF + régression logistique (ML) et un DistilBERT fine-tuné (DL). Un tag est prédit si sa probabilité dépasse 0.3.

Résultats sur le jeu de test : ML F1 micro 0.52 et precision@1 0.57 ; BERT F1 micro 0.59 et precision@1 0.65.

Utilisation : `python train_ml_Emile_Zanna.py`, `python train_bert_Emile_Zanna.py`, puis `python predict.py "votre titre de question"`. Les poids de BERT (model_bert_Emile_Zanna.pt, environ 260 Mo) ne sont pas sur GitHub, ils sont recréés par le script d'entraînement (quelques minutes sur GPU).
