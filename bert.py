from transformers import pipeline

# Pipeline de question-réponse avec BERT
qa = pipeline("question-answering", model="mrm8488/bert-multi-cased-finetuned-xquadv1")

# Contexte
contexte = """
Le modèle BERT (Bidirectional Encoder Representations from Transformers) a été développé par Google en 2018.
Il permet de mieux comprendre le sens des mots dans un contexte en lisant la phrase à gauche et à droite.
"""

# Question
question = "Qui a développé BERT ?"

# Réponse
reponse = qa(question=question, context=contexte)
print("🧠 Réponse à la question :")
print(reponse)