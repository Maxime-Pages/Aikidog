from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import pearsonr

app = Flask(__name__)

def matchcommand(command, actions, threshold=0.50):
    texts = [command] + actions
    vect = TfidfVectorizer()
    tfidfg_mat = vect.fit_transform(texts).toarray()

    query_tf_idf = tfidfg_mat[0]
    corpus = tfidfg_mat[1:]

    results = []

    for id, document_tf_idf in enumerate(corpus):
        pearson_corr, _ = pearsonr(query_tf_idf, document_tf_idf)
        if pearson_corr > threshold:
            result = {"ID": id, "document": texts[id+1], "similarity": float(pearson_corr)}
            results.append(result)
    
    return results.sort(key=lambda x: x['similarity'], reverse=True)

def search(query):
    texts = [
        query,
        "Le traitement du language naturel est fascinant.",
        "Le traitement des langues est une branche de l'intelligence artificielle.",
        "L'analyse de texte est utilisée pour la traduction automatique.",
        "Les modèles de langage comme GPT sont très puissants.",
        "La vectorisation TF-IDF est une technique courante en NLP.",
        "Les réseaux neuronaux sont utilisés pour diverses tâches d'apprentissage automatique.",
        "La similarité entre documents peut être mesurée de différentes manières.",
        "Les algorithmes de machine learning nécessitent des données pour s'entraîner.",
        "La compréhension du contexte est essentielle pour les modèles de langage.",
        "Les systèmes de recommandation utilisent souvent des techniques de NLP.",
        "L'intelligence artificielle transforme de nombreux secteurs industriels.",
        "Les outils comme Flask facilitent le développement d'applications web.",
        "La recherche d'information est un domaine clé en NLP.",
        "Les modèles pré-entraînés peuvent être adaptés à des tâches spécifiques."
    ]

    vect = TfidfVectorizer()
    tfidfg_mat = vect.fit_transform(texts).toarray()

    query_tf_idf = tfidfg_mat[0]
    corpus = tfidfg_mat[1:]

    #print("Query TF-IDF:", str(query_tf_idf))
    #print("Corpus TF-IDF:", str(corpus))
    found = []

    for id,document_tf_idf in enumerate(corpus):
        pearson_corr, _ = pearsonr(query_tf_idf, document_tf_idf)
        if pearson_corr > 0.20:
                result = {"ID": id, "document": texts[id+1], "similarity": float(pearson_corr)}
                found.append(result)
        
    return found

@app.route('/')
def index():
    return ""

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080,debug=True)