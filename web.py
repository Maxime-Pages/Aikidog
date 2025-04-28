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

@app.route('/')
def index():
    return ""

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080,debug=True)