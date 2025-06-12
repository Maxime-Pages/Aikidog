import requests
import json

def analyse_sentiment(text):
    url = "http://52.47.99.5:5000/predict"
    payload = {"sentence": text}

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Lève une erreur pour les codes de statut HTTP 4xx/5xx
        data = response.json()

        # Vérifie que la réponse contient bien la clé "sentiment"
        if "sentiment" in data:
            if data["sentiment"] == "NEGATIVE":
                return 0
            elif data["sentiment"] == "POSITIVE":
                return 1
        return None  # Si la clé "sentiment" est absente ou a une autre valeur

    except requests.exceptions.RequestException as e:
        print(f"Error during request: {e}")
        return None


def search(text):
    url = "http://52.47.99.5:8080/search"
    payload = {"question": text}

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Lève une erreur pour les codes de statut HTTP 4xx/5xx
        data = response.json()

        # Vérifie que la réponse contient bien la clé "sentiment"
        if "answer" in data:
            return data["answer"]
        return None  # Si la clé "sentiment" est absente ou a une autre valeur

    except requests.exceptions.RequestException as e:
        print(f"Error during request: {e}")
        return None

def get_response():
    text = input(">")
    sentiment = analyse_sentiment(text)
    answer = search(text)
    if sentiment is not None and answer is not None:
        dct = {
            "sentiment": sentiment,
            "answer": answer
        }
    else:
        dct = {"Error": "no response"}
    result = json.dumps(dct)
    return result
