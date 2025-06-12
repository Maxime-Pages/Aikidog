from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import pearsonr
from pidog import Pidog
from time import sleep
import speechrecognition as sr

app = Flask(__name__)
my_dog = Pidog(head_init_angles=[0, 0, -30])
sleep(1)
amazon_url = "https://www.amazon.com/dp/B0C5Z3K6XG?ref=myi_title_dp"



def matchcommand(command,threshold=0.20):
    texts = [
        command,
        "stand",
        "sit",
        "lie",
        "lie_with_hands_out",
        "forward",
        "backward",
        "turn_left",
        "turn_right",
        "trot",
        "stretch",
        "push_up",
        "doze_off",
        "nod_lethargy",
        "shake_head",
        "tilting_head_left",
        "tilting_head_right",
        "tilting_head",
        "head_bark",
        "wag_tail",
        "head_up_down",
        "half_sit",
    ]
    print(f"Command: {command}")
    print(f"Texts: {texts}")
    vect = TfidfVectorizer()
    tfidfg_mat = vect.fit_transform(texts).toarray()

    query_tf_idf = tfidfg_mat[0]
    corpus = tfidfg_mat[1:]

    results = []

    for id, document_tf_idf in enumerate(corpus):
        pearson_corr, _ = pearsonr(query_tf_idf, document_tf_idf)
        print(f"Pearson correlation for {texts[id+1]}: {pearson_corr}")
        if pearson_corr > threshold:
            result = {"ID": id, "action": texts[id+1], "similarity": float(pearson_corr)}
            results.append(result)
    
    return results.sort(key=lambda x: x['similarity'], reverse=True) if results else None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/command', methods=['POST'])
def command():
    command = request.json.get('command', '')
    print(f"Received command: {command}")
    command = command.replace(" ", "_").lower()
    action = matchcommand(command)
    if action:
        print(f"Executing action: {action[0]['action']}")
        my_dog.do_action(action[0]['action'])
        return action[0]['action']
    return "No matching action found"

@app.route('/listen')
def listen():
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 4000
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 0.5
    recognizer.non_speaking_duration = 0.5
    #Listen to the mic
    with sr.Microphone(chunk_size=8192) as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
        text = recognizer.recognize_google(audio)
        answer = getresponse(text)
        return answer


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080)