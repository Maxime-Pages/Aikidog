import warnings
warnings.simplefilter('ignore')
import os
import pandas as pd
import numpy as np
import unidecode
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from flask import Flask, request, jsonify
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class ImprovedSentimentClassifier:
    def __init__(self):
        # Initialize components
        self.dataset_path = os.path.join('model', "dataset.txt")
        self.separator = ";"
        self.vectorizer = None
        self.model = None
        self.scaler = None
        self.score = None
        self.stemmer = SnowballStemmer('french')
        
        # Enhanced French stopwords
        try:
            self.french_stopwords = set(stopwords.words('french'))
        except:
            # Comprehensive fallback French stopwords
            self.french_stopwords = set([
                'le', 'de', 'et', 'à', 'il', 'être', 'et', 'en', 'avoir', 'que', 'pour',
                'dans', 'ce', 'son', 'une', 'sur', 'avec', 'ne', 'se', 'pas', 'tout',
                'plus', 'par', 'grand', 'celui', 'me', 'bien', 'autre', 'comme', 'notre',
                'temps', 'très', 'sans', 'deux', 'lui', 'voir', 'nom', 'long', 'je', 'leur',
                'y', 'donner', 'elle', 'faire', 'celui-ci', 'ou', 'là', 'les', 'la', 'du',
                'des', 'un', 'ma', 'mon', 'mes', 'ta', 'ton', 'tes', 'sa', 'ses', 'nos',
                'vos', 'leurs', 'ce', 'cet', 'cette', 'ces', 'qui', 'quoi', 'dont', 'où'
            ])
        
        # Enhanced sentiment lexicons
        self.positive_words = {
            'excellent', 'super', 'génial', 'parfait', 'formidable', 'magnifique', 
            'fantastique', 'merveilleux', 'extraordinaire', 'sublime', 'splendide',
            'remarquable', 'exceptionnel', 'incroyable', 'sensationnel', 'éblouissant',
            'somptueux', 'captivant', 'enchanteur', 'ravissant', 'délicieux', 'charmant',
            'génial', 'classe', 'chouette', 'sympa', 'cool', 'top', 'bon', 'bien',
            'content', 'heureux', 'satisfait', 'ravi', 'enchanté', 'comblé', 'impressionné',
            'adorer', 'aimer', 'apprécier', 'recommander', 'bravo', 'félicitations',
            'merci', 'reussi', 'parfaitement', 'impeccable', 'inoubliable', 'unique'
        }
        
        self.negative_words = {
            'nul', 'horrible', 'mauvais', 'déçu', 'terrible', 'lamentable', 'catastrophe',
            'affreux', 'atroce', 'ignoble', 'répugnant', 'détestable', 'odieux', 
            'abominable', 'exécrable', 'infect', 'déplorable', 'pitoyable', 'scandaleux',
            'désastreux', 'consternant', 'navrant', 'épouvantable', 'inadmissible',
            'inacceptable', 'médiocre', 'décevant', 'frustrant', 'ennuyeux', 'pénible',
            'triste', 'malheureux', 'déprimé', 'furieux', 'en colère', 'mécontent',
            'insatisfait', 'dégoûté', 'choqué', 'outré', 'regret', 'regretter',
            'détester', 'haïr', 'éviter', 'fuir', 'arnaque', 'vol', 'honte',
            'cauchemar', 'désastre', 'échec', 'raté', 'pourri', 'merde'
        }
        
        # Load and train model
        self.load_data()
        self.train()

    def preprocess_text(self, text):
        """Enhanced text preprocessing with better French handling"""
        if not text or pd.isna(text):
            return ""
        
        # Convert to string and lowercase
        text = str(text).lower().strip()
        
        # Handle common contractions and negations first (before removing accents)
        text = re.sub(r"\bn'", " ne ", text)  # n'est -> ne est
        text = re.sub(r"\bj'", "je ", text)   # j'ai -> je ai
        text = re.sub(r"\bl'", "le ", text)   # l'endroit -> le endroit
        text = re.sub(r"\bd'", "de ", text)   # d'accord -> de accord
        text = re.sub(r"\bqu'", "que ", text) # qu'il -> que il
        
        # Remove accents after handling contractions
        text = unidecode.unidecode(text)
        
        # Enhanced punctuation handling
        text = re.sub(r'[!]{2,}', ' EXCLAMATION_MULTIPLE ', text)
        text = re.sub(r'[?]{2,}', ' QUESTION_MULTIPLE ', text)
        text = re.sub(r'[.]{3,}', ' DOTS_MULTIPLE ', text)
        text = re.sub(r'[:;]{2,}', ' PUNCTUATION_MULTIPLE ', text)
        
        # Handle emphasis patterns
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)  # Reduce repeated characters
        
        # Keep letters, numbers, and basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s!?.,;:\-]', ' ', text)
        
        # Handle negation patterns
        negation_pattern = r'\b(ne|n|pas|non|jamais|rien|aucun|nullement|point)\b'
        if re.search(negation_pattern, text):
            text = text + ' NEGATION_PRESENT'
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def load_data(self):
        """Load and preprocess dataset with better error handling"""
        try:
            # Read dataset with proper encoding
            self.dataset = pd.read_csv(
                self.dataset_path, 
                names=['sentence', 'label'], 
                sep=self.separator,
                encoding='utf-8',
                quoting=3  # QUOTE_NONE to handle special characters
            )
            
            print(f"Dataset loaded: {len(self.dataset)} samples")
            
            # Clean data
            self.dataset = self.dataset.dropna()
            self.dataset = self.dataset[self.dataset['sentence'].str.strip() != '']
            
            # Ensure labels are integers
            self.dataset['label'] = pd.to_numeric(self.dataset['label'], errors='coerce')
            self.dataset = self.dataset.dropna()
            self.dataset['label'] = self.dataset['label'].astype(int)
            
            # Preprocess sentences
            self.dataset['processed_sentence'] = self.dataset['sentence'].apply(self.preprocess_text)
            
            # Remove empty processed sentences
            self.dataset = self.dataset[self.dataset['processed_sentence'].str.strip() != '']
            
            # Check class distribution
            label_counts = self.dataset['label'].value_counts()
            print(f"Class distribution: {dict(label_counts)}")
            
            # Check for class imbalance
            if len(label_counts) < 2:
                raise ValueError("Dataset must contain both positive and negative examples")
            
            min_class_size = min(label_counts.values)
            max_class_size = max(label_counts.values)
            imbalance_ratio = max_class_size / min_class_size
            
            if imbalance_ratio > 2:
                print(f"Warning: Class imbalance detected (ratio: {imbalance_ratio:.2f})")
            
            print(f"Final dataset size: {len(self.dataset)} samples")
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise

    def create_features(self, texts):
        """Create enhanced features from text"""
        features = []
        
        for text in texts:
            feature_dict = {}
            
            # Basic length features
            feature_dict['char_count'] = len(text)
            feature_dict['word_count'] = len(text.split())
            feature_dict['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
            
            # Punctuation features
            feature_dict['exclamation_count'] = text.count('!')
            feature_dict['question_count'] = text.count('?')
            feature_dict['caps_ratio'] = sum(1 for c in text if c.isupper()) / max(len(text), 1)
            
            # Sentiment word features
            words = set(text.lower().split())
            feature_dict['positive_words'] = len(words.intersection(self.positive_words))
            feature_dict['negative_words'] = len(words.intersection(self.negative_words))
            feature_dict['sentiment_ratio'] = (feature_dict['positive_words'] - feature_dict['negative_words']) / max(len(words), 1)
            
            # Negation features
            feature_dict['has_negation'] = 1 if 'NEGATION_PRESENT' in text else 0
            feature_dict['negation_count'] = len(re.findall(r'\b(ne|pas|non|jamais|rien|aucun)\b', text.lower()))
            
            # Intensity features
            feature_dict['has_multiple_exclamation'] = 1 if 'EXCLAMATION_MULTIPLE' in text else 0
            feature_dict['has_multiple_question'] = 1 if 'QUESTION_MULTIPLE' in text else 0
            
            # Specific French patterns
            feature_dict['has_very'] = 1 if any(word in text.lower() for word in ['tres', 'vraiment', 'tellement', 'extremement']) else 0
            feature_dict['has_recommendation'] = 1 if any(word in text.lower() for word in ['recommande', 'conseille', 'suggere']) else 0
            
            features.append(list(feature_dict.values()))
        
        return np.array(features)

    def train(self):
        """Enhanced training with better model selection"""
        try:
            # Prepare data
            sentences = self.dataset['processed_sentence'].values
            y = self.dataset['label'].values
            
            # Stratified split
            X_train_text, X_test_text, y_train, y_test = train_test_split(
                sentences, y, test_size=0.2, random_state=42, stratify=y
            )
            
            print("Training improved model...")
            
            # Optimized TF-IDF vectorizer
            self.vectorizer = TfidfVectorizer(
                max_features=5000,  # Reduced to prevent overfitting
                ngram_range=(1, 2),  # Reduced to bigrams only
                min_df=2,
                max_df=0.9,
                stop_words=list(self.french_stopwords),
                sublinear_tf=True,
                use_idf=True,
                lowercase=True,
                token_pattern=r'\b[a-zA-Z]{2,}\b'  # Only words with 2+ letters
            )
            
            # Fit vectorizer and transform text
            X_train_tfidf = self.vectorizer.fit_transform(X_train_text)
            X_test_tfidf = self.vectorizer.transform(X_test_text)
            
            # Create additional features
            X_train_features = self.create_features(X_train_text)
            X_test_features = self.create_features(X_test_text)
            
            # Combine features
            from scipy.sparse import hstack
            X_train_combined = hstack([X_train_tfidf, X_train_features])
            X_test_combined = hstack([X_test_tfidf, X_test_features])
            
            # Better model configurations
            models = {
                'lr': LogisticRegression(
                    C=0.5,  # Reduced regularization
                    max_iter=2000,
                    random_state=42,
                    class_weight='balanced',
                    solver='liblinear'
                ),
                'rf': RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    class_weight='balanced'
                ),
                'xgb': XGBClassifier(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    eval_metric='logloss',
                    scale_pos_weight=1
                )
            }
            
            # Train and evaluate models
            best_model = None
            best_score = 0
            model_scores = {}
            
            for name, model in models.items():
                print(f"\nTraining {name}...")
                model.fit(X_train_combined, y_train)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train_combined, y_train, cv=5, scoring='accuracy')
                print(f"{name} CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
                
                # Test accuracy
                y_pred = model.predict(X_test_combined)
                test_accuracy = accuracy_score(y_test, y_pred)
                print(f"{name} Test Accuracy: {test_accuracy:.4f}")
                
                model_scores[name] = test_accuracy
                
                if test_accuracy > best_score:
                    best_score = test_accuracy
                    best_model = model
            
            # Use the best single model (ensemble often overfits on small datasets)
            self.model = best_model
            self.score = best_score
            
            print(f"\nBest model: {max(model_scores, key=model_scores.get)}")
            print(f"Best accuracy: {best_score:.4f}")
            
            # Final evaluation
            y_pred_final = self.model.predict(X_test_combined)
            print(f"\nFinal Model Accuracy: {self.score:.2%}")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred_final, target_names=['Negative', 'Positive']))
            
            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred_final)
            print(f"\nConfusion Matrix:")
            print(f"True Neg: {cm[0,0]}, False Pos: {cm[0,1]}")
            print(f"False Neg: {cm[1,0]}, True Pos: {cm[1,1]}")
            
            # Test with some examples
            print("\nTesting with sample phrases:")
            test_phrases = [
                "Je suis très triste",
                "C'est excellent",
                "Je suis déçu",
                "Fantastique!",
                "Pas terrible",
                "J'adore"
            ]
            
            for phrase in test_phrases:
                result = self.predict(phrase)
                if isinstance(result, tuple):
                    sentiment, confidence = result
                    print(f"'{phrase}' -> {sentiment} (confidence: {confidence:.3f})")
                else:
                    print(f"'{phrase}' -> {result}")
            
            print("\nModel trained successfully!")
            
        except Exception as e:
            print(f"Error during training: {e}")
            raise

    def predict(self, sentence):
        """Enhanced prediction with confidence score"""
        try:
            if not sentence or pd.isna(sentence):
                return "ERROR"
            
            # Preprocess sentence
            processed_sentence = self.preprocess_text(sentence)
            
            if not processed_sentence.strip():
                return "NEUTRAL"
            
            # Transform text
            text_tfidf = self.vectorizer.transform([processed_sentence])
            
            # Create additional features
            additional_features = self.create_features([processed_sentence])
            
            # Combine features
            from scipy.sparse import hstack
            combined_features = hstack([text_tfidf, additional_features])
            
            # Make prediction
            prediction = self.model.predict(combined_features)[0]
            
            # Get confidence if available
            confidence = None
            if hasattr(self.model, 'predict_proba'):
                try:
                    probabilities = self.model.predict_proba(combined_features)[0]
                    confidence = max(probabilities)
                except:
                    pass
            
            # Return result
            result = "POSITIVE" if prediction == 1 else "NEGATIVE"
            
            if confidence is not None:
                return result, confidence
            else:
                return result
                
        except Exception as e:
            print(f"Error during prediction: {e}")
            return "ERROR"

    def save_model(self, filepath):
        """Save the trained model"""
        try:
            model_data = {
                'model': self.model,
                'vectorizer': self.vectorizer,
                'score': self.score,
                'french_stopwords': self.french_stopwords,
                'positive_words': self.positive_words,
                'negative_words': self.negative_words
            }
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"Model saved to {filepath}")
        except Exception as e:
            print(f"Error saving model: {e}")

    def load_model(self, filepath):
        """Load a pre-trained model"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.vectorizer = model_data['vectorizer']
            self.score = model_data['score']
            self.french_stopwords = model_data['french_stopwords']
            self.positive_words = model_data.get('positive_words', set())
            self.negative_words = model_data.get('negative_words', set())
            
            print(f"Model loaded from {filepath}")
            print(f"Model accuracy: {self.score:.2%}")
        except Exception as e:
            print(f"Error loading model: {e}")


def main():
    # Initialize and train model
    classifier = ImprovedSentimentClassifier()
    
    # Save model for later use
    classifier.save_model('improved_sentiment_model.pkl')
    
    # Create Flask app
    app = Flask(__name__)
    
    @app.route('/predict', methods=['POST'])
    def predict():
        try:
            data = request.get_json()
            sentence = data.get('sentence', '')
            
            if not sentence:
                return jsonify({'error': 'No sentence provided'}), 400
            
            # Predict sentiment
            result = classifier.predict(sentence)
            
            if isinstance(result, tuple):
                sentiment, confidence = result
                return jsonify({
                    'sentence': sentence,
                    'sentiment': sentiment,
                    'confidence': float(confidence),
                    'model_accuracy': float(classifier.score)
                })
            else:
                return jsonify({
                    'sentence': sentence,
                    'sentiment': result,
                    'model_accuracy': float(classifier.score)
                })
                
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/health', methods=['GET'])
    def health():
        return jsonify({
            'status': 'healthy',
            'model_accuracy': float(classifier.score)
        })
    
    @app.route('/test', methods=['GET'])
    def test():
        """Test endpoint with common French phrases"""
        test_phrases = [
            "Je suis très triste",
            "C'est excellent",
            "Je suis déçu",
            "Fantastique!",
            "Pas terrible",
            "J'adore",
            "C'est nul",
            "Parfait!",
            "Je recommande",
            "Une catastrophe"
        ]
        
        results = []
        for phrase in test_phrases:
            result = classifier.predict(phrase)
            if isinstance(result, tuple):
                sentiment, confidence = result
                results.append({
                    'sentence': phrase,
                    'sentiment': sentiment,
                    'confidence': float(confidence)
                })
            else:
                results.append({
                    'sentence': phrase,
                    'sentiment': result
                })
        
        return jsonify({'test_results': results})
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)


if __name__ == "__main__":
    main()