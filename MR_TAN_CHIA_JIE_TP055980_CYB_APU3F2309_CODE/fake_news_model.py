import os
os.environ['GENSIM_NO_COMPILED_EXTENSIONS'] = '1'
import gensim
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from gensim.models import Word2Vec
from transformers import BertTokenizer, BertModel
import torch
from nltk.tokenize import word_tokenize
import nltk
import joblib
from datetime import datetime

nltk.download('punkt')

data = pd.read_csv('MalaysiaNews.csv')
data['text'] = data['title'] + ' ' + data['content']

with ProcessPoolExecutor(max_workers=5) as executor:
    tokenized_text = [word_tokenize(text.lower()) for text in data['text']]
    word2vec_model = Word2Vec(sentences=tokenized_text, vector_size=100, window=5, min_count=1, workers=4)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

def get_word2vec_features(text):
    words = word_tokenize(text.lower())
    word_vectors = [word2vec_model.wv[word] for word in words if word in word2vec_model.wv]
    return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(100)

def get_bert_features(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

class Word2VecTransformer:
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return np.array([get_word2vec_features(text) for text in X])

class BertTransformer:
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return np.array([get_bert_features(text) for text in X])

preprocessor = ColumnTransformer(
    transformers=[
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000), 'text'),
        ('word2vec', Word2VecTransformer(), 'text'),
        ('bert', BertTransformer(), 'text'),
        ('year', OneHotEncoder(sparse_output=False), ['year'])
    ])

ensemble_model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

X = data[['text', 'year']]
y = data['boolean']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(ensemble_model, param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))

joblib.dump(best_model, 'fake_news_model.joblib')

def predict_fake_news(title, content, year):
    model = joblib.load('fake_news_model.joblib')
    input_data = pd.DataFrame({'text': [title + ' ' + content], 'year': [year]})
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0]
    
    if prediction[0]:
        result = "Real"
        confidence = probability[1]
    else:
        result = "Fake"
        confidence = probability[0]
    
    return result, confidence

def update_model(new_data):
    current_model = joblib.load('fake_news_model.joblib')
    X_new = new_data[['text', 'year']]
    y_new = new_data['boolean']
    
    current_model.fit(X_new, y_new)
    joblib.dump(current_model, f'fake_news_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.joblib')
