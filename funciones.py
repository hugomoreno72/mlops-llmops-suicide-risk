import pandas as pd
import re
import argparse
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score


MLFLOW_TRACKING_URI = 'http://localhost:5000'
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)


stopwords_es = set(stopwords.words('spanish'))
lemmatizer = WordNetLemmatizer()

def preprocessing(text):
    if not text or str(text).lower() in ['none', 'nan']:
        return ""

    text = BeautifulSoup(text, "html5lib").get_text()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r"[^a-zñáéíóúü]", " ", text.lower())
    text = re.sub(r'\d+', '', text)
    return text

def tokenize_lemmatize(text):
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords_es]
    lemas = []
    for word, tag in nltk.pos_tag(tokens):
        pos = 'v' if tag.startswith('V') else 'n'
        lemas.append(lemmatizer.lemmatize(word, pos=pos))
    return " ".join(lemas)

def train_model(c, max_df, min_df):

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment('suicide_analysis')
    
    try:
        df = pd.read_json("corpus_mod.json")
    except FileNotFoundError:
        print("Error: No se encontró 'corpus_mod.json'.")
        return
    
    X = df['text_processed'] 
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=0)

    cv = TfidfVectorizer(ngram_range=(1, 3), max_df=max_df, min_df=min_df)
    X_train_tfidf = cv.fit_transform(X_train)
    X_val_tfidf = cv.transform(X_val)
    X_test_tfidf = cv.transform(X_test)

    with mlflow.start_run(run_name=f"Script_C_{c}"):

        model = LogisticRegression(C=c, max_iter=1000, random_state=0)
        model.fit(X_train_tfidf, y_train)
        

        accuracy_train = accuracy_score(y_train, model.predict(X_train_tfidf))
        accuracy_val = accuracy_score(y_val, model.predict(X_val_tfidf))
        accuracy_test = accuracy_score(y_test, model.predict(X_test_tfidf))

        mlflow.log_params({"C": c, "max_df": max_df, "min_df": min_df})
        mlflow.log_metric("accuracy_train", accuracy_train)
        mlflow.log_metric("accuracy_val", accuracy_val)
        mlflow.log_metric("accuracy_test", accuracy_test)
        mlflow.sklearn.log_model(model, "model")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--c", type=float, default=10.0, help="Regularización C de la linear regression")
    parser.add_argument("--max_df", type=float, default=0.8, help="Filtra términos que aparecen en muchos registros (siendo max_df del 0 al 1)")
    parser.add_argument("--min_df", type=int, default=5, help="Filtra términos raros que aparecen en menos de min_df registros")

    args = parser.parse_args()
    train_model(args.c, args.max_df, args.min_df)