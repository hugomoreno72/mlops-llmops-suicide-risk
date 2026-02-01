import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI
from transformers import pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

import funciones 

app = FastAPI(title="Clasificación de Riesgo para comentarios sospechosos de tendencia suicida")

mlflow.set_tracking_uri("http://127.0.0.1:5000")
model_name = 'suicide-opt-model'
model_mlflow = mlflow.pyfunc.load_model(f'models:/{model_name}/Production')


sentiment_pipe = pipeline("sentiment-analysis", model="pysentimiento/robertuito-sentiment-analysis")
emotion_pipe = pipeline("text-classification", model="pysentimiento/robertuito-emotion-analysis")

df = pd.read_json("corpus_mod.json")
X = df['text_processed'] 
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=0)

cv = TfidfVectorizer(ngram_range=(1, 3), max_df=0.8, min_df=5)
cv.fit(X_train)

print(f"INFO: API Vectorizer features: {len(cv.get_feature_names_out())}")

@app.get("/")
def root():

    return {"message": "Clasificación de Riesgo para comentarios sospechosos de tendencia suicida"}


@app.get("/predict")
def predict_risk(text: str):

    processed = funciones.tokenize_lemmatize(funciones.preprocessing(text))
    vectorized = cv.transform([processed])
    prediction = model_mlflow.predict(vectorized)
    return {
        "text_original": text,
        "prediction": int(prediction[0]),
        "label": "Riesgo Detectado" if prediction[0] == 1 else "Sin Riesgo"
    }


@app.get("/hf-sentiment")
def get_sentiment(text: str):

    result = sentiment_pipe(text)
    return {"sentiment_analysis": result}


@app.get("/hf-emotion")
def get_emotion(text: str):

    result = emotion_pipe(text)
    
    return {
        "text": text,
        "emotion_analysis": result[0]
    }


@app.get("/model-info")
def get_info():

    return {
        "model_name": model_name,
        "mlflow_experiment": "suicide_analysis",
        "frameworks": ["FastAPI", "MLflow", "Scikit-learn", "Hugging Face"]
    }
