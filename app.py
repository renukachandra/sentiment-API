import numpy as np
from fastapi import FastAPI, Form
from starlette.responses import HTMLResponse
import tensorflow as tf
import uvicorn

from model import my_pipeline

app = FastAPI()


@app.get('/')
def basic_view():
    return {"WELCOME": "GO TO /docs route, or /post or send post request to /predict "}


@app.get('/predict', response_class=HTMLResponse)
def take_inp():
    return '''
        <form method="post">
        <input maxlength="28" name="text" type="text" value="Text Emotion to be tested" />
        <input type="submit" />'''


@app.post('/predict')
def predict(text: str = Form(...)):
    clean_text = my_pipeline(text)  # clean, and preprocess the text through pipeline
    loaded_model = tf.keras.models.load_model('sentiment.h5')  # load the saved model
    predictions = loaded_model.predict(clean_text)  # predict the text
    sentiment = int(np.argmax(predictions))  # calculate the index of max sentiment
    probability = max(predictions.tolist()[0])  # calculate the probability
    if sentiment == 0:
        t_sentiment = 'negative'  # set appropriate sentiment
    elif sentiment == 1:
        t_sentiment = 'neutral'
    elif sentiment == 2:
        t_sentiment = 'positive'
    return {  # return the dictionary for endpoint
        "ACTUAL SENTENCE": text,
        "PREDICTED SENTIMENT": t_sentiment,
        "Probability": probability
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5049)
