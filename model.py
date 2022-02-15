import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
import re


def preprocess_data(text):
    text = text.lower()
    new_text = re.sub('[^a-zA-z0-9\s]', '', text)
    new_text = re.sub('rt', '', new_text)
    return new_text


def my_pipeline(text):
    text_new = preprocess_data(text)
    x = tokenizer.texts_to_sequences(pd.Series(text_new).values)
    x = pad_sequences(x, maxlen=28)
    return x


data = pd.read_csv('data/Sentiment.csv')
tokenizer = Tokenizer(num_words=2000, split=' ')
tokenizer.fit_on_texts(data['text'].values)

# Keeping only the necessary columns
data = data[['text', 'sentiment']]

data['text'] = data['text'].apply(preprocess_data)
max_features = 2000

tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(data['text'].values)
X = tokenizer.texts_to_sequences(data['text'].values)
X = pad_sequences(X, 28)

Y = pd.get_dummies(data['sentiment']).values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20)
embed_dim = 128
lstm_out = 196

model = Sequential()
model.add(Embedding(max_features, embed_dim, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.3, recurrent_dropout=0.2, return_sequences=True))
model.add(LSTM(128, recurrent_dropout=0.2))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

batch_size = 512
model.fit(X_train, Y_train, epochs=10, batch_size=batch_size, validation_data=(X_test, Y_test))

model.save('sentiment.h5')
