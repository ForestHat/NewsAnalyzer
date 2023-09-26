import numpy as np
import re

from tensorflow.keras.layers import Dense, LSTM, Input, Dropout, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow

with open('games.txt', 'r', encoding='utf-8') as f:
    games = f.readlines()
    games[0] = games[0].replace('\ufeff', '') #убираем первый невидимый символ

with open('politics.txt', 'r', encoding='utf-8') as f:
    politics = f.readlines()
    politics[0] = politics[0].replace('\ufeff', '') #убираем первый невидимый символ

with open('science.txt', 'r', encoding='utf-8') as f:
    science = f.readlines()
    science[0] = science[0].replace('\ufeff', '') #убираем первый невидимый символ

with open('technology.txt', 'r', encoding='utf-8') as f:
    technology = f.readlines()
    technology[0] = technology[0].replace('\ufeff', '') #убираем первый невидимый символ

texts = games + politics + science + technology


maxWordsCount = 2000
tokenizer = Tokenizer(num_words=maxWordsCount, filters='!–"—#$%&amp;()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r«»', lower=True, split=' ', char_level=False)
tokenizer.fit_on_texts(texts)

dist = list(tokenizer.word_counts.items())
print(dist[:10])
print(texts[0][:100])


max_text_len = 20
data = tokenizer.texts_to_sequences(texts)
data_pad = pad_sequences(data, maxlen=max_text_len)
print(data_pad)

print( list(tokenizer.word_index.items()) )

def create_model():
    model = Sequential()
    model.add(Embedding(maxWordsCount, 128, input_length = max_text_len))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(64))
    model.add(Dense(4, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam(0.0001))

    return model

model = create_model()
model = tensorflow.keras.models.load_model("model.keras")

reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))

def sequence_to_text(list_of_indices):
    words = [reverse_word_map.get(letter) for letter in list_of_indices]
    return(words)

t = "Баг в Starfield приводит к тому, что за игроком начинают летать различные объекты от астероидов до целых поселений".lower()
data = tokenizer.texts_to_sequences([t])
data_pad = pad_sequences(data, maxlen=max_text_len)
print( sequence_to_text(data[0]) )

res = model.predict(data_pad)
print(res, np.argmax(res), sep='\n')