import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import re

from tensorflow.keras.layers import Dense, LSTM, Input, Dropout, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences

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

count_games = len(games)
count_politics = len(politics)
count_science = len(science)
count_technology = len(technology)

total_lines = count_games + count_politics + count_science + count_technology
print(total_lines)

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


X = data_pad
Y = np.array([[1, 0, 0, 0]]*count_games + [[0, 1, 0, 0]]*count_politics + [[0, 0, 1, 0]]*count_science + [[0, 0, 0, 1]]*count_technology)
print(X.shape, Y.shape)

indeces = np.random.choice(X.shape[0], size=X.shape[0], replace=False)
X = X[indeces]
Y = Y[indeces]


model = Sequential()
model.add(Embedding(maxWordsCount, 128, input_length = max_text_len))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(64))
model.add(Dense(4, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam(0.0001))

history = model.fit(X, Y, batch_size=1, epochs=50)

reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))

def sequence_to_text(list_of_indices):
    words = [reverse_word_map.get(letter) for letter in list_of_indices]
    return(words)

model.save("model.keras")

t = "Apple поддержала право на ремонт в Калифорнии".lower()
data = tokenizer.texts_to_sequences([t])
data_pad = pad_sequences(data, maxlen=max_text_len)
print( sequence_to_text(data[0]) )

res = model.predict(data_pad)
print(res, np.argmax(res), sep='\n')