import tensorflow as tf
from keras.datasets import imdb
import pandas as pd
import numpy as np
from keras.layers import LSTM, Activation, Dropout, Dense, Input, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from keras.models import Model
import string
import re
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.sequence import pad_sequences
import keras
from sklearn.model_selection import train_test_split

def remove_stopwords(data):
  data['review without stopwords'] = data['review'].apply(lambda x : ' '.join([word for word in x.split() if word not in (stopwords)]))
  return data

def remove_tags(string):
    result = re.sub('<.*?>','',string)
    return result

def read_glove_vector(glove_vec):
    with open(glove_vec, 'r', encoding='UTF-8') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            w_line = line.split()
            curr_word = w_line[0]
            word_to_vec_map[curr_word] = np.array(w_line[1:], dtype=np.float64)

    return word_to_vec_map

def imdb_rating(input_shape):

  X_indices = Input(input_shape)

  embeddings = embedding_layer(X_indices)

  X = LSTM(128, return_sequences=True)(embeddings)

  X = Dropout(0.6)(X)

  X = LSTM(128, return_sequences=True)(X)

  X = Dropout(0.6)(X)

  X = LSTM(128)(X)

  X = Dense(1, activation='sigmoid')(X)

  model = Model(inputs=X_indices, outputs=X)

  return model

def conv1d_model(input_shape):
    X_indices = Input(input_shape)

    embeddings = embedding_layer(X_indices)

    X = Conv1D(512, 3, activation='relu')(embeddings)

    X = MaxPooling1D(3)(X)

    X = Conv1D(256, 3, activation='relu')(X)

    X = MaxPooling1D(3)(X)

    X = Conv1D(256, 3, activation='relu')(X)
    X = Dropout(0.8)(X)
    X = MaxPooling1D(3)(X)

    X = GlobalMaxPooling1D()(X)

    X = Dense(256, activation='relu')(X)
    X = Dense(1, activation='sigmoid')(X)

    model = Model(inputs=X_indices, outputs=X)

    return model

def add_score_predictions(data, reviews_list_idx):
    data['sentiment score'] = 0

    reviews_list_idx = pad_sequences(reviews_list_idx, maxlen=maxLen, padding='post')

    review_preds = model.predict(reviews_list_idx)

    data['sentiment score'] = review_preds

    pred_sentiment = np.array(list(map(lambda x: 'positive' if x > 0.5 else 'negative', review_preds)))

    data['predicted sentiment'] = 0

    data['predicted sentiment'] = pred_sentiment

    return data


if __name__ == '__main__':
    data = pd.read_csv('/kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv')
    data['review'] = data['review'].str.lower()
    stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at",
                 "be", "because",
                 "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does",
                 "doing", "down", "during",
                 "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's",
                 "her", "here",
                 "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm",
                 "i've", "if", "in", "into",
                 "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on",
                 "once", "only", "or",
                 "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll",
                 "she's", "should",
                 "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
                 "there", "there's",
                 "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too",
                 "under", "until", "up",
                 "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's",
                 "where", "where's",
                 "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll",
                 "you're", "you've",
                 "your", "yours", "yourself", "yourselves"]
    data_without_stopwords = remove_stopwords(data)
    data_without_stopwords['clean_review']= data_without_stopwords['review without stopwords'].apply(lambda cw : remove_tags(cw))
    data_without_stopwords['clean_review'] = data_without_stopwords['clean_review'].str.replace('[{}]'.format(string.punctuation), ' ')
    data_without_stopwords.head()

    reviews = data_without_stopwords['clean_review']

    reviews_list = []
    for i in range(len(reviews)):
        reviews_list.append(reviews[i])
    sentiment = data_without_stopwords['sentiment']
    y = np.array(list(map(lambda x: 1 if x == "positive" else 0, sentiment)))
    X_train, X_test, Y_train, Y_test = train_test_split(reviews_list, y, test_size=0.2, random_state=45)

    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(X_train)
    words_to_index = tokenizer.word_index

    word_to_vec_map = read_glove_vector('/content/drive/My Drive/glove.6B.50d.txt')
    maxLen = 150
    vocab_len = len(words_to_index)
    embed_vector_len = word_to_vec_map['moon'].shape[0]

    emb_matrix = np.zeros((vocab_len, embed_vector_len))

    for word, index in words_to_index.items():
        embedding_vector = word_to_vec_map.get(word)
        if embedding_vector is not None:
            emb_matrix[index, :] = embedding_vector

    embedding_layer = Embedding(input_dim=vocab_len, output_dim=embed_vector_len, input_length=maxLen,
                                weights=[emb_matrix], trainable=False)

    model = imdb_rating((maxLen,))
    print(model.summary())

    model_1d = conv1d_model((maxLen,))
    print(model_1d.summary())

    X_train_indices = tokenizer.texts_to_sequences(X_train)
    X_train_indices = pad_sequences(X_train_indices, maxlen=maxLen, padding='post')
    print(X_train_indices.shape)
    adam = tf.optimizers.Adam(learning_rate=0.0001)
    model_1d.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    model_1d.fit(X_train_indices, Y_train, batch_size=64, epochs=15)
    print(model_1d)
    adam = tf.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train_indices, Y_train, batch_size=64, epochs=15)

    X_test_indices = tokenizer.texts_to_sequences(X_test)

    X_test_indices = pad_sequences(X_test_indices, maxlen=maxLen, padding='post')
    model.evaluate(X_test_indices, Y_test)
    model_1d.evaluate(X_test_indices, Y_test)
    preds = model_1d.predict(X_test_indices)

    n = np.random.randint(0, 9999)
    if preds[n] > 0.5:
        print('predicted sentiment : positive')
    else:
        print('precicted sentiment : negative')
    if Y_test[n] == 1:
        print('correct sentiment : positive')
    else:
        print('correct sentiment : negative')
    print(preds[n])
    print(Y_test[n])

    model_1d.save_weights('/kaggle/working/imdb_weights_con1vd.hdf5')
    reviews_list_idx = tokenizer.texts_to_sequences(reviews_list)

    data = add_score_predictions(data, reviews_list_idx)
    print(data)

