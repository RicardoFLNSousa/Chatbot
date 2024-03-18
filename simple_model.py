import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Flatten
from sklearn.preprocessing import LabelEncoder
import preprocessing
import os
from keras.models import load_model
import numpy as np
import pickle

# transform the questions and answers into tokens and create sequences
def get_sequences(df):
    train_data = df.Perguntas.to_list()
    train_labels = df.Respostas.to_list()

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(train_labels)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_data)
    train_sequences = tokenizer.texts_to_sequences(train_data)
    train_sequences = pad_sequences(train_sequences)

    model_info_dict = {'train_sequences':train_sequences,
                        'encoded_labels':encoded_labels,
                        'label_encoder':label_encoder,
                        'tokenizer':tokenizer,
                        'train_labels':train_labels}
    
    with open(os.getcwd()+'/models/model_info_dict_simple.pkl', 'wb') as f:
            pickle.dump(model_info_dict, f)

    return train_sequences, encoded_labels, label_encoder, tokenizer, train_labels

# train a simple NN
def train_model(train_sequences, encoded_labels, tokenizer, train_labels):
    model = Sequential()

    model.add(Embedding(len(tokenizer.word_index) + 1, 100))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(len(train_labels), activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_sequences, encoded_labels, epochs=50)
    model.save(os.getcwd()+"/models/training_model_simple.h5")

    return model

def generate_response(input_text, model_path):

    with open(model_path+'model_info_dict_simple.pkl', 'rb') as f:
        model_info_dict = pickle.load(f)

    train_sequences = model_info_dict['train_sequences']
    label_encoder = model_info_dict['label_encoder']
    tokenizer = model_info_dict['tokenizer']

    model = load_model(os.getcwd()+"/models/training_model_simple.h5")

    df_temp = pd.DataFrame.from_dict({'col': [input_text]})
    df = preprocessing.preprocess_dataframe(df_temp)
    input_text = df['col'].iloc[0]

    sequence = tokenizer.texts_to_sequences([input_text])
    sequence = pad_sequences(sequence, maxlen=train_sequences.shape[1])
    prediction = model.predict(sequence)
    predicted_label = np.argmax(prediction)
    response = label_encoder.inverse_transform([predicted_label])[0]
    return response