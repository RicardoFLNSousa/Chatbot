import pandas as pd
import tensorflow as tf
from keras.layers import Input, LSTM, Dense
from keras.models import Model
from sklearn.preprocessing import LabelEncoder
import regex as re
import numpy as np
import pickle
from keras.models import load_model
import preprocessing


# function to create the matrixes used in the model, the vocabulary and the padding
def get_encoder_and_decoder_matrix(df):
    perguntas = []
    respostas = []
    perguntas_tokens = set()
    respostas_tokens = set()
    for index,row in df.iloc[:500].iterrows():
        input_doc, target_doc = row['Perguntas'], row['Respostas']
        # appending each input sentence to the questions
        perguntas.append(input_doc)
        # add start of sentence and end of sentence tokens to the answer
        target_doc = '<START> ' + target_doc + ' <END>'
        respostas.append(target_doc)
        
        # split each sentence into tokens to create the vocabulary
        for token in re.findall(r"[\w']+|[^\s\w]", input_doc):
            if token not in perguntas_tokens:
                perguntas_tokens.add(token)
        for token in target_doc.split():
            if token not in respostas_tokens:
                respostas_tokens.add(token)

    # get the tokens and size of the list of tokens
    perguntas_tokens = sorted(list(perguntas_tokens))
    respostas_tokens = sorted(list(respostas_tokens))
    num_perguntas_tokens = len(perguntas_tokens)
    num_respostas_tokens = len(respostas_tokens)

    # encode the tokens
    perguntas_dict = dict([(token, i) for i, token in enumerate(perguntas_tokens)])
    respostas_dict = dict([(token, i) for i, token in enumerate(respostas_tokens)])

    # reverse the dictionary of tokens so the key is the enconded value
    inverted_perguntas_dict = dict((i, token) for token, i in perguntas_dict.items())
    inverted_respostas_dict = dict((i, token) for token, i in respostas_dict.items())

    # max size of the questions and answers
    max_encoder_seq_length = max([len(re.findall(r"[\w']+|[^\s\w]", pergunta)) for pergunta in perguntas])
    max_decoder_seq_length = max([len(re.findall(r"[\w']+|[^\s\w]", resposta)) for resposta in respostas])

    # create the matrixes for the encoder input and decoder input and output, with the respective sizes with all zeros
    encoder_input_data = np.zeros((len(perguntas), max_encoder_seq_length, num_perguntas_tokens),dtype='float32')
    decoder_input_data = np.zeros((len(perguntas), max_decoder_seq_length, num_respostas_tokens),dtype='float32')
    decoder_target_data = np.zeros((len(perguntas), max_decoder_seq_length, num_respostas_tokens),dtype='float32')

    # change to 1 for each token
    for line, (input_doc, target_doc) in enumerate(zip(perguntas, respostas)):
        for timestep, token in enumerate(re.findall(r"[\w']+|[^\s\w]", input_doc)):
            #Assign 1. for the current line, timestep, & word in encoder_input_data
            encoder_input_data[line, timestep, perguntas_dict[token]] = 1.
        
        for timestep, token in enumerate(target_doc.split()):
            decoder_input_data[line, timestep, respostas_dict[token]] = 1.
            if timestep > 0:
                decoder_target_data[line, timestep - 1, respostas_dict[token]] = 1.


    model_info_dict = {'encoder_input_data':encoder_input_data,
                        'decoder_input_data':decoder_input_data,
                        'decoder_target_data':decoder_target_data,
                        'num_perguntas_tokens':num_perguntas_tokens,
                        'num_respostas_tokens':num_respostas_tokens,
                        'inverted_perguntas_dict':inverted_perguntas_dict,
                        'inverted_respostas_dict':inverted_respostas_dict,
                        'perguntas_dict':perguntas_dict,
                        'respostas_dict':respostas_dict,
                        'max_encoder_seq_length':max_encoder_seq_length,
                        'max_decoder_seq_length':max_decoder_seq_length}

    return model_info_dict

# train the model with the questions and answers
def train_model(model_info_dict, save_model = None):

    num_encoder_tokens = model_info_dict['num_perguntas_tokens']
    num_decoder_tokens = model_info_dict['num_respostas_tokens']
    encoder_input_data = model_info_dict['encoder_input_data']
    decoder_input_data = model_info_dict['decoder_input_data']
    decoder_target_data = model_info_dict['decoder_target_data']

    # initial dim
    dimensionality = 256
    # training epochs and batch size 
    batch_size = 10
    epochs = 50
    # Encoder
    encoder_inputs = Input(shape=(None, num_encoder_tokens))
    encoder_lstm = LSTM(dimensionality, return_state=True)
    encoder_outputs, state_hidden, state_cell = encoder_lstm(encoder_inputs)
    encoder_states = [state_hidden, state_cell]
    # Decoder
    decoder_inputs = Input(shape=(None, num_decoder_tokens))
    decoder_lstm = LSTM(dimensionality, return_sequences=True, return_state=True)
    decoder_outputs, decoder_state_hidden, decoder_state_cell = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # mode
    training_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    training_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    training_model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size = batch_size, epochs = epochs, validation_split = 0.2)

     # save the model
    if save_model is not None:
        training_model.save(save_model+'training_model.h5')

        with open(save_model+'model_info_dict.pkl', 'wb') as f:
            pickle.dump(model_info_dict, f)

    return training_model, model_info_dict


# create the inference model that will take into account only the question, since we do not know the answer
def create_inference_model(trained_model):
    # encoder model. use the hidden states and hidden cells from the encoder layer of the previous model
    encoder_inputs = trained_model.input[0]
    encoder_outputs, state_h_enc, state_c_enc = trained_model.layers[2].output
    encoder_states = [state_h_enc, state_c_enc]
    encoder_model = Model(encoder_inputs, encoder_states)

    # decoder model. create the decoder model with the help of the decoder of the previous model.
    latent_dim = 256
    decoder_state_input_hidden = Input(shape=(latent_dim,))
    decoder_state_input_cell = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_hidden, decoder_state_input_cell]

    decoder_lstm = trained_model.layers[3]
    decoder_inputs = trained_model.layers[3].input[0]
    decoder_outputs, state_hidden, state_cell = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_hidden, state_cell]
    decoder_dense = trained_model.layers[4]
    decoder_outputs = decoder_dense(decoder_outputs)

    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    return encoder_model, decoder_model

# create a response based on an input from the user
def decode_response(test_input, model_info_dict, model_path):

    trained_model = load_model(model_path+'/training_model.h5')

    encoder_model, decoder_model = create_inference_model(trained_model)

    num_decoder_tokens = model_info_dict['num_respostas_tokens']
    respostas_dict = model_info_dict['respostas_dict']
    inverted_respostas_dict = model_info_dict['inverted_respostas_dict']
    max_decoder_seq_length = model_info_dict['max_decoder_seq_length']

    # getting the output states to pass into the decoder
    states_value = encoder_model.predict(test_input)
    # generating empty target sequence of length 1
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # setting the first token of target sequence with the start token
    target_seq[0, 0, respostas_dict['<START>']] = 1.

    decoded_sentence = ''
    
    stop_condition = False
    while not stop_condition:
        # predicting output tokens with probabilities and states
        output_tokens, hidden_state, cell_state = decoder_model.predict([target_seq] + states_value)
        # choosing the one with highest probability
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = inverted_respostas_dict[sampled_token_index]

        decoded_sentence += " " + sampled_token
        # stop if hit max length or found the stop token
        if (sampled_token == '<END>' or len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True
        # update the target sequence
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.
        # update states
        states_value = [hidden_state, cell_state]
    
    return decoded_sentence

def string_to_matrix(user_input, model_info_dict):
    max_encoder_seq_length = model_info_dict['max_encoder_seq_length']
    num_encoder_tokens = model_info_dict['num_perguntas_tokens']
    input_features_dict = model_info_dict['perguntas_dict']

    tokens = re.findall(r"[\w']+|[^\s\w]", user_input)
    user_input_matrix = np.zeros((1, max_encoder_seq_length, num_encoder_tokens),dtype='float32')
    for timestep, token in enumerate(tokens):
        if token in input_features_dict:
            user_input_matrix[0, timestep, input_features_dict[token]] = 1.

    return user_input_matrix

# method that will create a response using seq2seq model we built
def generate_response(user_input, model_path):
    
    with open(model_path+'model_info_dict.pkl', 'rb') as f:
        model_info_dict = pickle.load(f)

    df_temp = pd.DataFrame.from_dict({'col': [user_input]})
    df = preprocessing.preprocess_dataframe(df_temp)
    input_text = df['col'].iloc[0]

    input_matrix = string_to_matrix(input_text, model_info_dict)
    chatbot_response = decode_response(input_matrix, model_info_dict, model_path)
    # remove <START> and <END> tokens from chatbot_response
    chatbot_response = chatbot_response.replace("<START>",'')
    chatbot_response = chatbot_response.replace("<END>",'')
    return chatbot_response