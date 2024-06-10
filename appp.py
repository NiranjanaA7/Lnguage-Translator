from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.models import Model
from tensorflow.keras import models
from tensorflow.keras.layers import Input, LSTM, Dense



app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/translate', methods=['POST'])
def translate():
    lang = request.form['lang']
    text = request.form['text']


    if lang == 'tamil':
        translation = translate_to_tamil(text)
    elif lang == 'french':
        translation = translate_to_french(text)
    elif lang == 'spanish':
        translation = translate_to_spanish(text)
    else:
        return render_template('home.html', error='Invalid language selection.')

    return render_template('home.html', translation=translation)

def translate_to_tamil(text):
    datafile = pickle.load(open("training_data_tamil.pkl", "rb"))
    input_characters = datafile['input_characters']
    target_characters = datafile['target_characters']
    max_input_length = datafile['max_input_length']
    max_target_length = datafile['max_target_length']
    num_en_chars = datafile['num_en_chars']
    num_dec_chars = datafile['num_dec_chars']

    model = models.load_model("tamil.h5")
    enc_outputs, state_h_enc, state_c_enc = model.layers[2].output
    en_model = Model(model.input[0], [state_h_enc, state_c_enc])

    dec_state_input_h = Input(shape=(256,), name="input_3")
    dec_state_input_c = Input(shape=(256,), name="input_4")
    dec_states_inputs = [dec_state_input_h, dec_state_input_c]

    dec_lstm = model.layers[3]
    dec_outputs, state_h_dec, state_c_dec = dec_lstm(model.input[1], initial_state=dec_states_inputs)
    dec_states = [state_h_dec, state_c_dec]
    dec_dense = model.layers[4]
    dec_outputs = dec_dense(dec_outputs)
    dec_model = Model([model.input[1]] + dec_states_inputs, [dec_outputs] + dec_states)

    en_in_data = bagofcharacters(text.lower() + ".", input_characters, max_input_length)
    decoded_sentence = decode_sequence(en_in_data, en_model, dec_model, target_characters, max_target_length, num_dec_chars)

    return decoded_sentence

def translate_to_french(text):
    datafile = pickle.load(open("training_data_french.pkl", "rb"))
    input_characters = datafile['input_characters']
    target_characters = datafile['target_characters']
    max_input_length = datafile['max_input_length']
    max_target_length = datafile['max_target_length']
    num_en_chars = datafile['num_en_chars']
    num_dec_chars = datafile['num_dec_chars']

    model = models.load_model("french.h5")
    enc_outputs, state_h_enc, state_c_enc = model.layers[2].output
    en_model = Model(model.input[0], [state_h_enc, state_c_enc])

    dec_state_input_h = Input(shape=(256,), name="input_3")
    dec_state_input_c = Input(shape=(256,), name="input_4")
    dec_states_inputs = [dec_state_input_h, dec_state_input_c]

    dec_lstm = model.layers[3]
    dec_outputs, state_h_dec, state_c_dec = dec_lstm(model.input[1], initial_state=dec_states_inputs)
    dec_states = [state_h_dec, state_c_dec]
    dec_dense = model.layers[4]
    dec_outputs = dec_dense(dec_outputs)
    dec_model = Model([model.input[1]] + dec_states_inputs, [dec_outputs] + dec_states)

    en_in_data = bagofcharacters(text.lower() + ".", input_characters, max_input_length)
    decoded_sentence = decode_sequence(en_in_data, en_model, dec_model, target_characters, max_target_length, num_dec_chars)

    return decoded_sentence

def translate_to_spanish(text):
    datafile = pickle.load(open("training_data_spanish.pkl", "rb"))
    input_characters = datafile['input_characters']
    target_characters = datafile['target_characters']
    max_input_length = datafile['max_input_length']
    max_target_length = datafile['max_target_length']
    num_en_chars = datafile['num_en_chars']
    num_dec_chars = datafile['num_dec_chars']

    model = models.load_model("spanish.h5")
    enc_outputs, state_h_enc, state_c_enc = model.layers[2].output
    en_model = Model(model.input[0], [state_h_enc, state_c_enc])

    dec_state_input_h = Input(shape=(256,), name="input_3")
    dec_state_input_c = Input(shape=(256,), name="input_4")
    dec_states_inputs = [dec_state_input_h, dec_state_input_c]

    dec_lstm = model.layers[3]
    dec_outputs, state_h_dec, state_c_dec = dec_lstm(model.input[1], initial_state=dec_states_inputs)
    dec_states = [state_h_dec, state_c_dec]
    dec_dense = model.layers[4]
    dec_outputs = dec_dense(dec_outputs)
    dec_model = Model([model.input[1]] + dec_states_inputs, [dec_outputs] + dec_states)

    en_in_data = bagofcharacters(text.lower() + ".", input_characters, max_input_length)
    decoded_sentence = decode_sequence(en_in_data, en_model, dec_model, target_characters, max_target_length, num_dec_chars)

    return decoded_sentence
def bagofcharacters(input_t, input_characters, max_input_length):
    cv = CountVectorizer(binary=True, tokenizer=lambda txt: txt.split(), stop_words=None, analyzer='char')
    en_in_data = []
    pad_en = [1] + [0] * (len(input_characters) - 1)

    cv_inp = cv.fit(input_characters)
    en_in_data.append(cv_inp.transform(list(input_t)).toarray().tolist())

    if len(input_t) < max_input_length:
        for _ in range(max_input_length - len(input_t)):
            en_in_data[0].append(pad_en)

    return np.array(en_in_data, dtype="float32")

def decode_sequence(input_seq, en_model, dec_model, target_characters, max_target_length, num_dec_chars):
    reverse_target_char_index = dict(enumerate(target_characters))
    states_value = en_model.predict(input_seq)

    # Fit CountVectorizer with target characters
    co = CountVectorizer(binary=True, tokenizer=lambda txt: txt.split(), stop_words=None, analyzer='char')
    co.fit(target_characters)

    # Initialize target sequence with start character '\t'
    target_seq = np.array([co.transform(list("\t")).toarray().tolist()], dtype="float32")

    stop_condition = False
    decoded_sentence = ""
    while not stop_condition:
        output_chars, h, c = dec_model.predict([target_seq] + states_value)

        char_index = np.argmax(output_chars[0, -1, :])
        text_char = reverse_target_char_index[char_index]
        decoded_sentence += text_char

        if text_char == "\n" or len(decoded_sentence) > max_target_length:
            stop_condition = True

        target_seq = np.zeros((1, 1, num_dec_chars))
        target_seq[0, 0, char_index] = 1.0
        states_value = [h, c]

    return decoded_sentence

if __name__ == "__main__":
    app.run(debug=True)
