from flask import Flask, jsonify, request
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd
import re
import string
from string import digits

app = Flask(__name__)

# Load the saved model
model = tf.keras.models.load_model('./models/model_diuso.h5')

encoder_model = tf.keras.models.load_model('./models/encoder_model.h5')
decoder_model = tf.keras.models.load_model('./models/decoder_model.h5')



# Load the data
train_data = pd.read_table('./data/train.csv', sep=',', names=['question','sql'])
val_data = pd.read_table('./data/validation.csv', sep=',', names=['question','sql'])
test_data = pd.read_table('./data/test.csv', sep=',', names=['question','sql'])

print("---------------------------------------------------------------------------------------------")


# Preprocess the data
def preprocess(text):
    text = text.lower()
    text = re.sub("'", '', text)
    text = re.sub(",", " COMMA", text)
    exclude = set(string.punctuation)
    text = ''.join(ch for ch in text if ch not in exclude)
    remove_digits = str.maketrans('', '', digits)
    text = text.translate(remove_digits)
    return text

train_data.question = train_data.question.apply(preprocess)
train_data.sql = train_data.sql.apply(preprocess)

val_data.question = val_data.question.apply(preprocess)
val_data.sql = val_data.sql.apply(preprocess)

test_data.question = test_data.question.apply(preprocess)

# Add start and end tokens to SQL sentences
train_data.sql = train_data.sql.apply(lambda x: 'start ' + x + ' end')
val_data.sql = val_data.sql.apply(lambda x: 'start ' + x + ' end')



# Load the tokenizers
tokenizer = Tokenizer()
max_input_length=42 
max_output_length=58


sql_tokenizer = Tokenizer()
sql_tokenizer.fit_on_texts(train_data.sql)



@app.route('/')
def hello():
    return "Hello, Flask!"

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    # data = request.json
    # question = data['question']
    question = ''
    # Tokenize and pad the input sequence
    input_sequence = tokenizer.texts_to_sequences([question])
    input_sequence = pad_sequences(input_sequence, maxlen=max_input_length, padding='post')

    # Generate prediction
    prediction = predict_sequence(input_sequence)

    return jsonify({'prediction': prediction})

def predict_sequence(input_sequence):
    states_value = encoder_model.predict(input_sequence)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = sql_tokenizer.word_index['start']
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        print("#########################")
        print(sampled_token_index)
        print("##########################")
        sampled_token = sql_tokenizer.index_word[sampled_token_index]
        if sampled_token != 'end' and sampled_token != 'start':
            decoded_sentence += ' ' + sampled_token
        if sampled_token == 'end' or len(decoded_sentence.split()) >= (max_output_length-1):
            stop_condition = True
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]
    return decoded_sentence.strip()

if __name__ == '__main__':
    app.run(debug=True)

