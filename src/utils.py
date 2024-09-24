import os 
import sys
import pickle
import numpy as np
from src.exception import CustomException
from src.logger import logging
import tensorflow
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        logging.info(f"An Exception occured while saving {obj}")
        raise CustomException(e, sys)
    
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise CustomException(e,sys)

def get_training_data(text,tokenizer_model_path):
    try:
        tokenizer = load_object(tokenizer_model_path)
        tokenizer.fit_on_texts([text])
        total_words = len(tokenizer.word_index)+1
        logging.info(f"The tokenizer has been loaded and text tokenzation has been completed and total number of tokenized words is:\n{total_words}\n")
        logging.info(f"Below is the dictionary of the indexed data\n{tokenizer.word_index}\n")

        input_sequences = []
        for line in text.split('\n'):
            token_list = tokenizer.texts_to_sequences([line])[0]
            for i in range(1,len(token_list)):
                n_gram_sequence = token_list[:i+1]
                input_sequences.append(n_gram_sequence)

        logging.info(f"Few example of input sequence:\n{input_sequences[10]}\n")
        max_sequence_len = max([len(x) for x in input_sequences])
        logging.info(f"The max sequence length is: {max_sequence_len}")
        input_sequences =  np.array(pad_sequences(input_sequences,maxlen=max_sequence_len))
        logging.info(f"After padding the input sequence:\n{input_sequences[10]}\n")
        X,y = input_sequences[:,:-1],input_sequences[:,-1]
        logging.info(f"The X dimension is: {X.shape} and y dimension is: {y.shape}")
        y = tensorflow.keras.utils.to_categorical(y,num_classes=total_words)
        logging.info(f"The example of true value after converting it to categorical form:\n{y[10]}")
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)
        logging.info(f"X_train shape: {X_train.shape}\nX_test shape: {X_test.shape}\nY_train shape: {y_train.shape}\ny_test shape: {y_test.shape}")
    
    except Exception as e:
        logging.info('Exception occured while getting the training data\n',e)
        raise CustomException(e,sys)
    
    return max_sequence_len,total_words,X,y,X_train,X_test,y_train,y_test