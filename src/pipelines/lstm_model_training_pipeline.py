import os
import sys
import nltk
import tensorflow
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.models import load_model
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import get_training_data

@dataclass
class lstm_model_training_config:
    lstm_model_path = os.path.join(os.getcwd(),'models','lstm_model.h5')

class lstm_model_training:
    def __init__(self):
        self.lstm_model_training_config = lstm_model_training_config()
    
    def lstm_model_training(self,text,tokenizer_model_path):
        try:
            text = text
            tokenizer_model_path = tokenizer_model_path
            logging.info("Details of data before starting LSTM model training")
            max_sequence_len,total_words,X,y,X_train,X_test,y_train,y_test = get_training_data(text,tokenizer_model_path)
            
            logging.info("LSTM model training starts")
            model = Sequential()
            model.add(Embedding(total_words,100,input_length=max_sequence_len-1))
            model.add(LSTM(150,return_sequences=True))
            model.add(Dropout(0.2))
            model.add(LSTM(100))
            model.add(Dense(total_words,activation='softmax'))

            model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

            history = model.fit(
                        X,y,
                        epochs=3,
                        validation_split=0.25,
                             )
            
            model.save(self.lstm_model_training_config.lstm_model_path)
             
            lstm_model_val_loss = history.history['val_loss']
            lstm_model_avg_val_loss = np.mean(lstm_model_val_loss)

            accuracy = history.history['accuracy']
            accuracy = accuracy[-1]
            logging.info(f"LSTM model training completed with {accuracy*100}% of accuracy")
            
        except Exception as e:
            logging.info('Exception occured while LSTM model training\n',e)
            raise CustomException(e,sys)
        
        return self.lstm_model_training_config.lstm_model_path,lstm_model_avg_val_loss
    



    