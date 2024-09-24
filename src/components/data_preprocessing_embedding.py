import os 
import sys
import pandas as pd
import numpy as np
import nltk
import tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer
import warnings
warnings.filterwarnings('ignore')
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class data_preprocessing_tokenization_embedding_config:
    data_tokenizer_path = os.path.join(os.getcwd(),'models','tokenizer.pkl')

class data_preprocessing_tokenization_embedding:
    def __init__(self):
       self.data_preprocessing_tokenization_embedding_config = data_preprocessing_tokenization_embedding_config()
      
    def device_selection(self):
        try:
           if tensorflow.config.list_physical_devices('GPU'):
              logging.info("GPU is available and will be used.")
           else:
              logging.info("GPU is not available. Using CPU instead.")

        except Exception as e:
           logging.info("Exception occured while selecting device:\n",e)
           raise CustomException(e,sys)
        
    def read_data(self):
        try:
           data_folder_path = os.path.join(os.getcwd(),'data')
           all_files = os.listdir(data_folder_path)
           txt_files = [file for file in all_files if file.endswith('.txt')]

           if len(txt_files) == 1:
              text_file =  txt_files[0]
              logging.info(f"We will train the model on data in {text_file}")
           elif len(txt_files) > 1:
              logging.info("Please make sure that the data folder has only one .txt file.")
           else:
              logging.info("No .txt files found in the folder.")

           text_data_file_path = os.path.join(data_folder_path,text_file)
           logging.info(f"The data file is located at:\n{text_data_file_path}")

           with open(text_data_file_path,'r',encoding='utf-8') as file:
              text = file.read().lower()
           logging.info(f"Below is the textual data in the file you provided:\n\n{text}")

        except Exception as e:
           logging.info("Exception occured while reading the data:\n",e)
           raise CustomException(e,sys)
        
        return text
    
    def tokenization_embedding(self,text):
        try:
           tokenizer = Tokenizer()
           tokenizer.fit_on_texts([text])
           logging.info(f"The text is tokenized and tokenizer is created")
           path = self.data_preprocessing_tokenization_embedding_config.data_tokenizer_path
           save_object(path,tokenizer)
           logging.info(f"The tokenizer model is saved at:\n{path}")

        except Exception as e:
           logging.info("Exception occured while tokenization and embedding\n",e)
           raise CustomException(e,sys)
        
        return path

    