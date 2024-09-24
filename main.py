import os
from src.logger import logging
from src.components.data_preprocessing_embedding import data_preprocessing_tokenization_embedding
from src.pipelines.lstm_model_training_pipeline import lstm_model_training
from src.pipelines.gru_model_training_pipeline import gru_model_training

data_preprocessing_tokenization_embedding_obj = data_preprocessing_tokenization_embedding()
data_preprocessing_tokenization_embedding_obj.device_selection()
text = data_preprocessing_tokenization_embedding_obj.read_data()
tokenizer_model_path = data_preprocessing_tokenization_embedding_obj.tokenization_embedding(text)


def choose_model(model_choice):
    if model_choice == "LSTM":
        lstm_model_training_obj = lstm_model_training()
        lstm_model_path, lstm_model_avg_val_loss = lstm_model_training_obj.lstm_model_training(text,tokenizer_model_path)
        logging.info(f"The selected model is LSTM for application")
        lstm_model_path = "lstm_model.h5"
        return lstm_model_path

    elif model_choice == "GRU":
        gru_model_training_obj = gru_model_training()
        gru_model_path, gru_model_avg_val_loss = gru_model_training_obj.gru_model_training(text,tokenizer_model_path)
        logging.info(f"The selected model is GRU for application")
        gru_model_path = "gru_model.h5"
        return gru_model_path

    elif model_choice == "best of both":
        lstm_model_training_obj = lstm_model_training()
        lstm_model_path, lstm_model_avg_val_loss = lstm_model_training_obj.lstm_model_training(text,tokenizer_model_path)

        gru_model_training_obj = gru_model_training()
        gru_model_path, gru_model_avg_val_loss = gru_model_training_obj.gru_model_training(text,tokenizer_model_path)

        if  lstm_model_avg_val_loss < gru_model_avg_val_loss:
            logging.info(f"The model selected for application is LSTM model as it has minimum validation loss")
            lstm_model_path = "lstm_model.h5"
            return lstm_model_path
        else:
            logging.info(f"The model selected for application is GRU model as it has minimum validation loss")
            gru_model_path = "gru_model.h5"
            return gru_model_path

model_choice = "GRU"
model_path = choose_model(model_choice)
model_path_detail_text_folder_path = os.path.join(os.getcwd(),'artifacts')
model_path_filename = 'model_path.txt'
model_path_detail_text_file_path = os.path.join(model_path_detail_text_folder_path,model_path_filename)
tokenizer_model_path = "tokenizer.pkl"
with open(model_path_detail_text_file_path, 'w') as file:
     file.write(tokenizer_model_path)
     file.write("==")
     file.write(model_path)



