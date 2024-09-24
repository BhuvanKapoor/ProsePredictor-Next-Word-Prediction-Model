# <div align="center">ProsePredict: Advanced LSTM & GRU Text Forecasting</div>
<div align="center">
  <img src="readme_data/ProsePredictor.jpeg" alt="Designer" width="500"/>
</div>

## Table of content
--------------
1. [Introduction](#introduction)
2. [Features](#features)
4. [Installation and Customization](#installation-and-customization)
4. [Tools and  techniques used](#tools-and-techniques-used)
5. [Contribution](#contribution)

## Introduction
---------------
ProsePredictor is a customizable next word prediction model that uses LSTM and GRU neural networks, along with word embedding techniques, to predict the next word in a sequence based on the previous words. This project is designed to be flexible and easy to train on your own data for personal use.

## Features
-----------
- **Customizable**: Train the model on your own dataset.
- **Multiple Models**: Utilizes both LSTM and GRU architectures.
- **Embedding Techniques**: Employs word embedding for improved accuracy.
- **User-Friendly**: Clear instructions and a simple workflow.
- **Interactive**: Streamlit application for easy interaction with the model.

## Installation and Customization
---------------------------------
1. Clone the repository:
    ```bash
    git clone https://github.com/BhuvanKapoor/ProsePredict
    ```
2. Navigate to the project directory:
    ```bash
    cd ProsePredict
    ```
3. Install the required packages
    ```bash
    pip install -r requirements.txt
    ```
4. In the data folder upload your own data.
* Make sure your data is in .txt format for sample you can refer base_data.txt present in data folder.
* Make sure that data folder has only one .txt file that is your data just delete the base_data.txt if you are adding your own data in it.

5. Clean the folders
* Delet every thing present in artifacts folder
* Delet all the models in model folder if your training yours to prevent any error.

6. Go to main.py file and select your prefferences
* Change the model_choice as per your requirement.
* There are three option 'LSTM', 'GRU', 'best of both'
* if you choose best of both both LSTM and GRU will be trained and best of them with less validation loss will be selected automatically.

7. Train the model
    ```bash
    python run main.py
    ```
* While training you can reffer log file generated in log folder with current time stamp and date name to see whats happening while training and also to debug.

8. run streamlit application
    ```bash
    streamlit run application.py
    ```
## Tools and  techniques used
-----------------------------
* LSTM
* GRU
* Word Embedding
* Tensorflow
* Scikit-learn
* Natural Language Processing
* Python
* Modular Coding
* Neural Networks

## Contribution
---------------
* Feel free to contribute to this project by submitting issues or pull requests. Any feedback or suggestions are welcome!

