# NEWS CLASIFFICATION 🗞️

<p align="center">
    <img src="https://cdn.britannica.com/25/93825-050-D1300547/collection-newspapers.jpg" width="500" height="400"/>
</p>

This repository hosts a notebook featuring an in-depth analysis of several **RNN** models, together with **CNN** and **Multinomial Naive Bayes** along with an app deployment using Streamlit. The following models were meticulously evaluated:

- Basic Keras Model
- LSTM Model
- LSTM GRU Model
- LSTM Bidirectional Model
    - TextVectorization + Keras Embedding
    - Text_to_word_sequence	+ Word2Vec Embedding
- Basic CNN Model



The dataset used has been downloaded from [Kaggle](https://www.kaggle.com/datasets/gpiosenka/100-bird-species) and contains a set of Fake and Real News.

The app can be tested following this [link](https://huggingface.co/spaces/bmartinc80/birds_pytorch).

## 👨‍💻 **Tech Stack**


![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23d9ead3.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Linux](https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black)
![Git](https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)

## 📐 Set Up

In the first stage, a set of helper functions was created in order to easily visualize the data analysis and modelling results

- **Plot WordCLoud**: Generate a word cloud for a specific label value and display it in a subplot
- **Plot Confusion Matrix**: Plot a confusion matrix to visualize classification results
- **Plot Precision/Recall Results**: Calculates model accuracy, precision, recall, and F1-score of a binary classification model and returns the results as a DataFrame

## 👨‍🔬 Data Analysis

The first approach was to analyze the dataset columns and ist distribution. The dataset contains the following columns:

- Title
- Text

<p align="center">
    <img src="images/dataset.png" width="700" height="500"/>
</p>

The labels are pretty well balance as they are close to 50% each.


<p align="center">
    <img src="images/raw_lablels_distribution.png" width="700" height="500"/>
</p>

On the other hands, the subjects contains 8 topics, from which the 2 most popular are all true news and the other 6 fake. This means that there is no mix of labels within subjects
</p>
<p align="center">
    <img src="images/subjects distribution.png" width="700" height="500"/>
</p>

<p align="center">
    <img src="images/subjects vs labels distribution.png" width="700" height="500"/>

Within the wordcloud, Trump and US are along the most common words in both labels

<p align="center">
    <img src="images/wordcloud.png"/>
</p>

## 👨‍🔬 Preprocessing

Along with the data analysis, the following data preprocessing steps where taken in order to create a clean dataset for the further modelling step:

- Removal of duplicated rows
- Removal of rows with empty cells
- Merging of text and title column

## 👨‍🔬 Modelling

The first approach was to train 2 Pytorch EfficientNet models (EffNetB0, EffNetB2) with **5 and 10 epochs** using the pretrained model weights of EffNetB0 for the DataLoaders in order to stablish a baseline. The **EffNetB2 with 10 epochs** showed the best performance above **93%** on the test set.



## ↗️ Model Improvement

Then the EffNetB2 with 10 epochs was trained again but this time using the pretrained model weights of EffNetB2 for the DataLoaders. This time an accuracy above **95%** on the **test set** and above **93%** on the **validation set** was achieved .

<p align="center">
    <img src="images/effnetB2.png"/>
</p>

## 👏 App Deployment

The last step was to deploy and app hosted in Hugging Face using Gradio. This app can be tested with available sample images or with own ones.

<p align="center">
    <img src="images/app_deployment.png"/>
</p>