# NEWS CLASIFFICATION 🦅

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



The dataset used has been downloaded from [Kaggle](https://www.kaggle.com/datasets/gpiosenka/100-bird-species) and contains a set of 525 bird species. 84635 training images, 2625 test images(5 images per species) and 2625 validation images(5 images per species).

The app can be tested in **Hugging Face** (.py files hosted there) following this [link](https://huggingface.co/spaces/bmartinc80/birds_pytorch).

## 👨‍💻 **Tech Stack**


![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23d9ead3.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![Linux](https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black)
![Git](https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white)


## 🪶 Set Up

In the first stage, a set of helper functions was created in order to easily perform the modelling and prediction

- **Set seed**: Set random seeds for PyTorch operations, both on the CPU and the GPU
- **DataLoaders**: Create data loaders for training and testing datasets using PyTorch's DataLoader class
- **Writer**: SummaryWriter object for logging experiments and metrics in TensorBoard
- **Training and Testing**: Several functions for training and testing a PyTorch model 
- **Plots**: Several plots including loss curve, predictions and images

## 📳 Modelling

The first approach was to train 2 Pytorch EfficientNet models (EffNetB0, EffNetB2) with **5 and 10 epochs** using the pretrained model weights of EffNetB0 for the DataLoaders in order to stablish a baseline. The **EffNetB2 with 10 epochs** showed the best performance above **93%** on the test set.

<p align="center">
    <img src="images/accuracy.png" width="700" height="500"/>
</p>


<p align="center">
    <img src="images/birds_predictions.png"/>
</p>

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