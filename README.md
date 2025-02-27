**Fake News Detection**
This project focuses on building a machine learning model to classify news articles as real or fake. The model uses various techniques to process and clean the data, and then applies a logistic regression classifier to predict the authenticity of news articles.
**Project Overview**
The goal of this project is to detect fake news using machine learning. The dataset consists of news articles with their corresponding titles, authors, content, and labels (1 for fake news and 0 for real news).
**Dataset**
The dataset contains the following columns:
id: A unique identifier for each news article.
title: The title of the news article.
author: The author of the news article.
text: The content of the article (may be incomplete).
label: A binary label indicating whether the news article is fake (1) or real (0).
**Libraries Used
The following libraries are used in this project:**
NumPy: For numerical computations.
Pandas: For handling and processing the dataset.
re: For regular expressions and text processing.
nltk: For text preprocessing such as stopwords removal and stemming.
scikit-learn: For machine learning algorithms and vectorization.
**Install the necessary dependencies using the following:**
pip install numpy pandas scikit-learn nltk
**NLTK Setup**
Before running the code, you will need to download the stopwords dataset from NLTK. This is done automatically by running:
import nltk
nltk.download('stopwords')
**Data Preprocessing**
Load the Data: The dataset is loaded into a pandas DataFrame.
Handle Missing Data: Any missing values are filled with an empty string.
Merge Title and Author: The 'author' and 'title' columns are combined to create a new 'content' column, which will be used as input to the model.
Text Cleaning: Stopwords are removed, and stemming is applied to reduce words to their root form.
**Example of cleaned content:**
"darrel lucu hous dem aid even see comey letter jason chaffetz tweet"
"daniel j flynn flynn hillari clinton big woman campu breitbart"
**Model Building**
The model uses TfidfVectorizer from scikit-learn to convert the text data into numerical features that can be used by the machine learning model.
**TfidfVectorizer**: Converts the text into a matrix of TF-IDF features.
Logistic Regression: A machine learning model that is trained on the TF-IDF features to classify whether the news is real or fake.
**Training and Evaluation:**
The data is split into training and test sets. The logistic regression model is trained on the training set and evaluated on the test set. The model achieves an accuracy of 89.5% on the test data.
**Training Accuracy:**
Accuracy score of the training data :  0.9379443717649736
**Test Accuracy:**
Accuracy score of the test data :  0.8951559389515594
**Making Predictions**
**Once the model is trained, you can use it to make predictions on new articles. For example:**
X_new = X_test[1]
prediction = model.predict(X_new)
If the prediction is 1, the news is Fake. If it is 0, the news is Real.
**Example output:**
The news is Fake
