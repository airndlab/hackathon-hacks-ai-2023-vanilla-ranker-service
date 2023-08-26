import os
import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import NearestNeighbors


def preprocess_text(text):
    """Function to preprocess the given text."""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and punctuation
    text = re.sub(r'[^a-zA-Zа-яА-Я0-9\s]', '', text)
    return text


data_path = os.getenv('EXCEL_DATA_PATH')
if data_path is None:
    raise Exception('No EXCEL_DATA_PATH')

# Load the data into a DataFrame
faq_df = pd.read_excel(data_path)

# Display the first few rows of the dataset
# df.head()

# Apply preprocessing to the QUESTION column
faq_df['preprocessed_question'] = faq_df['QUESTION'].apply(preprocess_text)

# Display the preprocessed questions
# df[['QUESTION', 'preprocessed_question']].head()

# TF-IDF vectorizer - fit and transform, init and train
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(faq_df['preprocessed_question'])
tfidf_model = NearestNeighbors(n_neighbors=1, metric='cosine')
tfidf_model.fit(tfidf_matrix)

# Count vectorizer - fit and transform, init and train
count_vectorizer = CountVectorizer()
count_matrix = count_vectorizer.fit_transform(faq_df['preprocessed_question'])
count_model = NearestNeighbors(n_neighbors=1, metric='cosine')
count_model.fit(count_matrix)


# TODO: Add Word2Vec

def get_answer(question, vectorizer_type='tfidf'):
    """Function to get the answer for a given question based on the trained model."""
    # Preprocess the question
    preprocessed_question = preprocess_text(question)
    # Get index
    if vectorizer_type == 'count':
        index = get_index_by_count(preprocessed_question)
    else:
        # default - tfidf
        index = get_index_by_tfidf(preprocessed_question)
    # Return the corresponding answer
    return faq_df.iloc[index[0][0]]['ANSWER']


def get_index_by_tfidf(preprocessed_question):
    # Convert the question into its numerical representation
    question_vector = tfidf_vectorizer.transform([preprocessed_question])
    # Find the most similar question from the dataset
    _, index = tfidf_model.kneighbors(question_vector)
    return index


def get_index_by_count(preprocessed_question):
    # Convert the question into its numerical representation
    question_vector = count_vectorizer.transform([preprocessed_question])
    # Find the most similar question from the dataset
    _, index = count_model.kneighbors(question_vector)
    return index
