# %%
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

# %%
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
nltk.download('punkt')
nltk.download('wordnet')

# %%
def remove_punctation(text):
    return re.sub(r'[^\w\s]','', text)

# %%
def remove_stop_words(text):
    words = word_tokenize(text)
    removed_words = [word for word in words if word not in stop_words]
    removed_stop_words_text = ' '.join(removed_words)
    return removed_stop_words_text

# %%
porter_stemmer = PorterStemmer()
def apply_stemming(text):
    words = word_tokenize(text)
    stemmed_words = [porter_stemmer.stem(word) for word in words]
    stemmed_text = ' '.join(stemmed_words)
    return stemmed_text

# %%
lemetizer = WordNetLemmatizer()
def apply_lemme(text):
    words = word_tokenize(text)
    lemmetized_words = [lemetizer.lemmatize(word) for word in words]
    lemmatized_text = ' '.join(lemmetized_words)
    return lemmatized_text

# %%



