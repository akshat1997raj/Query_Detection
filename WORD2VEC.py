# %%
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from Functions import remove_punctation,apply_lemme,apply_stemming,remove_stop_words
from sklearn.feature_extraction.text import TfidfVectorizer

# %%
df = pd.read_csv('train_3.csv')
df.drop(columns=['ID'], axis= 1, inplace= True)
df['Title'] = df['Title'].str.lower()
df['Title'] = df['Title'].astype(str)
df['Title'] = df['Title'].apply(remove_punctation)
df['Title'] = df['Title'].apply(remove_stop_words)

# %%
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from nltk.tokenize import word_tokenize

# %%
from gensim.models import Word2Vec

# %%
df1 = df.copy()

# %% [markdown]
# ### random forest

# %%
tokenized_corpus = [word_tokenize(sentence) for sentence in df1['Title']]
Word2Vec_model = Word2Vec(sentences=tokenized_corpus, min_count=1)

# %%
word2vec_embeddings = np.array([np.mean([Word2Vec_model.wv[token] for token in tokens if token in Word2Vec_model.wv.index_to_key], axis=0) for tokens in tokenized_corpus])

# %%
x = word2vec_embeddings
y = df1['Domain']

# %%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# %%
rf_classifier = RandomForestClassifier(n_estimators= 100, random_state=42)

# %%
rf_classifier.fit(x_train,y_train)

# %%
y_pred = rf_classifier.predict(x_test)

# %%
accuracy = accuracy_score(y_test,y_pred)

# %%
accuracy

# %%



