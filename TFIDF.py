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

# %%
df1 = df.copy()

# %% [markdown]
# ### random forest

# %%
rf_classifier = RandomForestClassifier()
tf_vectorizer = TfidfVectorizer()
x = tf_vectorizer.fit_transform(df1['Title'])
x_train, x_test, y_train, y_test = train_test_split(x, df1['Domain'], test_size= 0.2, random_state=50)
rf_classifier.fit(x_train,y_train)
predictions = rf_classifier.predict(x_test)
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)
print(f"{accuracy}")
print(report)

# %% [markdown]
# ### naive bayes

# %%
classifier = MultinomialNB()
x = tf_vectorizer.fit_transform(df1['Title'])
x_train, x_test, y_train,y_test = train_test_split(x, df1['Domain'], test_size= 0.2, random_state= 50)
classifier.fit(x_train, y_train)
predictions = classifier.predict(x_test)
print(accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))

# %%



