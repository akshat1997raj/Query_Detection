# %%
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from Functions import remove_punctation,apply_lemme,apply_stemming,remove_stop_words

# %%
df = pd.read_csv('train_3.csv')
df.drop(columns=['ID'], axis= 1, inplace= True)

# %%
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

encoder = LabelEncoder()
df['Domain'] = encoder.fit_transform(df['Domain'])
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(df['Title'])
x_train, x_test, y_train, y_test = train_test_split(x, df['Domain'], test_size= 0.2 , random_state= 50)
classifier = MultinomialNB()
classifier.fit(x_train, y_train)
predictions = classifier.predict(x_test)

accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)

accuracy
print(report)
encoder.classes_

# %%
# Now we will try to do Stemming to see the results

# %%
df1 = df.copy()
df1['Title'] = df1['Title'].apply(apply_stemming)
x = vectorizer.fit_transform(df1['Title'])
x_train, x_test, y_train, y_test = train_test_split(x, df1['Domain'], test_size=0.2, random_state= 50)
classifier.fit(x_train, y_train)
predictions = classifier.predict(x_test)
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)
accuracy
print(report)

# %% [markdown]
# ## Random Forest

# %%

from sklearn.ensemble import RandomForestClassifier
x = vectorizer.fit_transform(df1['Title'])
x_train,x_test,y_train,y_test = train_test_split(x,df1['Domain'], test_size= 0.2, random_state=50)
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=44)
rf_classifier.fit(x_train,y_train)
predictions = rf_classifier.predict(x_test)
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)
print(f"Accuracy: {accuracy}")
print(report)


# %% [markdown]
# ### Naive Bayes with lemmatization

# %%
# now after stemming, we will apply lemmatization on the words to see the results
df2 = df.copy()
df2['Title'] = df2['Title'].apply(apply_lemme)
x = vectorizer.fit_transform(df2['Title'])
x_train, x_test, y_train, y_test = train_test_split(x,df2['Domain'], test_size= 0.2, random_state=50)
classifier.fit(x_train,y_train)
predictions = classifier.predict(x_test)
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)
# accuracy
# print(report)
print(report)

# %%



