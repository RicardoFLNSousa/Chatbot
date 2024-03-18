import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
import os

nltk.download('stopwords')
nltk.download('punkt')


def preprocess_dataframe(df):

    # remove rows with nan values from dataframe
    df = df.dropna()
    # for all columns in the dataframe
    for col in df.columns:
        # turn all text into lowercase
        df[col] = df[col].apply(lambda x: x.lower())

        # remove special characters such as accents
        df[col] = df[col].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')

        # remove punctuation from the text
        df[col] = df[col].replace(to_replace=r'[^\w\s]', value='', regex=True)
        df[col] = df[col].replace(to_replace=r'[_]', value='', regex=True)

        # tokenize the sentences
        df[col] = df[col].apply(word_tokenize)

        # remove stop words from text
        stopwords = nltk.corpus.stopwords.words('portuguese')
        df[col] = df[col].apply(lambda x: [item for item in x if item not in stopwords])

        # join again the string into a list of strings
        df[col] = df[col].apply(lambda x: ' '.join(x))

    return df



df = pd.read_csv('datasets/dataset_translated.csv', index_col=0)
df = preprocess_dataframe(df)
df.to_csv(os.getcwd()+'/datasets/dataset_preprocessed.csv')

