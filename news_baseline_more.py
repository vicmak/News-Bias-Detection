"""Using [source, article_name, article_content]"""

import os
import xlwt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier, Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from stop_words import get_stop_words
from sklearn.calibration import CalibratedClassifierCV


STOP_WORDS = get_stop_words("english")

RANDOM_SEED = 20

IS = 1
PA = 0


SOURCES_DIR = '/Users/macbook/Dropbox/EventRegistry/ArticlesCopy/'

SOURCES = [
    ('/Users/macbook/Dropbox/EventRegistry/ArticlesCopy/www.jpost.com', IS),
    ('/Users/macbook/Dropbox/EventRegistry/ArticlesCopy/Times of Israel', IS),
    ('/Users/macbook/Dropbox/EventRegistry/ArticlesCopy/www.israelnationalnews.com', IS),
    ('/Users/macbook/Dropbox/EventRegistry/ArticlesCopy/hamodia', IS),
    ('/Users/macbook/Dropbox/EventRegistry/ArticlesCopy/www.aljazeera.com', PA),
    ('/Users/macbook/Dropbox/EventRegistry/ArticlesCopy/wafa', PA),
    ('/Users/macbook/Dropbox/EventRegistry/ArticlesCopy/www.palestinechronicle.com', PA),
    ('/Users/macbook/Dropbox/EventRegistry/ArticlesCopy/www.arabnews.com', PA),
    ('/Users/macbook/Dropbox/EventRegistry/ArticlesCopy/www.albawaba.com', PA),
    ('/Users/macbook/Dropbox/EventRegistry/ArticlesCopy/english.palinfo.com', PA)
]

EXCEL_FILE = '/Users/macbook/Desktop/corpora/news/annotated.xlsx'


def ExtractAlphanumeric(ins):
    ins = ins.lower()
    ins = ins.replace("-","")
    from string import ascii_letters, digits, whitespace, punctuation
    return "".join([ch for ch in ins if ch in (ascii_letters + whitespace + punctuation)])


def read_files(path):

    for root, dirs, files in os.walk(path):

        for filename in files:
            full_file_path = root + "/" + filename
            with open(full_file_path) as f:
                content = f.read()


            content = content.lower()
            content = ExtractAlphanumeric(content)

            title = ExtractAlphanumeric(filename.lower())

            #yield full_file_path, title
            #yield full_file_path, content
            yield full_file_path, title + " " + content



def build_data_frame(path, classification):
    rows = []
    index = []
    for file_path, text in read_files(path):
        rows.append({'text': text, 'class': classification})
        index.append(os.path.basename(file_path))

    data_frame = pd.DataFrame(rows, index=index)
    return data_frame

def build_data_frame_for_csv(path, classification):
    rows = []
    index = []
    i=0
    for file_path, text in read_files(path):
        rows.append({'text': text, 'label': classification})
        index.append(i)
        i = i+1

    data_frame = pd.DataFrame(rows, index=index)
    return data_frame


def load_data():
    data = pd.DataFrame({'text': [], 'class': []})
    #csv_data = pd.DataFrame({ 'text': [], 'label': []})
    for path, classification in SOURCES:
        data = data.append(build_data_frame(path, classification))
        #csv_data = csv_data.append(build_data_frame_for_csv(path, classification))
    return data


def load_from_excel_title_and_content(excel_file):
    data = pd.DataFrame({'article_name': [], 'text': []})
    df = pd.read_excel(excel_file)

    rows = []
    index = []

    for i in xrange(len(df)):
        with open(os.path.join(SOURCES_DIR, df.Column1[i], df.Column2[i])) as f:
            rows.append({'article_name': df.Column2[i],
                         'text': ExtractAlphanumeric(df.Column2[i] + f.read())})
        index.append(df.Column2[i])

    return pd.DataFrame(rows, index=index)


def load_from_excel_title(excel_file):
    data = pd.DataFrame({'article_name': [], 'text': []})
    df = pd.read_excel(excel_file)

    rows = []
    index = []

    for i in xrange(len(df)):
        with open(os.path.join(SOURCES_DIR, df.Column1[i], df.Column2[i])) as f:
            rows.append({'article_name': df.Column2[i],
                         'text': ExtractAlphanumeric(df.Column2[i])})
        index.append(df.Column2[i])

    return pd.DataFrame(rows, index=index)


def load_from_excel(excel_file):
    data = pd.DataFrame({'article_name': [], 'text': []})
    df = pd.read_excel(excel_file)

    rows = []
    index = []

    for i in xrange(len(df)):
        with open(os.path.join(SOURCES_DIR, df.Column1[i], df.Column2[i])) as f:
            rows.append({'article_name': df.Column2[i],
                         'text': f.read()})
        index.append(df.Column2[i])

    return pd.DataFrame(rows, index=index)


class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.dim = len(word2vec.itervalues().next())

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])

def read_word2vec_embeddings(word_embeddings_filename):
    embeddings = dict()
    counter = 0
    with file(word_embeddings_filename) as f:
        for line in f:
            if counter > 0:
                tokens = line.split(" ")
                word = tokens[0]
                emb = tokens[1:]
                float_emb = [float(x) for x in emb]
                embeddings[word] = float_emb
            counter += 1
    return embeddings


def main():

    embeddings_filename = "/Users/macbook/Desktop/corpora/embeddings/glove.6B.300d.txt"

    with open(embeddings_filename, "rb") as lines:
        word2vec = {line.split()[0]: np.array(map(float, line.split()[1:]))
                    for line in lines}

    #embeddings_filename = "/Users/macbook/Desktop/corpora/embeddings/news300d.txt"
    #word2vec = read_word2vec_embeddings(embeddings_filename)

    np.random.seed(RANDOM_SEED)

    data = load_data()
    #data.to_csv("/Users/macbook/Desktop/corpora/embeddings/news.csv")

    print data

 #   data = data.reindex(np.random.permutation(data.index))

    pipeline = Pipeline([
        ("glove vectorizer", MeanEmbeddingVectorizer(word2vec)),
        #('vectorizer', CountVectorizer()),
        # ('classifier',  SVC(C=1.0,  kernel='linear', max_iter=-1, probability=True)),
       # ('classifier', CalibratedClassifierCV(LinearSVC(), method='sigmoid', cv=3))
        # ('classifier',  MultinomialNB()),
        ('classifier',  LogisticRegression(penalty='l1', C=1))
        #('classifier', Lasso())
        # ('classifier',  RandomForestClassifier())
        # ('classifier',  DecisionTreeClassifier())
    ])

    signal = 'text'

    pipeline.fit(data[signal].values, data['class'].values)

    print "Loading from excel"
    unlabeled_data = load_from_excel_title_and_content(EXCEL_FILE)

    predictions = pipeline.predict(unlabeled_data[signal].values)
    predictions_proba = np.array([i[1] for i in pipeline.predict_proba(unlabeled_data[signal].values)])
    df = pd.read_excel(EXCEL_FILE)

    rounded = []
    for prediction_value in predictions:
        if prediction_value > 0.5:
            rounded.append(1)
        else:
            rounded.append(0)

    print "Printing "
    print df.daddy

    rec_is = recall_score(df.daddy, rounded, pos_label=IS)
    rec_pa = recall_score(df.daddy, rounded, pos_label=PA)

    print(rec_is)
    print(rec_pa)

    roc = roc_auc_score(df.daddy, predictions_proba)

    print(roc)

if __name__ == "__main__":
    main()
