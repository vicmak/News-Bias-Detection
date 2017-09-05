
import numpy
# Set random seed to produce repeatable results
numpy.random.seed(7)
import copy
from keras.models import Sequential
import random
from keras.layers import Dense
from keras.layers import LSTM, Bidirectional
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from collections import Counter, defaultdict
from itertools import count
import nltk
import mmap
import os
from sklearn import model_selection
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from keras.layers import Dropout
from string import ascii_letters, digits, whitespace, punctuation
from keras import backend as K


class Vocab:  # Storing the vocabulary and word-2-id mappings
    def __init__(self, w2i=None):
        if w2i is None: w2i = defaultdict(count(0).next)
        self.w2i = dict(w2i)
        self.i2w = {i:w for w,i in w2i.iteritems()}

    @classmethod
    def from_corpus(cls, corpus):
        w2i = defaultdict(count(0).next)
        for sent in corpus:
            [w2i[word] for word in sent]
        return Vocab(w2i)

    def size(self): return len(self.w2i.keys())


def ExtractAlphanumeric(ins):
    return "".join([ch for ch in ins if ch in (ascii_letters + whitespace + punctuation)])


def get_padded_sentences_tokens_list(text, mark=""):
    tokens = []
    sentences = nltk.sent_tokenize(text)
    for sent in sentences:
        sent_tokens = nltk.word_tokenize(sent)
        new_tokens = [token + mark for token in sent_tokens]
        tokens += ["<sentence_start>"] + new_tokens + ["<sentence_stop>"]

    return tokens


def log_train_file(message):
    log_file = "news_lp.txt"
    with open(log_file, "a") as myfile:
        myfile.write(message)


def read_embeddings(embeddings_filename):
    embeddings = dict()
    with file(embeddings_filename) as f:
        for line in f:
            tokens = line.split(" ")
            word = tokens[0].strip()
            emb = tokens[1:]
            float_emb = [float(x) for x in emb]
            embeddings[word] = float_emb
    return embeddings


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


def get_int_texts(sentences, vocab):
    int_sents = []
    s_counter = 0
    for sentence in sentences:
        print "Converting a sentence to i", s_counter
        s_counter += 1
        isent = [vocab.w2i[w] for w in sentence]
        int_sents.append(isent)
    return int_sents

class NewsCorpusReader:
    def __init__(self, positive_news_path, negative_news_path):
        self.positive_news_path = positive_news_path
        self.negative_news_path = negative_news_path
        self.positive_number = 0
        self.negative_number = 0
        self.vocab_filename = "/home/ise/victor/news_50k_vocab.txt"
        self.vocab = []

        with open(self.vocab_filename) as vocab_file:
            for line in vocab_file:
                line = line.strip()
                self.vocab.append(line)

    def get_token_list(self, filename, full_path):

        title = filename.lower()
        #title = ""
        #  Uncomment this for using the text of the articles
        #full_path = root_path + "/" + filename
        with open(full_path) as news_file:
            news_text = news_file.read().replace('\n', ' ')
            title = title + news_text.lower()

        title = ExtractAlphanumeric(title)
        title_tokens = get_padded_sentences_tokens_list(title)
        clean_tokens = [token for token in title_tokens if token in self.vocab]
        tokens_list = ["<start>"] + clean_tokens + ["<stop>"]
        return tokens_list

    def __iter__(self):  # Yields one instance as a list of words

        for root, dirs, files in os.walk(self.positive_news_path):
            path = root.split(os.sep)
            print((len(path) - 1) * '---', os.path.basename(root))
            for filename in files:
                tokens_list = self.get_token_list(filename, root + "/" + filename)
                self.positive_number += 1
                yield tokens_list

        for root, dirs, files in os.walk(self.negative_news_path):
            path = root.split(os.sep)
            print((len(path) - 1) * '---', os.path.basename(root))
            for filename in files:
                tokens_list = self.get_token_list(filename, root + "/" + filename)
                self.negative_number += 1
                yield tokens_list


print "Read Training folder..."
train = NewsCorpusReader("/home/ise/victor/news/israeli", "/home/ise/victor/news/palestinian")
print "Read Testing folder..."
test = NewsCorpusReader("/home/ise/victor/news/only agreement/israeli", "/home/ise/victor/news/only agreement/arab")
print "Creating train vocab..."
vocab = Vocab.from_corpus(train)

positive_train_num = train.positive_number
negative_train_num = train.negative_number

print "Positive number:", positive_train_num
print "Negative number:", negative_train_num

print "Vocabulary size:", vocab.size()

train_list = list(train)
test_list = list(test)

print "TEST SIZE:", len(test_list)

positive_test_num = test.positive_number
negative_test_num = test.negative_number

print "Positive test N", positive_test_num
print "Negative test N", negative_test_num

# Creating the train set - labels
Ys = []
for i in range(0, positive_train_num):
    Ys.append(1)
for i in range(0, negative_train_num):
    Ys.append(0)

# We need to shuffle the train with its labels accordingly
c = list(zip(train_list, Ys))
random.shuffle(c)
train_list, arr_Ys = zip(*c)  # The results are arrays

# Convert back into list, so we can use it in training
Ys = []
for item in arr_Ys:
    Ys.append(item)

test_Ys = []

for i in range(0, positive_test_num):
    test_Ys.append(1)
for i in range(0, negative_test_num):
    test_Ys.append(0)

# Shuffle the train instances and labels together
c = list(zip(test_list, test_Ys))
random.shuffle(c)
test_list, test_Ys = zip(*c)

int_train = []

for item in train_list:
    int_item = [vocab.w2i[w] for w in item]
    int_train.append(int_item)

int_test = []

for item in test_list:
    int_item = [vocab.w2i[w] for w in item if w in vocab.w2i.keys()]
    int_test.append(int_item)


max_text_length = 300
embedding_vector_length = 300
memory_size = 300  # The size of LSTM memory cell
WORDS_NUM = vocab.size()
train_batch_size = 200
embeddings_filename = "news300d.txt"
unlabeled_batch_size = 100

unlabeled_folder_name = "/home/ise/victor/news/neutral"
unlabeled_instances_number = 4000  # Total number of unlabeled instances
unlabeled_batches_number = unlabeled_instances_number / unlabeled_batch_size
indexes = list(range(unlabeled_batches_number))
random.shuffle(indexes)


print "Reading embeddings"
embs = read_word2vec_embeddings(embeddings_filename)
print "Computing embeddings"
embedding_weights = numpy.zeros((WORDS_NUM, embedding_vector_length))
for word,index in vocab.w2i.items():
    if word in embs.keys():
        embedding_weights[index,:] = embs[word]
    else:
        sampl = numpy.random.uniform(low=-1.0, high=1.0, size=(embedding_vector_length,))
        embs[word] = sampl
        embedding_weights[index, :] = sampl


X_train = int_train
Y_train = Ys

new_instances = []
new_Ys = []
original_instances = copy.deepcopy(int_train)
original_Ys = copy.deepcopy(Y_train)


X_test = int_test
Y_test = test_Ys
X_test = sequence.pad_sequences(X_test, maxlen=max_text_length)


def read_next_articles(folder_name, from_article, to_article):

    articles_paths = []
    for root, dirs, files in os.walk(folder_name):
        for filename in files:
            articles_paths.append([filename, root + "/" + filename])

    new_unlabeled_instances = []
    for i in range(from_article, to_article):
        new_unlabeled_instances.append(train.get_token_list(articles_paths[i][0],articles_paths[i][1]))

    return new_unlabeled_instances



for i in indexes:

    K.clear_session()

    unlabeled_train = read_next_articles(unlabeled_folder_name, i*unlabeled_batch_size, i*unlabeled_batch_size + unlabeled_batch_size)
    print "Creating unlabeled i-sentences for training, batch:", i
    int_unlabeled_train = get_int_texts(unlabeled_train, vocab)

    model = Sequential()
    model.add(Embedding(WORDS_NUM, embedding_vector_length, input_length=max_text_length, weights=[embedding_weights]))
    model.add(LSTM(memory_size))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    print "Padding train"
    padded_train = sequence.pad_sequences(X_train, maxlen=max_text_length)

    print "Fitting the model - Train start"
    model.fit(padded_train, Y_train, epochs=20, batch_size=train_batch_size)
    log_train_file("Train size: " + str(len(padded_train)) + "\n")

    padded_test_unlabeled = sequence.pad_sequences(int_unlabeled_train, maxlen=max_text_length)
    print "Starting prediction"
    predictions = model.predict(padded_test_unlabeled)

    # Here add the best predictions to the labeled instances
    num_positive_instances = 0
    num_negative_instances = 0
    new_instances_num = 0

    for i in range(0, len(predictions)):
        if predictions[i] > 0.9:
            X_train.append(int_unlabeled_train[i])
            new_instances.append(int_unlabeled_train[i])
            Y_train.append(1)
            new_Ys.append(1)
            new_instances_num += 1
            num_positive_instances += 1
        if predictions[i] < 0.1:
            X_train.append(int_unlabeled_train[i])
            new_instances.append(int_unlabeled_train[i])
            Y_train.append(0)
            new_Ys.append(0)
            new_instances_num += 1
            num_negative_instances += 1
    print "Added new instances:", new_instances_num
    log_train_file("Added new instances: " + str(new_instances_num) + "\n")

    model = Sequential()
    model.add(Embedding(WORDS_NUM, embedding_vector_length, input_length=max_text_length, weights=[embedding_weights]))
    model.add(LSTM(memory_size))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    padded_train = sequence.pad_sequences(X_train, maxlen=max_text_length)
    model.fit(padded_train, Y_train, epochs=20, batch_size=train_batch_size)
    predictions = model.predict(X_test)


    rounded = []
    for prediction_value in predictions:
        if prediction_value > 0.5:
            rounded.append(1)
        else:
            rounded.append(0)

    print "Test size:", len(rounded)
    print "Israeli Recall:", recall_score(Y_test, rounded, pos_label=1)
    print "Palestinian Recall:", recall_score(Y_test, rounded, pos_label=0)
    print "AUC:", roc_auc_score(Y_test, predictions)
    log_train_file("Israeli Recall:" + str(recall_score(Y_test, rounded, pos_label=1)) + "\n")
    log_train_file("Palestinian Recall:" + str(recall_score(Y_test, rounded, pos_label=0)) + "\n")
    log_train_file("AUC:" + str(roc_auc_score(Y_test, predictions)) + "\n")




def send_email(user='VictorNotificationMail@gmail.com', pwd='12345ABC', recipient='vitiokm@gmail.com', subject='finish expirement', body='finish well'):
    import smtplib

    gmail_user = user
    gmail_pwd = pwd
    FROM = user
    TO = recipient if type(recipient) is list else [recipient]
    SUBJECT = subject
    TEXT = body

    # Prepare actual message
    message = """From: %s\nTo: %s\nSubject: %s\n\n%s
    """ % (FROM, ", ".join(TO), SUBJECT, TEXT)
    try:
        # SMTP_SSL Example
        server_ssl = smtplib.SMTP_SSL("smtp.gmail.com", 465)
        server_ssl.ehlo()  # optional, called by login()
        server_ssl.login(gmail_user, gmail_pwd)
        # ssl server doesn't support or need tls, so don't call server_ssl.starttls()
        server_ssl.sendmail(FROM, TO, message)
        # server_ssl.quit()
        server_ssl.close()
        print 'successfully sent the mail'
    except:
        print "failed to send mail"

text_body = "Israeli Recall:" + str(recall_score(Y_test, rounded, pos_label=1)) + "\n" + "Palestinian Recall:" + str("Israeli:" + str(recall_score(Y_test, rounded, pos_label=0))) + "\n" + "AUC:" + str(roc_auc_score(Y_test, predictions))
text_body += "\n" + "LSTM size:" + str(memory_size) + "Batch size:" + str(train_batch_size) + "Max text length:" + str(max_text_length)

send_email(subject="news experiment finished", body=text_body)
