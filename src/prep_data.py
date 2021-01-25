import os
from types import new_class
from bs4 import BeautifulSoup
from nltk import tokenize
from nltk.corpus import stopwords
import pickle
import numpy as np

stop_words = set(stopwords.words("english"))



def prepare_vocabulary():
    """
    The dir structure of the data dir is known. Therefore, this function uses hardcoded
    file paths to iterate over the files in corresponding dir to read data.
    The function calls preprocess_comments to clean the text.
    The function calls build_vocab to add new words or increase the count of existing 
    words to the vocabulary.
    """

    vocab = {}
    word_count = {}
    data_path = os.scandir("../data/train/pos")
    
    for entry in data_path:
        if entry.is_file():
            with open(entry) as f:
                comment = f.read()

                tokens = preprocess_comment(comment)

                count_words(tokens, word_count)


    data_path = os.scandir("../data/train/neg")
    
    for entry in data_path:
        if entry.is_file():
            with open(entry) as f:
                comment = f.read()

                tokens = preprocess_comment(comment)

                count_words(tokens, word_count)
        

    sorted_word_count = sorted(word_count.items(), key=lambda x:x[1], reverse=True)

    frequent_words = [w for (w,c) in sorted_word_count[:2000]]   

    word2id = {w: (index+1) for (index, w) in enumerate(frequent_words)}
    id2word = {index: w for (w, index) in enumerate(frequent_words)}

    vocab = {'word2id': word2id, 'id2word': id2word}

    # print(list(word2id.items())[:10])
    # print(list(id2word.items())[:10])
    pickle.dump(vocab, open("../serialized/vocab.pkl", "wb"))

    
def preprocess_comment(comment):
    """
    Takes a movie comment in its original form. 
    Remove html tags, <br />, in comment body and return the plain text comments. 
    Convert to lower case and tokenize to words. Single space was the tokenizer.
    Removes stopwords.
    Function returns a new string.
    """

    soup = BeautifulSoup(comment, 'lxml')
    text_only = soup.get_text()

    tokens = [w for w in text_only.strip().lower().split() if w not in stop_words]
     
    new_string = ""
    for w in tokens:
        new_string += w + " "
    return new_string
    

def count_words(tokens, vocab):
    """
    Takes a list of token and update the vocabulary dictionary.
    """
    for w in tokens.split():
        if w in vocab.keys():
            vocab[w] += 1
        else:
            vocab[w] = 1


def prepare_data():
    """
    Read training and testing data dirs. Vectorize sentences based on the vocabulary.
    """

    vocab = pickle.load(open("../serialized/vocab.pkl", "rb"))

    # print(vocab["word2id"]["like"])
    # print(vocab["id2word"]["like"])

    X_test = np.zeros((2000, 25000))

    data_path = os.scandir("../data/test/pos/")
    
    i = 0
    for entry in data_path:
        if entry.is_file():
            with open(entry) as f:
                comment = f.read()

                tokens = preprocess_comment(comment)

                unique_words = list(set(tokens.split()))

                for word in unique_words:
                    if word in vocab["id2word"].keys():
                        X_test[vocab["id2word"][word], i] = 1
                i += 1

    data_path = os.scandir("../data/test/neg/")

    for entry in data_path:
        if entry.is_file():
            with open(entry) as f:
                comment = f.read()

                tokens = preprocess_comment(comment)

                unique_words = list(set(tokens.split()))

                for word in unique_words:
                    if word in vocab["id2word"].keys():
                        X_test[vocab["id2word"][word], i] = 1
                i += 1
    y_test = [1] * 12500 + [0] * 12500
    y_test = np.array(y_test)
    y_test.reshape(1, 25000)

    # print(X_test.shape)
    # print(y_test.shape)

    X_train = np.zeros((2000, 25000))

    data_path = os.scandir("../data/train/pos/")
    
    i = 0
    for entry in data_path:
        if entry.is_file():
            with open(entry) as f:
                comment = f.read()

                tokens = preprocess_comment(comment)

                unique_words = list(set(tokens.split()))

                for word in unique_words:
                    if word in vocab["id2word"].keys():
                        X_train[vocab["id2word"][word], i] = 1
                i += 1

    data_path = os.scandir("../data/train/neg/")

    for entry in data_path:
        if entry.is_file():
            with open(entry) as f:
                comment = f.read()

                tokens = preprocess_comment(comment)

                unique_words = list(set(tokens.split()))

                for word in unique_words:
                    if word in vocab["id2word"].keys():
                        X_train[vocab["id2word"][word], i] = 1
                i += 1

    X_val = np.concatenate((X_train[:, 10000:12500], X_train[:, 22500:25000]), axis=1)
    X_train = np.concatenate((X_train[:, :10000], X_train[:, 12500:22500]), axis=1) 

    y_train = [1] * 10000 + [0] * 10000
    y_train = np.array(y_train)
    y_train.reshape(1, 20000)

    y_val = [1] * 2500 + [0] * 2500
    y_val = np.array(y_val)
    y_val.reshape(1,5000)

    np.random.seed(314)
    np.random.shuffle(X_train)
    np.random.seed(314)
    np.random.shuffle(y_train)

    # print(X_val.shape)
    # print(X_train.shape)

    pickle.dump((X_train, y_train, X_val, y_val, X_test, y_test), open("../serialized/data.pkl", "wb"))

