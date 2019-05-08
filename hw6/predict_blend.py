from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from gensim.models.word2vec import Word2Vec
import numpy as np
import sys

def text_to_index(corpus):
    new_corpus = []
    for doc in corpus:
        new_doc = []
        for word in doc:
            try:
                new_doc.append(word2idx[word])
            except:
                new_doc.append(0)
        new_corpus.append(new_doc)
    return np.array(new_corpus)

model = []
for i in range(8, 22):
	model.append(load_model("model_%d.h5"%i))
	model[-1].compile(loss = "categorical_crossentropy", optimizer = "Adam", metrics = ["accuracy"])

x_test = np.load("test_data.npy")

word_model = Word2Vec.load("word_data_2")

embedding_matrix = np.zeros((len(word_model.wv.vocab.items()) + 1, word_model.vector_size))
word2idx = {}

vocab_list = [(word, word_model.wv[word]) for word, _ in word_model.wv.vocab.items()]

for i, vocab in enumerate(vocab_list):
	word, vec = vocab
	embedding_matrix[i + 1] = vec
	word2idx[word] = i + 1

PADDING_LENGTH = 100
X = text_to_index(x_test)
X = pad_sequences(X, maxlen=PADDING_LENGTH)

classes = 0

for i in model:
	classes += i.predict(X)

with open(sys.argv[1], "w", newline = "\n") as out:
	out.write("id,label\n")
	for i in range(len(classes)):
		out.write("%d,%d\n"%(i, np.argmax(classes[i])))
