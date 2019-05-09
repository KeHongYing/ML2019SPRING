import numpy as np
from gensim.models.word2vec import Word2Vec
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding, CuDNNGRU, Dense, BatchNormalization, regularizers, CuDNNLSTM, Dropout, GRU, Bidirectional, Conv1D
from keras.callbacks import ModelCheckpoint
import sys
import pickle

def add_new_model():
	model = Sequential()
	model.add(embedding_layer)
	model.add(Bidirectional(CuDNNGRU(128, return_sequences = True)))
	model.add(Bidirectional(CuDNNGRU(128, return_sequences = False)))
	model.add(Dense(2, kernel_regularizer = regularizers.l1_l2(l1 = 3e-3, l2 = 5e-3), activation='softmax'))
	model.summary()
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	return model

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

training_data = np.load(sys.argv[1])
word_model = Word2Vec.load(sys.argv[2])

label = np.load(sys.argv[3])

embedding_matrix = np.zeros((len(word_model.wv.vocab.items()) + 1, word_model.vector_size))
word2idx = {}

vocab_list = [(word, word_model.wv[word]) for word, _ in word_model.wv.vocab.items()]

for i, vocab in enumerate(vocab_list):
	word, vec = vocab
	embedding_matrix[i + 1] = vec
	word2idx[word] = i + 1

embedding_layer = Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1], weights=[embedding_matrix], trainable=False)

PADDING_LENGTH = 100
X = text_to_index(training_data)
X = pad_sequences(X, maxlen=PADDING_LENGTH)

Y = to_categorical(label)

model = add_new_model()

checkpoint = ModelCheckpoint("model_%s.h5"%(sys.argv[4]), monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
history = model.fit(x = X, y = Y, batch_size = 128, epochs = 5, validation_split = 0.1, callbacks = callbacks_list)

pickle.dump(history, open("best", "wb"))
model.save("model_backup.h5")

