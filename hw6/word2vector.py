import numpy as np
import jieba
from gensim.models.word2vec import Word2Vec
import sys

data = np.load(sys.argv[1])
test_data = np.load(sys.argv[2])
data_list = [i for i in data] + [i for i in test_data]

model = Word2Vec(data_list, iter = 10, size = 250, window = 10)

#for i in model.most_similar(positive = ["笨"], topn = 20):
#	print(i[0], i[1])

#vocab = model.wv.vocab
#word_vec = {}
#for w in vocab:
#	word_vec[w] = model[w]
#
#print(word_vec)

model.save("word_data_2")
