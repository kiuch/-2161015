from gensim.models import word2vec
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sentences = word2vec.Text8Corpus('./wiki_wakati.txt')

model = word2vec.Word2Vec(sentences, vector_size=200, min_count=20, window=15)
model.wv.save_word2vec_format("./wiki.vec.pt", binary=True)
from gensim.models import KeyedVectors

wv = KeyedVectors.load_word2vec_format('./wiki.vec.pt', binary=True)
results = wv.most_similar(positive=['講義'])
for result in results:
    print(result)
