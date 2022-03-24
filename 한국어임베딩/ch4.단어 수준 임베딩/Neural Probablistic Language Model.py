corpus_fname = '/notebooks/embedding/data/tokenized/corpus_mecab.txt'
model_fname = '/notebooks/embedding/data/word-embeddings/word2vec/word2vec'

from gensim.models import Word2Vec

corpus = [sent.strip().split(' ') for sent in open(corpus_fname, 'r').readlines()]
# sg=1 -> skip-gram / sg=0 -> CBOW / size -> window_size 
model = Word2Vec(corpus, size=100, workers=4, sg=1, vector_size=5)
model.save(model_fname)

# vector_dim = 100 
from models.word_eval import WordEmbeddingEvaluator 
model = WordEmbeddingEvaluator('/notebooks/embedding/data/word-embeddings/word2vec/word2vec', method='word2vec', dim=100, tokenizer_name='mecab')
model.most_similar('희망', topn=5)

