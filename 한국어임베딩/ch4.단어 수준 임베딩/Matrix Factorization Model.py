# Latent Semantic Analysis 
# concat dataset 
# cat data/tokenized/ratings_mecab.txt data/tokenized/korquad_mecab.txt > data/tokenized/for-lsa-mecab.txt 

# use word-context matrix 
from sklearn.decomposition import TruncatedSVD
from soynlp.vectorizer import sent_to_word_contexts_matrix

corpus_fname = '/notebooks/embedding/data/tokenized/for-lsa-mecab.txt'
corpus = [sent.replace('\n', '').strip() for sent in open(corpus_fname,'r').readlines()]
input_matrix, idx2vocab = sent_to_word_contexts_matrix(
    corpus, 
    windows=3, 
    min_tf=10, 
    dynamic_weight=True, 
    verbose=True
)

cooc_svd = TruncatedSVD(n_components=100)
cooc_vecs = cooc_svd.fit_transform(input_matrix)

# use PPMI matrix 
from soynlp.word import pmi 
ppmi_matrix, _, _ = pmi(input_matrix, min_pmi=0) # if min_pmi != 0 -> PMI 
ppmi_svd = TruncatedSVD(n_components=100)
ppmi_vecs = ppmi_svd.fit_transform(ppmi_matrix)

# co-occurrence matrix 
from models.word_eval import WordEmbeddingEvaluator
model = WordEmbeddingEvaluator('data/word-embeddings/lsa/lsa-cooc.vecs', 
method='lsa', dim=100, tokenizer_name='mecab')
model.most_similar("희망", topn=5)

# ppmi matrix 
from models.word_eval import WordEmbeddingEvaluator
model = WordEmbeddingEvaluator('data/word-embeddings/lsa/lsa-pmi.vecs', 
method='lsa', dim=100, tokenizer_name='mecab')
model.most_similar('희망', topn=5)