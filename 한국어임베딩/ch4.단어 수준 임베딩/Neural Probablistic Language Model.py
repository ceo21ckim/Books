corpus_fname = '/notebooks/embedding/data/tokenized/corpus_mecab.txt'
model_fname = '/notebooks/embedding/data/word-embeddings/word2vec/word2vec'

# Word2Vec
from gensim.models import Word2Vec
corpus = [sent.strip().split(' ') for sent in open(corpus_fname, 'r').readlines()]
# sg=1 -> skip-gram / sg=0 -> CBOW / size -> window_size 
model = Word2Vec(corpus, size=100, workers=4, sg=1)
model.save(model_fname)

# vector_dim = 100 
from models.word_eval import WordEmbeddingEvaluator 
model = WordEmbeddingEvaluator('/notebooks/embedding/data/word-embeddings/word2vec/word2vec', method='word2vec', dim=100, tokenizer_name='mecab')
model.most_similar('희망', topn=5)


# FastText 
from models.word_eval import WordEmbeddingEvaluator
model = WordEmbeddingEvaluator(
    vecs_txt_fname = 'data/word-embeddings/fasttext/fasttext.vec',
    vecs_bin_fname = 'data/word-embeddings/fasttext/fasttext.bin', 
    method='fasttext', dim=100, tokenizer_name='mecab')

model.most_similar('희망', topn=5)


from models.word_eval import WordEmbeddingEvaluator 
model = WordEmbeddingEvaluator(
    vecs_txt_fname = 'data/word-embeddings/fasttext-jamo/fasttext-jamo.vec', 
    vecs_bin_fname = 'data/word-embeddings/fasttext-jamo/fasttext-jamo.bin', 
    method='fasttext-jamo', dim=100, tokenizer_name='mecab')

model.most_similar('희망')


# unknown token
model._is_in_vocabulary('서울특벌시') # False
model.get_word_vector('서울특벌시').shape
model.most_similar('서울특벌시', topn=5)


# GloVe
# GloVe는 C++ 구현체로 학습된다. 
# models/glove/build/vocab_count -min-count 5 -verbose 2 < data/tokenized/corpus_mecab.txt > data/word-embeddings/glove/glove.vocab
# models/glove/build/cooccur -memory 10.0 -vocab-file data/word-embeddings/glove/glove.vocab -verbose 2 -window-size 15 < data/tokenized/corpus_mecab.txt > data/word-embeddings/glove/glove.cooc
# models/glove/build/shuffle -memory 10.0 -verbose 2 < data/word-embeddings/glove/glove.cooc > data/word-embeddings/glove/glove.shuf
# models/glove/build/glove -save-file data/word-embeddings/glove/glove 
# -threads 4 -input-file data/word-embeddings/glove/glove.shuf -x-max 10 -iter 15 -vector-size 100 -binary 2 -vocab-file data/word-embeddings/glove/glove.vocab -verbose 2

from models.word_eval import WordEmbeddingEvaluator
model = WordEmbeddingEvaluator('data/word-embeddings/glove/glove.txt', method='glove', 
dim=100, tokenizer_name = 'mecab')
model.most_similar('희망', topn=5)


# Swivel 
# models/swivel/fastprep --input data/tokenized/corpus_mecab.txt --output_dir data/word-embeddings/swivel/swivel.data
# python models/swivel/swivel.py --input_base_path data/word-embeddings/swivel/swivel.data --output_base_path data/word-embeddings/swivel --dim 100
from models.word_eval import WordEmbeddingEvaluator
model = WordEmbeddingEvaluator('data/word-embeddings/swivel/row_embedding.tsv', 
method='swivel', dim=100, tokenizer_name='mecab')

# evaluation word similarity
# wget https://github.com/dongjun-Lee/kor2vec/raw/master/test_dataset/kor_ws353.csv -P /notebooks/embedding/data/raw

from models.word_eval import WordEmbeddingEvaluator

model_name = 'word2vec'

if model_name == 'word2vec':
    model = WordEmbeddingEvaluator(
        vecs_txt_fname = 'data/word-embeddings/word2vec/word2vec', 
        method='word2vec', dim=100, tokenizer_name='mecab'
    )

elif model_name == 'fasttext':
    model = WordEmbeddingEvaluator(
        vecs_txt_fname = 'data/word-embeddings/fasttext/fasttext.vec', 
        vecs_bin_fname = 'data/word-embeddings/fasttext/fasttext.bin', 
        method='fasttext', dim=100, tokenizer_name='mecab'
    )
elif model_name == 'glove':
    model = WordEmbeddingEvaluator(
        vecs_txt_fname = 'data/word-embeddings/glove/glove.txt', 
        method = 'glove', dim=100, tokenizer_name='mecab'
    )
elif model_name == 'swivel':
    model = WordEmbeddingEvaluator(
        vecs_txt_fname='data/word-embeddings/swivel/row_embedding.tsv', 
        method='swivel', dim=100, tokenizer_name='mecab'
    )
else:
    print('model name error!')

model.word_sim_test('data/raw/kor_ws353.csv')


model.word_analogy_test('data/raw/kor_analogy_semantic.txt', verbose=False)

# visualization 
from models.word_eval import WordEmbeddingEvaluator
model = WordEmbeddingEvaluator('/notebooks/embedding/data/word-embeddings/word2vec/word2vec', method='word2vec', dim=100, tokenizer_name='mecab')

# t-SNE
model.visualize_words('data/raw/kor_analogy_semantic.txt')

model.visualize_between_words('data/raw/kor_analogy_semantic.txt')