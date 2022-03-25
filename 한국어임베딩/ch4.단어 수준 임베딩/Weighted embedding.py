# CBoWModel
# Continuous Bag of Words Model 

# python 2.0 버전에서는 object를 입력해주어야 하지만, python 3.0 이후 버전부터는 object를 선언해주지 않아도 작동함.
import os
import numpy as np 
from gensim.models import Word2Vec
from preprocess import get_tokenizer
from collections import defaultdict

class CBoWModel(object): 
    def __init__(
        self, 
        train_fname, 
        embedding_fname,
        model_fname, 
        embedding_corpus_fname, 
        embedding_method='fasttext', 
        is_weighted=True, 
        average=False, 
        dim=100, 
        tokenizer_name='mecab'):

        # configurations
        self.make_save_path(model_fname)
        self.average = average 
        if is_weighted:
            model_full_fname = model_fname + '-weighted'
        else:
            model_full_fname = model_fname + '-original'
        self.tokenizer = get_tokenizer(tokenizer_name)
        if is_weighted:
            # weighted embeddings
            self.embeddings = self.load_or_construct_weighted_embedding(embedding_fname, embedding_method, embedding_corpus_fname)
            print('loading weighted embeddings, complete!')
        else: 
            # original embeddings 
            words, vectors = self.load_word_embeddings(embedding_fname, embedding_method)
            self.embeddings = defaultdict(list)
            for word, vector in zip(words, vectors):
                self.embeddings[word] = vector
            print('loading original embeddings, complete!')
        
        if not os.path.exists(model_full_fname):
            print('train Continuous Bag of Words model')
            self.model = self.train_model(train_fname, model_full_fname)
        else:
            print('load Continuous Bag of Words model')
            self.model = self.train_model(train_fname, model_full_fname)

    def make_save_path(self, full_path):
        if full_path[:4] == "data":
            full_path = os.path.join(os.path.abspath("."), full_path)
        model_path = '/'.join(full_path.split("/")[:-1])
        if not os.path.exists(model_path):
            os.makedirs(model_path)

    def load_or_construct_weighted_embedding(self, embedding_fname, embedding_method, embedding_corpus_fname, a=0.0001):
        dictionary = {}
        if os.path.exists(embedding_fname + "-weighted"):
            # load weighted word embeddings
            with open(embedding_fname + "-weighted", "r") as f2:
                for line in f2:
                    word, weighted_vector = line.strip().split("\u241E")
                    weighted_vector = [float(el) for el in weighted_vector.split()]
                    dictionary[word] = weighted_vector
        else:
            # load pretrained word embeddings
            words, vecs = self.load_word_embeddings(embedding_fname, embedding_method)
            # compute word frequency
            words_count, total_word_count = self.compute_word_frequency(embedding_corpus_fname)
            # construct weighted word embeddings
            with open(embedding_fname + "-weighted", "w") as f3:
                for word, vec in zip(words, vecs):
                    if word in words_count.keys():
                        word_prob = words_count[word] / total_word_count
                    else:
                        word_prob = 0.0
                    weighted_vector = (a / (word_prob + a)) * np.asarray(vec)
                    dictionary[word] = weighted_vector
                    f3.writelines(word + "\u241E" + " ".join([str(el) for el in weighted_vector]) + "\n")
        return dictionary

    def load_word_embeddings(self, vecs_fname, method):
        if method == "word2vec":
            model = Word2Vec.load(vecs_fname)
            words = model.wv.index2word
            vecs = model.wv.vectors
        else:
            words, vecs = [], []
            with open(vecs_fname, 'r', encoding='utf-8') as f1:
                if "fasttext" in method:
                    next(f1)  # skip head line
                for line in f1:
                    if method == "swivel":
                        splited_line = line.replace("\n", "").strip().split("\t")
                    else:
                        splited_line = line.replace("\n", "").strip().split(" ")
                    words.append(splited_line[0])
                    vec = [float(el) for el in splited_line[1:]]
                    vecs.append(vec)
        return words, vecs

    # 임베딩 학습 말뭉치 내 단어별 빈도 확인
    def compute_word_frequency(self, embedding_corpus_fname):
        total_count = 0 
        words_count = defaultdict(list)
        with open(embedding_corpus_fname, 'r') as f:
            for line in f:
                tokens = line.strip().split()
                for token in tokens:
                    words_count[token] += 1 
                    total_count += 1
        return words_count, total_count

    def train_model(self, train_data_fname, model_fname):
        model = {'vectors': [], 'labels': [], 'sentences': []}
        train_data = self.load_or_construct_weighted_embedding(train_data_fname)
        with open(model_fname, 'w') as f:
            for sentence, tokens, label in train_data:
                tokens = self.tokenizer.morphs(sentence)
                sentence_vector = self.get_sentence_vector(tokens)
                model['sentences'].append(sentence)
                model['vectors'].append(sentence_vector)
                model['labels'].append(label)

                str_vector=' '.join([str(el) for el in sentence_vector])
                f.writelines(sentence + '\u241E' + ' '.join(tokens) + '\u241E' + str_vector + '\u241E' + label + '\n')
        return model 

    def get_sentence_vector(self, tokens):
        vector = np.zeros(self.dim)
        for token in tokens:
            if token in self.embeddings.keys():
                vector += self.embeddings[token]

        if self.average:
            vector /= len(tokens)
        vector_norm = np.linalg.norm(vector)
        if vector_norm != 0 :
            unit_vector = vector / vector_norm
        else:
            unit_vector = np.zeros(self.dim)
            
        return unit_vector 

    # 문장 하나 예측 
    def predict(self, sentence):
        tokens = self.tokenizer.morphs(sentence)
        sentence_vector = self.get_sentence_vector(tokens)
        scores = np.dot(self.model['vectors'], sentence_vector)
        pred = self.model['labels'][np.argmax(scores)]
        return pred 

    def predict_by_batch(self, tokenized_sentences, labels):
        sentence_vectors, eval_score = [], 0
        for tokens in tokenized_sentences:
            sentence_vectors.append(self.get_sentence_vector(tokens))
        scores = np.dot(self.model['vectors'], np.array(sentence_vectors).T)
        preds = np.argmax(scores, axis=0)
        for pred, label in zip(preds, labels):
            if self.model['labels'][pred] == label:
                eval_score += 1
        return preds, eval_score 