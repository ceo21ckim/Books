# soynlp
# soynmlp의 경우 기존의 supervised와는 달리 별도의 학습 과정을 거쳐야 한다.
from soynlp.word import WordExtractor 

corpus_fname = '/notebooks/embedding/data/processed/processed_ratings.txt' # 데이터 위치
model_fname = '/notebooks/embedding/data/processed/soyword.model' # 모델 저장 위치

sentences = [sent.strip() for sent in open(corpus_fname, 'r').readlines()]
word_extractor = WordExtractor(
    min_frequency=100, 
    min_cohesion_forward = 0.05, # 응집 확률이 높을 때 형태소로 취급.
    min_right_branching_entropy=0.0) # 높을 때 문자열을 형태소로 취급.

word_extractor.train(sentences)
word_extractor.save(model_fname)

import math 
from soynlp.word import WordExtractor 
from soynlp.tokenizer import LTokenizer 

model_fname = '/notebooks/embedding/data/processed/soyword.model' # 모델 불러오기 

word_extractor = WordExtractor(
    min_frequency=100,
    min_cohesion_forward=0.05, 
    min_right_branching_entropy=0.0)

word_extractor.load(model_fname)

scores = word_extractor.word_scores()

# vocab의 key에 대한 점수를 산출하기 위한 작업.
scores = {key:(scores[key].cohesion_forward * math.exp(scores[key].right_branching_entropy)) for key in scores.keys()}
tokenizer = LTokenizer(scores=scores)
tokens = tokenizer.tokenize('애비는 종이었다')
tokens


# sentencepiece 
# sentencepiece는 2018년 구글에서 공개한 비지도 학습 기반 형태소 분석기.
# BPE (Byte Pair Encoding) 기법을 지원해줌.
import sentencepiece as spm
train = """--input=/notebooks/embedding/data/processed/processed_wiki_ko.txt \
    --model_prefix=sentpiece \
        --vocab_size=32000 \
            --model_type=bpe --character_coverage=0.9995"""

spm.SentencePieceTrainer.Train(train)


from models.bert.tokenization import FullTokenizer 
vocab_fname = '/notebooks/embedding/data/processed/bert.vocab'
tokenizer = FullTokenizer(vocab_file=vocab_fname, do_lower_case=False)

tokenizer.tokenize('집에좀 가자')


# 띄어쓰기 교정 
from soyspacing.countbase import CountSpace 

corpus_fname = '/notebooks/embedding/data/processed/processed_ratings.txt' # 데이터 위치
model_fname = '/notebooks/embedding/data/processed/space-correct.model' # 저장 위치

# training module
model = CountSpace()
model.train(corpus_fname)
model.save_model(model_fname, json_format=False)

# eval
from soyspacing.countbase import CountSpace 

model_fname = '/notebooks/embedding/data/processed/space-correct.model'
model = CountSpace()
model.load_model(model_fname, json_format=False)
model.correct('어릴때보고 지금다시봐도 재밌어요')
# ('어릴때 보고 지금 다시봐도 재밌어요', [0, 0, 1, 0, 1, 0, 1, 0, None, 0, 1, 0, 0, 0, 1])


