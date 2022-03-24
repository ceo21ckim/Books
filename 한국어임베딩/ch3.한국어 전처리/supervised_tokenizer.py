from gensim.corpora import WikiCorpus, Dictionary 
from gensim.utils import to_unicode
from konlpy.tag import Mecab

# supervised_nlp
from konlpy.tag import Okt, Komoran, Mecab, Hannanum, Kkma 

def get_tokenizer(tokenizer_name):
    if tokenizer_name == 'komoran':
        tokenizer = Komoran()
    elif tokenizer_name == 'okt':
        tokenizer = Okt()
    elif tokenizer_name == 'mecab':
        tokenizer = Mecab()
    elif tokenizer_name == 'hannanum':
        tokenizer = Hannanum()
    elif tokenizer_name == 'kkma':
        tokenizer = Kkma()
    else:
        print('please input tokenizer_name ( e.g. komoran, okt, mecab, hannanum, kkma )')
        return None
    return tokenizer 

tokenizer = get_tokenizer('komoran')
tokenizer.morphs('아버지가방에들어가신다')
tokenizer.pos('아버지가방에들어가신다')


# Khaiii
# 2018년 카카오에서 개발한 한국어 형태소 분석기이며 세종코퍼스에 CNN을 족용해 학습함.
from khaiii import KhaiiiApi
tokenizer = KhaiiiApi()

data = tokenizer.analyze('아버지가방에들어가신다')
tokens = []
pos_tags = []
for word in data:
    tokens.extend([str(m).split('/')[0] for m in word.morphs])
    pos_tags.extend([ tuple(str(m).split('/')) for m in word.morphs])


# Mecab 
# cannot use konlpy.tag.Mecab in Window...
from konlpy.tag import Mecab
tokenizer = Mecab()
tokenizer.morphs('가우스전자 텔레비전 정말 좋네요')

# 가우스전자 -> 가우스, 전자
# 분할되는 것을 막기 위해서는 mecab-user-dic.csv에 단어를 추가해주면 된다.
# bash preprocess.sh mecab-user-dic
