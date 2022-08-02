import os
import warnings
from gensim.models import doc2vec
from gensim.models.doc2vec import TaggedDocument

#형태소 분석
import jpype
from konlpy.tag import Kkma

faqs = [["1", "당해년도 납입액은 수정 가능 한가요?", "네, 당해년도 납입액은 12464 화면 등록전까지 수정 가능합니다."],
            ["2", "대리인통보 대상계좌 기준은 어떻게 되나요?", "모계좌 기준 가장 최근에 개설된 계좌의 관리점에서 조회 됩니다.  의원폐쇄된 자계좌는 조회대상 계좌에서 제외됩니다. 계좌주 계좌가 사절원 계좌가 아닌 경우만 조회됩니다"],
            ["3", "등록가능 단말기수는 어떻게 되나요?", "5대까지 등록 가능입니다."],
            ["4", "모바일계좌개설 가능한 시간은 어떻게 되나요?", "08:00 ~ 20:00(영업일만 가능"],
            ["5", "미국인일때 미국납세자등록번호 작성 방법은 어떻게 되나요?", "계좌주가 미국인일 때 계좌주의 미국납세자등록번호(사회보장번호(Social Security Number), 고용주식별번호(Employer Identification Number), 개인납세자번호(Individual Taxpayer Identification Number))를 기재합니다.."]
    ]

kkma = Kkma()

def tokenize_kkma(doc):
    jpype.attachThreadToJVM()
    token_doc = ['/'.join(word) for word in kkma.pos(doc) ]
    return token_doc

print(tokenize_kkma(faqs[0][1]))

# 리스트에서 각 문장부분 토큰화
token_faqs = [(tokenize_kkma(row[1]), row[0]) for row in faqs]

# Doc2Vec에서 사용하는 태그문서형으로 변경
tagged_faqs = [TaggedDocument(d, [c]) for d, c in token_faqs]


print(tagged_faqs)

# make model
import multiprocessing
cores = multiprocessing.cpu_count()
d2v_faqs = doc2vec.Doc2Vec(
                                vector_size=50, 
                                alpha=0.025,
                                min_alpha=0.025,
                                hs=1,
                                negative=0,
                                dm=0,
                                dbow_words = 1,
                                min_count = 1,
                                workers = cores,
                                seed=0
                                )
d2v_faqs.build_vocab(tagged_faqs)

# train document vectors
for epoch in range(10):
    d2v_faqs.train(tagged_faqs,
                                total_examples = d2v_faqs.corpus_count,
                                epochs = d2v_faqs.epochs
                                )
    d2v_faqs.alpha -= 0.0025 # decrease the learning rate
    d2v_faqs.min_alpha = d2v_faqs.alpha # fix the learning rate, no decay


predict_vector = d2v_faqs.infer_vector(['당해년도 납입액은 수정 가능 한가요?'])

