import argparse
import pickle
from sklearn.utils import class_weight
from nltk.corpus import stopwords
import nltk
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
# from utils import clean_str, clean_str_simple_version, show_statisctic, clean_document
import sys
from nltk import tokenize
import collections
from collections import Counter
import random
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation



def display_topics(model, feature_names, no_top_words):
    keywords_dic = {}
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        klist = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
        print(" ".join(klist))
        for k in klist:
            if not k in keywords_dic:
                keywords_dic[k] = []
            keywords_dic[k].append(topic_idx)
    return keywords_dic


def Generate_LDA(dataset):
    doc_content_list = []
    # doc_sentence_list = []
    f = pickle.load(open('./datasets/' + args.dataset + '/train.txt', 'rb'))
    for line in f[0]:
        # doc_content_list.append(line)
        line = [str(val) for val in line]
        doc_content_list.append(' '.join(line))

    vectorizer = CountVectorizer()
    vector = vectorizer.fit_transform(doc_content_list)
    feature_names = vectorizer.get_feature_names()
    lda = LatentDirichletAllocation(n_components=args.topics, learning_method='online', learning_offset=50.,
                                    random_state=0).fit(vector)

    keywords_dic = display_topics(lda, feature_names, args.topn)
    print(len(keywords_dic))
    print(keywords_dic)

    # open('datasets/' + args.dataset + '/_LDA.p', 'wb')
    pickle.dump(keywords_dic, open('datasets/' + args.dataset + '/_LDA.p', 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='Nowplaying', help='Diginetica/Nowplaying/Tmall/yoochoose1_4/yoochoose1_64')
    parser.add_argument('--topn', type=int, default= 10, help='top n keywords')
    parser.add_argument('--topics', type=int,default= 50, help='number of topics')
    args = parser.parse_args()
    print(args)

    Generate_LDA(args.dataset)

#
# from sklearn.feature_extraction.text import CountVectorizer
#
# texts=["dog cat fish","dog cat cat","fish bird", 'bird'] # “dog cat fish” 为输入列表元素,即代表一个文章的字符串
# cv = CountVectorizer()#创建词袋数据结构
# cv_fit=cv.fit_transform(texts)
# #上述代码等价于下面两行
# #cv.fit(texts)
# #cv_fit=cv.transform(texts)
#
# print(cv.get_feature_names())    #['bird', 'cat', 'dog', 'fish'] 列表形式呈现文章生成的词典
#
# print(cv.vocabulary_	)              # {‘dog’:2,'cat':1,'fish':3,'bird':0} 字典形式呈现，key：词，value:词频
#
# print(cv_fit)
# # （0,3） 1   第0个列表元素，**词典中索引为3的元素**， 词频
# #（0,1）1
# #（0,2）1
# #（1,1）2
# #（1,2）1
# #（2,0）1
# #（2,3）1
# #（3,0）1
#
# print(cv_fit.toarray()) #.toarray() 是将结果转化为稀疏矩阵矩阵的表示方式；
# #[[0 1 1 1]
# # [0 2 1 0]
# # [1 0 0 1]
# # [1 0 0 0]]
#
# print(cv_fit.toarray().sum(axis=0))  #每个词在所有文档中的词频
# #[2 3 2 2]
#
