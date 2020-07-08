# -*- coding: utf-8 -*-
"""
@create_time : 2020-04-07 16:40
@environment : python 3.6
@author      : zhangjiwen
@file        : query.py
"""
import numpy as np
from time import time

start = time()
print("Now loading MapReduce outcome...")
query_dict = {} # word->[doc_names : freq]
doc_count = {} # 文档次数统计

with open("./part-00000", "r", encoding="utf-8") as f:
    lines = f.readlines()

    for line in lines:
        word, rest = line.split("\t")
        titles = rest.split("<SEP>")

        word2doc = {}
        for title in titles:
            doc, freq = title.strip().split("@")
            word2doc[doc] = int(freq)
            doc_count[doc] = int(freq) + doc_count.get(doc, 0)
        query_dict[word] = word2doc

doc_list = list(doc_count.keys())
doc_idx = {doc:i for i, doc in enumerate(doc_list)}

word_list = list(query_dict.keys())
word_idx = {word:i for i, word in enumerate(word_list)}

M, N = len(doc_list), len(word_list)
print("Finished! Document dict and Word dict has been built...")
print("Time cost: {:.2f}\n".format(time()-start))


def construct_TFIDF(M, N, doc_dict, doc_idx, word_dict, word_idx):
    tf_shape = (M, N)
    idf_shape = (1, N)

    def TF_matrix():
        m = np.zeros(tf_shape, dtype=float)

        for word in word_dict:
            for doc in word_dict[word]:
                m[doc_idx[doc], word_idx[word]] = word_dict[word][doc] / doc_dict[doc]
        
        return m
    
    def IDF_matrix():
        m = np.zeros(idf_shape, dtype=float)

        for word in word_dict:
            c = len(list(word_dict[word].keys()))
            m[0, word_idx[word]] = np.log(N/c)
        
        m[m < 1e-6] = 1
        
        return np.repeat(m, M, axis=0) # shape(M, N)

    return TF_matrix() * IDF_matrix()


def input2vec(word_list, word_idx, shape):
    out = np.zeros(shape, dtype=float)

    for word in word_list:
        if word in word_idx:
            out[word_idx[word]] = 1
    
    return out


def compute_Cosine(query_vec, matrix):
    assert matrix.shape[1] == query_vec.shape[0]

    inner_product = np.dot(matrix, query_vec) # shape(M,1)
    norm1 = np.linalg.norm(matrix, ord=2, axis=1, keepdims=True) * np.linalg.norm(query_vec)

    cosine_similarity = inner_product / norm1
    return cosine_similarity.transpose()


if __name__ == "__main__":
    # get the query
    string = input("请输入查询词：...（按空格划分）")
    word_list = string.split(" ")
    query_vec = input2vec(word_list, word_idx, shape=(N, 1))

    # compute tf-idf
    tfidf_matrix = construct_TFIDF(M, N, doc_count, doc_idx, query_dict, word_idx)

    # compute cosine relevance
    similarity_score = compute_Cosine(query_vec, tfidf_matrix) #shape(1, M)
    rank = sorted([(s, i) for i, s in enumerate(similarity_score.tolist()[0])], key = lambda x: x[0],reverse =True)

    print("Query finished... We have following outcome!\n")
    print(string)
    leng = len(string)

    early_break = 5
    for score, idx in rank:
        if score > 0.0:
            title = doc_list[idx]
            if len(word_list) == 1:
                title = title + "@" + str(query_dict[word_list[0]][title])
            print(" " * leng + "\t", title)
            
            if early_break is not None:
                early_break -= 1
                if early_break <= 0:break




    